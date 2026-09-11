from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURE_CSV = ROOT / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
CLINICAL_XLSX = ROOT / "04_clinical_data" / "ClinicalDemogData_COFL.xlsx"
OUT_DIR = ROOT / "02_model"

FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]
EXCLUDED_SUBJECTS = ["CO024", "FL020"]
TARGET = "DGI_le19_or_TUG_ge12"
N_REPEATS = 100
N_JOBS = -1  # -1 uses all available CPU cores via joblib/loky.
JOBLIB_VERBOSE = 10


def normalize_subject_id(value: object) -> str:
    return str(value).strip().replace("-", "").upper()


def load_labels() -> pd.DataFrame:
    frames = []
    for sheet_name in ["Controls", "Fallers"]:
        frame = pd.read_excel(CLINICAL_XLSX, sheet_name=sheet_name)
        frame["subject_id"] = frame["#"].map(normalize_subject_id)
        frames.append(frame[["subject_id", "DGI", "TUG"]])
    labels = pd.concat(frames, ignore_index=True)
    labels["DGI"] = pd.to_numeric(labels["DGI"], errors="coerce")
    labels["TUG"] = pd.to_numeric(labels["TUG"], errors="coerce")
    labels[TARGET] = ((labels["DGI"] <= 19) | (labels["TUG"] >= 12)).astype("Int64")
    return labels.dropna(subset=[TARGET]).copy()


def load_subject_table() -> pd.DataFrame:
    features = pd.read_csv(FEATURE_CSV)
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    labels = load_labels()
    merged = features.merge(labels, on="subject_id", how="inner")
    merged = merged[~merged["subject_id"].isin(EXCLUDED_SUBJECTS)].copy()

    rows = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "target": int(group[TARGET].iloc[0])}
        for feature in FEATURES:
            row[feature] = float(pd.to_numeric(group[feature], errors="coerce").median())
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.dropna(subset=FEATURES, how="any").reset_index(drop=True)
    return table


def make_pipeline(random_state: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def decision_scores(model: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x_frame)[:, 1]


def choose_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_threshold = float(np.min(prob))
    best_score = -np.inf
    best_sensitivity = -np.inf
    for threshold in np.unique(prob):
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        score = sensitivity + specificity - 1.0
        if score > best_score or (score == best_score and sensitivity > best_sensitivity):
            best_score = score
            best_threshold = float(threshold)
            best_sensitivity = sensitivity
    return best_threshold


def clf_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def inner_oof_threshold(
    table: pd.DataFrame,
    outer_train_idx: np.ndarray,
    repeat: int,
    fold: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_table = table.iloc[outer_train_idx].reset_index(drop=True)
    y_train_outer = train_table["target"].to_numpy(dtype=int)
    inner_splits = min(5, int(np.bincount(y_train_outer).min()))
    if inner_splits < 2:
        raise ValueError("Not enough classes for inner CV threshold selection.")

    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=510000 + repeat * 100 + fold,
    )
    inner_prob = np.full(len(train_table), np.nan, dtype=float)
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_table[FEATURES], y_train_outer)):
        model = make_pipeline(random_state=610000 + repeat * 1000 + fold * 10 + inner_fold)
        model.fit(train_table.iloc[inner_train_idx][FEATURES], y_train_outer[inner_train_idx])
        inner_prob[inner_val_idx] = decision_scores(model, train_table.iloc[inner_val_idx][FEATURES])

    if np.isnan(inner_prob).any():
        raise RuntimeError("Inner OOF probabilities contain NaN.")

    threshold = choose_youden_threshold(y_train_outer, inner_prob)
    inner_pred = (inner_prob >= threshold).astype(int)
    return threshold, y_train_outer, inner_prob, inner_pred


def run_outer_fold(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
    repeat: int,
    fold: int,
) -> tuple[list[dict], list[dict]]:
    threshold, inner_y, inner_prob, inner_pred = inner_oof_threshold(table, train_idx, repeat, fold)

    x_train = table.iloc[train_idx][FEATURES]
    y_train = table.iloc[train_idx]["target"].to_numpy(dtype=int)
    x_test = table.iloc[test_idx][FEATURES]
    y_test = table.iloc[test_idx]["target"].to_numpy(dtype=int)

    final_model = make_pipeline(random_state=710000 + repeat * 100 + fold)
    final_model.fit(x_train, y_train)
    train_prob = decision_scores(final_model, x_train)
    test_prob = decision_scores(final_model, x_test)

    rows = []
    for split, y, prob, pred in [
        ("inner_oof_train", inner_y, inner_prob, inner_pred),
        ("train_apparent", y_train, train_prob, (train_prob >= threshold).astype(int)),
        ("test", y_test, test_prob, (test_prob >= threshold).astype(int)),
    ]:
        row = clf_metrics(y, prob, pred)
        row.update(
            {
                "scheme": scheme,
                "repeat": repeat,
                "fold": fold,
                "split": split,
                "threshold_strategy": "nested_inner_oof_youden",
                "threshold": float(threshold),
            }
        )
        rows.append(row)

    predictions = []
    for subject_id, target, p_value, pred_value in zip(
        table.iloc[test_idx]["subject_id"],
        y_test,
        test_prob,
        (test_prob >= threshold).astype(int),
    ):
        predictions.append(
            {
                "strategy": "domain4_fixed",
                "scheme": scheme,
                "repeat": repeat,
                "fold": fold,
                "subject_id": subject_id,
                "target": int(target),
                "probability": float(p_value),
                "threshold": float(threshold),
                "prediction": int(pred_value),
                "threshold_strategy": "nested_inner_oof_youden",
            }
        )
    return rows, predictions


def pooled_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, group in predictions.groupby("scheme", dropna=False):
        pooled = (
            group.groupby(["subject_id", "target"], as_index=False)
            .agg(
                probability=("probability", "mean"),
                threshold=("threshold", "mean"),
                positive_vote_rate=("prediction", "mean"),
                n_predictions=("prediction", "size"),
            )
        )
        for decision_name, pred in [
            ("mean_prob_vs_mean_threshold", (pooled["probability"] >= pooled["threshold"]).astype(int)),
            ("majority_vote", (pooled["positive_vote_rate"] >= 0.5).astype(int)),
        ]:
            row = clf_metrics(
                pooled["target"].to_numpy(dtype=int),
                pooled["probability"].to_numpy(dtype=float),
                pred.to_numpy(dtype=int),
            )
            row.update(
                {
                    "scheme": scheme,
                    "threshold_strategy": "nested_inner_oof_youden",
                    "pooled_decision": decision_name,
                    "n_subjects": int(len(pooled)),
                    "mean_predictions_per_subject": float(pooled["n_predictions"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["scheme", "split"], dropna=False)[
            ["auc", "accuracy", "sensitivity", "specificity", "f1"]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in col if part) for col in summary.columns.to_flat_index()]
    return summary


def train_final_nested_artifact(table: pd.DataFrame) -> tuple[dict, dict]:
    y = table["target"].to_numpy(dtype=int)
    full_idx = np.arange(len(table))
    threshold, inner_y, inner_prob, inner_pred = inner_oof_threshold(table, full_idx, repeat=999, fold=0)
    model = make_pipeline(random_state=0)
    model.fit(table[FEATURES], y)
    prob = decision_scores(model, table[FEATURES])
    pred = (prob >= threshold).astype(int)
    artifact = {
        "pipeline": model,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_strategy": "nested_inner_oof_youden",
        "decision_rule": "probability >= threshold -> motor_impairment_possible",
    }
    apparent = clf_metrics(y, prob, pred)
    apparent["inner_oof_threshold_auc"] = clf_metrics(inner_y, inner_prob, inner_pred)["auc"]
    return artifact, apparent


def run_one_repeat(table: pd.DataFrame, y: np.ndarray, repeat: int) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    prediction_rows = []

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=91000 + repeat)
    for fold, (train_idx, test_idx) in enumerate(cv5.split(table[FEATURES], y)):
        m, p = run_outer_fold(table, train_idx, test_idx, "A_5fold_x100", repeat, fold)
        metric_rows.extend(m)
        prediction_rows.extend(p)

    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=92000 + repeat)
    for fold, (train_idx, test_idx) in enumerate(cv3.split(table[FEATURES], y)):
        m, p = run_outer_fold(table, train_idx, test_idx, "B_3fold_x100", repeat, fold)
        metric_rows.extend(m)
        prediction_rows.extend(p)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=93000 + repeat)
    for fold, (train_idx, test_idx) in enumerate(split.split(table[FEATURES], y)):
        m, p = run_outer_fold(table, train_idx, test_idx, "C_repeated_80_20_x100", repeat, fold)
        metric_rows.extend(m)
        prediction_rows.extend(p)

    return metric_rows, prediction_rows


def run_one_loso_fold(table: pd.DataFrame, y: np.ndarray, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[list[dict], list[dict]]:
    return run_outer_fold(table, train_idx, test_idx, "E_LOSO_pooled", 0, fold)


def main() -> None:
    table = load_subject_table()
    y = table["target"].to_numpy(dtype=int)

    print(f"Nested CV started: subjects={len(table)}, positives={int(np.sum(y == 1))}, negatives={int(np.sum(y == 0))}")
    print(f"CPU cores visible={os.cpu_count()}, joblib n_jobs={N_JOBS}, repeats={N_REPEATS}")

    repeat_results = joblib.Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
        joblib.delayed(run_one_repeat)(table, y, repeat)
        for repeat in range(N_REPEATS)
    )

    metric_rows = []
    prediction_rows = []
    for m, p in repeat_results:
        metric_rows.extend(m)
        prediction_rows.extend(p)

    loo = LeaveOneOut()
    loso_splits = list(loo.split(table[FEATURES], y))
    loso_results = joblib.Parallel(n_jobs=N_JOBS, verbose=JOBLIB_VERBOSE)(
        joblib.delayed(run_one_loso_fold)(table, y, fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(loso_splits)
    )
    for m, p in loso_results:
        metric_rows.extend(m)
        prediction_rows.extend(p)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_metrics(metrics)
    pooled = pooled_subject_metrics(predictions)

    artifact, apparent_metrics = train_final_nested_artifact(table)
    joblib.dump(artifact, OUT_DIR / "final_motor_domain4_labwalks10_logistic_C0p5_nested_youden.joblib")

    metadata = {
        "target": TARGET,
        "clinical_label_rule": "DGI <= 19 OR TUG >= 12",
        "data": "Labwalks 10sec lab walking",
        "model": "LogisticRegression L2 class_weight=balanced C=0.5",
        "threshold_strategy": "nested_inner_oof_youden",
        "threshold": artifact["threshold"],
        "n_subjects": int(len(table)),
        "positive": int(np.sum(y == 1)),
        "negative": int(np.sum(y == 0)),
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "excluded_reason": "base_v_stride_regularity all NaN across all segments",
        "features": FEATURES,
        "cv_metrics": (
            pooled[
                (pooled["scheme"] == "A_5fold_x100")
                & (pooled["pooled_decision"] == "mean_prob_vs_mean_threshold")
            ][["auc", "sensitivity", "specificity", "f1"]]
            .iloc[0]
            .to_dict()
        ),
        "apparent_train_metrics": apparent_metrics,
    }
    metadata["cv_metrics"] = {
        "test_auc": float(metadata["cv_metrics"]["auc"]),
        "test_sensitivity": float(metadata["cv_metrics"]["sensitivity"]),
        "test_specificity": float(metadata["cv_metrics"]["specificity"]),
        "test_f1": float(metadata["cv_metrics"]["f1"]),
    }

    metrics.to_csv(OUT_DIR / "domain4_nested_full_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "domain4_nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "domain4_nested_full_validation_summary.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUT_DIR / "domain4_nested_pooled_subject_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "final_motor_domain4_labwalks10_logistic_C0p5_nested_youden_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nNested pooled subject metrics:")
    print(pooled.to_string(index=False))
    print("\nNested fold summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved nested outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
