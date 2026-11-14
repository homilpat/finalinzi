from __future__ import annotations

import json
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
    merged = merged.dropna(subset=FEATURES, how="all")

    subject_rows = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "target": int(group[TARGET].iloc[0])}
        for feature in FEATURES:
            row[feature] = float(pd.to_numeric(group[feature], errors="coerce").median())
        subject_rows.append(row)

    subject_table = pd.DataFrame(subject_rows)
    subject_table = subject_table.dropna(subset=FEATURES, how="any").reset_index(drop=True)
    return subject_table


def make_pipeline() -> Pipeline:
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
                    random_state=0,
                ),
            ),
        ]
    )


def choose_thresholds(y_train: np.ndarray, train_prob: np.ndarray) -> dict[str, float]:
    neg = train_prob[y_train == 0]
    pos = train_prob[y_train == 1]
    thresholds = {
        "sens80_p20": float(np.quantile(pos, 0.20)),
        "spec80_p80": float(np.quantile(neg, 0.80)),
    }

    best_youden = float(train_prob.min())
    best_youden_score = -np.inf
    best_sens90 = float(train_prob.min())
    best_sens90_spec = -np.inf
    best_sens90_sens = -np.inf

    for threshold in np.unique(train_prob):
        pred = (train_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        youden = sensitivity + specificity - 1.0

        if youden > best_youden_score:
            best_youden_score = youden
            best_youden = float(threshold)
        if sensitivity >= 0.90 and (
            specificity > best_sens90_spec
            or (specificity == best_sens90_spec and sensitivity > best_sens90_sens)
        ):
            best_sens90 = float(threshold)
            best_sens90_spec = specificity
            best_sens90_sens = sensitivity

    thresholds["youden"] = best_youden
    thresholds["sens90_maxspec"] = best_sens90
    return thresholds


def decision_scores(model: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x_frame)[:, 1]


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


def run_fold(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
    repeat: int,
    fold: int,
) -> tuple[list[dict], list[dict]]:
    x_train = table.iloc[train_idx][FEATURES]
    y_train = table.iloc[train_idx]["target"].to_numpy(dtype=int)
    x_test = table.iloc[test_idx][FEATURES]
    y_test = table.iloc[test_idx]["target"].to_numpy(dtype=int)

    model = make_pipeline()
    model.fit(x_train, y_train)
    train_prob = decision_scores(model, x_train)
    test_prob = decision_scores(model, x_test)
    thresholds = choose_thresholds(y_train, train_prob)

    metric_rows = []
    prediction_rows = []
    for strategy, threshold in thresholds.items():
        for split, split_idx, y, prob in [
            ("train", train_idx, y_train, train_prob),
            ("test", test_idx, y_test, test_prob),
        ]:
            pred = (prob >= threshold).astype(int)
            row = clf_metrics(y, prob, pred)
            row.update(
                {
                    "scheme": scheme,
                    "repeat": repeat,
                    "fold": fold,
                    "split": split,
                    "threshold_strategy": strategy,
                    "threshold": float(threshold),
                }
            )
            metric_rows.append(row)

            if split == "test":
                for subject_id, target, p_value, pred_value in zip(
                    table.iloc[split_idx]["subject_id"],
                    y,
                    prob,
                    pred,
                ):
                    prediction_rows.append(
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
                            "threshold_strategy": strategy,
                        }
                    )
    return metric_rows, prediction_rows


def pooled_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, strategy), group in predictions.groupby(["scheme", "threshold_strategy"], dropna=False):
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
                    "threshold_strategy": strategy,
                    "pooled_decision": decision_name,
                    "n_subjects": int(len(pooled)),
                    "mean_predictions_per_subject": float(pooled["n_predictions"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["scheme", "threshold_strategy", "split"], dropna=False)[
            ["auc", "accuracy", "sensitivity", "specificity", "f1"]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in col if part) for col in summary.columns.to_flat_index()]
    return summary


def fit_final_model(table: pd.DataFrame) -> tuple[dict, dict]:
    x = table[FEATURES]
    y = table["target"].to_numpy(dtype=int)
    model = make_pipeline()
    model.fit(x, y)
    prob = decision_scores(model, x)
    threshold = choose_thresholds(y, prob)["youden"]
    pred = (prob >= threshold).astype(int)
    train_metrics = clf_metrics(y, prob, pred)
    artifact = {
        "pipeline": model,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_strategy": "youden",
        "decision_rule": "probability >= threshold -> motor_impairment_possible",
    }
    return artifact, train_metrics


def main() -> None:
    table = load_subject_table()
    y = table["target"].to_numpy(dtype=int)

    metric_rows = []
    prediction_rows = []
    for repeat in range(100):
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=91000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv5.split(table[FEATURES], y)):
            m, p = run_fold(table, train_idx, test_idx, "A_5fold_x100", repeat, fold)
            metric_rows.extend(m)
            prediction_rows.extend(p)

        cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=92000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv3.split(table[FEATURES], y)):
            m, p = run_fold(table, train_idx, test_idx, "B_3fold_x100", repeat, fold)
            metric_rows.extend(m)
            prediction_rows.extend(p)

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=93000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(split.split(table[FEATURES], y)):
            m, p = run_fold(table, train_idx, test_idx, "C_repeated_80_20_x100", repeat, fold)
            metric_rows.extend(m)
            prediction_rows.extend(p)

        print(f"completed repeated CV {repeat + 1}/100")

    loo = LeaveOneOut()
    for fold, (train_idx, test_idx) in enumerate(loo.split(table[FEATURES], y)):
        m, p = run_fold(table, train_idx, test_idx, "E_LOSO_pooled", 0, fold)
        metric_rows.extend(m)
        prediction_rows.extend(p)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = write_summary(metrics)
    pooled = pooled_subject_metrics(predictions)

    artifact, apparent_metrics = fit_final_model(table)
    joblib.dump(artifact, OUT_DIR / "final_motor_domain4_labwalks10_logistic_C0p5.joblib")

    metadata = {
        "target": TARGET,
        "clinical_label_rule": "DGI <= 19 OR TUG >= 12",
        "data": "Labwalks 10sec lab walking",
        "model": "LogisticRegression L2 class_weight=balanced C=0.5",
        "threshold_strategy": "youden",
        "threshold": artifact["threshold"],
        "n_subjects": int(len(table)),
        "positive": int(np.sum(y == 1)),
        "negative": int(np.sum(y == 0)),
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "excluded_reason": "base_v_stride_regularity all NaN across all segments",
        "features": FEATURES,
        "feature_domains": {
            "v_amp_pool_median": "보행 활력 - 수직 진폭 중앙값",
            "ml_amp_pool_iqr": "좌우 안정성 - 좌우 진폭 변동성",
            "base_v_stride_regularity": "리듬 규칙성 - 수직축 stride 규칙성",
            "roll_amp_pool_iqr": "몸통 회전 안정성 - roll 진폭 변동성",
        },
        "cv_metrics": (
            pooled[
                (pooled["scheme"] == "A_5fold_x100")
                & (pooled["threshold_strategy"] == "youden")
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

    metrics.to_csv(OUT_DIR / "domain4_full_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "domain4_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "domain4_full_validation_summary.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUT_DIR / "domain4_pooled_subject_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "final_motor_domain4_labwalks10_logistic_C0p5_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSubject table:")
    print(table[["subject_id", "target"] + FEATURES].to_string(index=False))
    print("\nPooled subject metrics, youden:")
    print(pooled[pooled["threshold_strategy"] == "youden"].to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
