from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
FEATURE_CSV = ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
CLINICAL_XLSX = ROOT / "final__2026" / "04_clinical_data" / "ClinicalDemogData_COFL.xlsx"
OUT_DIR = ROOT / "final__2026" / "11_drop_stride_regularity_check"

TARGET = "DGI_le19_or_TUG_ge12"
EXCLUDED_SUBJECTS = {"CO024", "FL020"}
N_REPEATS = 100
RANDOM_SEED = 20260713

FEATURE_SETS = {
    "current_4": [
        "v_amp_pool_median",
        "ml_amp_pool_iqr",
        "base_v_stride_regularity",
        "roll_amp_pool_iqr",
    ],
    "drop_base_v_stride_regularity_3": [
        "v_amp_pool_median",
        "ml_amp_pool_iqr",
        "roll_amp_pool_iqr",
    ],
    "replace_with_v_amp_stride_regularity_4": [
        "v_amp_pool_median",
        "ml_amp_pool_iqr",
        "v_amp_pool_stride_regularity",
        "roll_amp_pool_iqr",
    ],
    "replace_with_base_ml_stride_regularity_4": [
        "v_amp_pool_median",
        "ml_amp_pool_iqr",
        "base_ml_stride_regularity",
        "roll_amp_pool_iqr",
    ],
}


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
    all_features = sorted({feature for features in FEATURE_SETS.values() for feature in features})
    features = pd.read_csv(FEATURE_CSV)
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    labels = load_labels()
    merged = features.merge(labels, on="subject_id", how="inner")
    merged = merged[~merged["subject_id"].isin(EXCLUDED_SUBJECTS)].copy()
    merged = merged[merged[TARGET].notna()].copy()

    rows = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "target": int(group[TARGET].iloc[0])}
        for feature in all_features:
            row[feature] = float(pd.to_numeric(group[feature], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=FEATURE_SETS["current_4"], how="any").reset_index(drop=True)


def make_pipeline(random_state: int) -> Pipeline:
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


def choose_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_threshold = float(np.min(prob))
    best_score = -np.inf
    best_sens = -np.inf
    for threshold in np.unique(prob):
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        score = sens + spec - 1.0
        if score > best_score or (score == best_score and sens > best_sens):
            best_score = score
            best_sens = sens
            best_threshold = float(threshold)
    return best_threshold


def inner_threshold(table: pd.DataFrame, train_idx: np.ndarray, features: list[str], repeat: int, fold: int) -> float:
    train = table.iloc[train_idx].reset_index(drop=True)
    y = train["target"].to_numpy(dtype=int)
    n_splits = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=510000 + repeat * 100 + fold)
    prob = np.full(len(train), np.nan)
    for inner_fold, (tr, va) in enumerate(cv.split(train[features], y)):
        model = make_pipeline(610000 + repeat * 1000 + fold * 10 + inner_fold)
        model.fit(train.iloc[tr][features], y[tr])
        prob[va] = model.predict_proba(train.iloc[va][features])[:, 1]
    return choose_youden_threshold(y, prob)


def pooled_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in pred_df.groupby("feature_set", sort=True):
        y = group["target"].to_numpy(dtype=int)
        prob = group["probability"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "feature_set": name,
                "features": "|".join(FEATURE_SETS[name]),
                "n_predictions": int(len(group)),
                "n_subjects_per_repeat": int(group.groupby("repeat")["subject_id"].nunique().median()),
                "n_repeats": int(group["repeat"].nunique()),
                "auc": float(roc_auc_score(y, prob)),
                "accuracy": float(accuracy_score(y, pred)),
                "sensitivity": float(recall_score(y, pred, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "f1": float(f1_score(y, pred, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_subject_table()
    y = table["target"].to_numpy(dtype=int)
    pred_rows = []

    total = len(FEATURE_SETS) * N_REPEATS * 5
    done = 0
    for name, feature_list in FEATURE_SETS.items():
        for repeat in range(N_REPEATS):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv.split(table, y)):
                threshold = inner_threshold(table, train_idx, feature_list, repeat, fold)
                model = make_pipeline(710000 + repeat * 1000 + fold)
                model.fit(table.iloc[train_idx][feature_list], y[train_idx])
                prob = model.predict_proba(table.iloc[test_idx][feature_list])[:, 1]
                for subject_id, target, p in zip(table.iloc[test_idx]["subject_id"], y[test_idx], prob):
                    pred_rows.append(
                        {
                            "feature_set": name,
                            "repeat": repeat,
                            "fold": fold,
                            "subject_id": subject_id,
                            "target": int(target),
                            "probability": float(p),
                            "threshold": float(threshold),
                            "prediction": int(p >= threshold),
                        }
                    )
                done += 1
                if done % 100 == 0:
                    print(f"completed {done}/{total} folds")

    pred_df = pd.DataFrame(pred_rows)
    metrics_df = pooled_metrics(pred_df)

    pred_df.to_csv(OUT_DIR / "drop_stride_regularity_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT_DIR / "drop_stride_regularity_metrics.csv", index=False, encoding="utf-8-sig")
    notes = {
        "analysis": "Nested-style repeated CV comparison after dropping/replacing base_v_stride_regularity.",
        "n_subjects": int(table.shape[0]),
        "positive": int(table["target"].sum()),
        "negative": int((1 - table["target"]).sum()),
        "feature_sets": FEATURE_SETS,
        "threshold": "Inner OOF Youden selected within each outer training fold.",
    }
    (OUT_DIR / "drop_stride_regularity_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
