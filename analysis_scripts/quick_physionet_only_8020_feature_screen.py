from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "analysis_outputs" / "axis_aligned_gait_model" / "axis_aligned_best10_subject_table.csv"
OUT_DIR = ROOT / "analysis_outputs" / "quick_physionet_only_8020_feature_screen"
MODEL_PATH = OUT_DIR / "physionet_only_quick_best.joblib"

FEATURES = [
    "v_acf_stride_peak",
    "v_acf_stride_peak_width_sec",
    "ap_acf_stride_peak",
    "ap_acf_stride_peak_width_sec",
    "ap_spec_entropy",
]
REPEATS = 300
TEST_SIZE = 0.20
SEED = 20260721


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_j = 0.5, -np.inf
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_thr, best_j = float(thr), float(j)
    return best_thr


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, pred)),
        "sens": float(tp / (tp + fn)) if tp + fn else np.nan,
        "spec": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def feature_sets() -> list[list[str]]:
    sets = []
    for k in range(1, len(FEATURES) + 1):
        for combo in combinations(FEATURES, k):
            sets.append(list(combo))
    return sets


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("features")
    summary = grouped.agg(
        k=("k", "first"),
        n_repeats=("auc", "count"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        auc_p10=("auc", lambda s: float(np.nanpercentile(s, 10))),
        acc_mean=("acc", "mean"),
        sens_mean=("sens", "mean"),
        spec_mean=("spec", "mean"),
        f1_mean=("f1", "mean"),
        threshold_median=("threshold", "median"),
    ).reset_index()
    summary["balanced_mean"] = (summary["sens_mean"] + summary["spec_mean"]) / 2.0
    return summary.sort_values(
        ["auc_mean", "balanced_mean", "auc_p10", "f1_mean", "k"],
        ascending=[False, False, False, False, True],
    )


def final_fit_and_sample(table: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    physio = table[table["dataset"].eq("PhysioNet_LabWalks")].dropna(subset=["target"]).copy()
    sample = table[table["dataset"].eq("OUR_SAMPLE")].copy()
    y = physio["target"].astype(int).to_numpy()
    clf = model()
    clf.fit(physio[features], y)
    train_p = clf.predict_proba(physio[features])[:, 1]
    threshold = youden_threshold(y, train_p)
    train_metric = metric_row(y, train_p, threshold)
    sample_p = clf.predict_proba(sample[features])[:, 1]
    sample_out = sample[["subject_id", "target", *features]].copy()
    sample_out["probability"] = sample_p
    sample_out["threshold"] = threshold
    sample_out["prediction"] = (sample_p >= threshold).astype(int)
    sample_out["correct"] = sample_out["prediction"].eq(sample_out["target"].astype(int))
    artifact = {
        "pipeline": clf,
        "features": features,
        "threshold": threshold,
        "threshold_strategy": "physionet_only_quick_8020_screen_final_train_youden",
        "model_mode": "physionet_only_axis_aligned_quick_screen",
    }
    joblib.dump(artifact, MODEL_PATH)
    return train_metric, sample_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE_PATH)
    for feature in FEATURES:
        table[feature] = pd.to_numeric(table[feature], errors="coerce")
    table["target"] = pd.to_numeric(table["target"], errors="coerce")
    physio = table[table["dataset"].eq("PhysioNet_LabWalks")].dropna(subset=["target"]).copy()

    rows = []
    for features in feature_sets():
        x = physio[features]
        y = physio["target"].astype(int).to_numpy()
        for repeat in range(REPEATS):
            train_idx, test_idx = train_test_split(
                np.arange(len(physio)),
                test_size=TEST_SIZE,
                stratify=y,
                random_state=SEED + repeat,
            )
            clf = model()
            clf.fit(x.iloc[train_idx], y[train_idx])
            train_p = clf.predict_proba(x.iloc[train_idx])[:, 1]
            threshold = youden_threshold(y[train_idx], train_p)
            test_p = clf.predict_proba(x.iloc[test_idx])[:, 1]
            row = metric_row(y[test_idx], test_p, threshold)
            row.update({"repeat": repeat, "features": " + ".join(features), "k": len(features)})
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    best_features = summary.iloc[0]["features"].split(" + ")
    train_metric, sample_out = final_fit_and_sample(table, best_features)

    detail.to_csv(OUT_DIR / "repeat_8020_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "feature_summary.csv", index=False, encoding="utf-8-sig")
    sample_out.to_csv(OUT_DIR / "heldout_sample_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "PhysioNet_LabWalks only",
                "repeats": REPEATS,
                "test_size": TEST_SIZE,
                "seed": SEED,
                "best_features": best_features,
                "train_metrics_at_final_threshold": train_metric,
                "model_path": str(MODEL_PATH),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cols = [
        "features",
        "k",
        "auc_mean",
        "auc_std",
        "auc_p10",
        "acc_mean",
        "sens_mean",
        "spec_mean",
        "f1_mean",
        "threshold_median",
    ]
    print("TOP_FEATURES")
    print(summary[cols].head(15).to_string(index=False))
    print("\nBEST_FINAL_TRAIN")
    print(train_metric)
    print("\nHELDOUT_OUR_SAMPLE")
    print(sample_out[["subject_id", "target", "probability", "threshold", "prediction", "correct"]].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
