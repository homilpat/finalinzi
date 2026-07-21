from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "fixed_best10_domain_stable_selected_model"
SUBJECT_TABLE = (
    ROOT
    / "analysis_outputs"
    / "all_extractors_domain_stability_screen"
    / "fixed_best10_quality_subject_table.csv"
)
SAMPLE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "fixed_best10_quality_pipeline"
    / "fixed_best10_sample_features.csv"
)

FEATURES = [
    "v_spec_entropy",
    "v_acf_stride_peak_width_sec",
    "ap_acf_stride_peak_width_sec",
    "v_stride_shape_cv_mean",
    "ml_acf_stride_peak_width_sec",
    "v_peak_timing_sd_pct",
]


def model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )


def threshold_for_sens(y: np.ndarray, prob: np.ndarray, min_sens: float = 0.8) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) <= 1:
        return float(vals[0]) if len(vals) else 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-9]
    best_t = candidates[0]
    best_spec = -np.inf
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_t = t
    return float(best_t)


def calc_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SUBJECT_TABLE)
    table = table[table["target"].notna()].copy()
    table["target"] = table["target"].astype(int)
    table = table.dropna(subset=FEATURES, how="all").reset_index(drop=True)
    y = table["target"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1510000)
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx]
        test = table.iloc[test_idx]
        clf = model(1520000 + repeat)
        clf.fit(train[FEATURES], y[train_idx])
        train_prob = clf.predict_proba(train[FEATURES])[:, 1]
        threshold = threshold_for_sens(y[train_idx], train_prob, 0.8)
        test_prob = clf.predict_proba(test[FEATURES])[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        row = calc_metrics(y[test_idx], test_prob, test_pred)
        row.update({"repeat": repeat, "threshold": threshold})
        metrics_rows.append(row)
        pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
        pred["repeat"] = repeat
        pred["probability_impaired"] = test_prob
        pred["prediction"] = test_pred
        pred_rows.append(pred)
    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(OUT_DIR / "domain_stable_selected_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "domain_stable_selected_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = metrics.agg(
        {
            "auc": ["mean", "std"],
            "accuracy": ["mean"],
            "sensitivity": ["mean"],
            "specificity": ["mean"],
            "f1": ["mean"],
            "tn": ["mean"],
            "fp": ["mean"],
            "fn": ["mean"],
            "tp": ["mean"],
        }
    )
    summary.to_csv(OUT_DIR / "domain_stable_selected_metrics_summary.csv", encoding="utf-8-sig")
    dataset_summary = (
        preds.groupby(["dataset", "target"], dropna=False)
        .agg(
            n_predictions=("prediction", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUT_DIR / "domain_stable_selected_dataset_summary.csv", index=False, encoding="utf-8-sig")

    samples = pd.read_csv(SAMPLE_TABLE)
    full = model(1530000)
    full.fit(table[FEATURES], y)
    full_prob = full.predict_proba(table[FEATURES])[:, 1]
    full_threshold = threshold_for_sens(y, full_prob, 0.8)
    sample_prob = full.predict_proba(samples[FEATURES])[:, 1]
    sample_out = samples[["source_id", "subject_id", "quality_score", *FEATURES]].copy()
    sample_out["probability_impaired"] = sample_prob
    sample_out["threshold"] = full_threshold
    sample_out["prediction"] = (sample_prob >= full_threshold).astype(int)
    sample_out.to_csv(OUT_DIR / "domain_stable_selected_sample_predictions.csv", index=False, encoding="utf-8-sig")
    print("summary")
    print(metrics.mean(numeric_only=True).to_string())
    print("\ndataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nsample")
    print(sample_out.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    run()
