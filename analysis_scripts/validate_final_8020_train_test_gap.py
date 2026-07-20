from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_8020_train_test_gap"
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from model_all_domains_to_service_reference import (  # noqa: E402
    FEATURES,
    SUBJECT_TABLE,
    align_to_service,
    fit_service_reference,
    model,
    threshold_for_sens,
)


TARGET_SENSITIVITY = 0.80
N_SPLITS = 300
RANDOM_STATE = 220719


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


def load_table() -> pd.DataFrame:
    table = pd.read_csv(SUBJECT_TABLE)
    table = table[table["target"].notna()].copy()
    table["target"] = table["target"].astype(int)
    table = table.dropna(subset=FEATURES, how="all").reset_index(drop=True)
    return table


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    table.to_csv(OUT_DIR / "subject_table_used.csv", index=False, encoding="utf-8-sig")
    y = table["target"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_STATE)
    rows = []
    gaps = []

    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        service_med, service_scale, refs = fit_service_reference(train)
        x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
        x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]
        clf = model(RANDOM_STATE + repeat)
        clf.fit(x_train, y[train_idx])

        train_prob = clf.predict_proba(x_train)[:, 1]
        threshold = threshold_for_sens(y[train_idx], train_prob, TARGET_SENSITIVITY)
        test_prob = clf.predict_proba(x_test)[:, 1]

        train_metric = calc_metrics(y[train_idx], train_prob, (train_prob >= threshold).astype(int))
        test_metric = calc_metrics(y[test_idx], test_prob, (test_prob >= threshold).astype(int))
        train_metric.update({"split": repeat, "set": "train_apparent", "threshold": threshold})
        test_metric.update({"split": repeat, "set": "test", "threshold": threshold})
        rows.extend([train_metric, test_metric])

        gap = {"split": repeat, "threshold": threshold}
        for key in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
            gap[f"{key}_train"] = train_metric[key]
            gap[f"{key}_test"] = test_metric[key]
            gap[f"{key}_gap_train_minus_test"] = train_metric[key] - test_metric[key]
        gaps.append(gap)

    metrics = pd.DataFrame(rows)
    gap_df = pd.DataFrame(gaps)
    summary = (
        metrics.groupby("set")
        .agg(
            n_splits=("split", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            threshold_median=("threshold", "median"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
        )
        .reset_index()
    )
    gap_summary = gap_df.drop(columns=["split"]).agg(["mean", "std", "median"]).reset_index().rename(columns={"index": "stat"})

    metrics.to_csv(OUT_DIR / "metrics_by_split.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(OUT_DIR / "train_test_gap_by_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    gap_summary.to_csv(OUT_DIR / "train_test_gap_summary.csv", index=False, encoding="utf-8-sig")

    print("summary")
    print(summary.to_string(index=False))
    print("\ngap summary")
    print(gap_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    run()
