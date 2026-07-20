from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_service_reference_cv_suite"
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


def split_jobs(table: pd.DataFrame) -> list[dict]:
    y = table["target"].to_numpy()
    jobs = []

    sss = StratifiedShuffleSplit(n_splits=300, test_size=0.2, random_state=1910000)
    for repeat, (train_idx, test_idx) in enumerate(sss.split(table, y)):
        jobs.append({"cv_scheme": "repeated_8to2", "repeat": repeat, "fold": 0, "train_idx": train_idx, "test_idx": test_idx})

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1920000)
    for fold, (train_idx, test_idx) in enumerate(skf.split(table, y)):
        jobs.append({"cv_scheme": "stratified_5fold", "repeat": 0, "fold": fold, "train_idx": train_idx, "test_idx": test_idx})

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=1930000)
    for i, (train_idx, test_idx) in enumerate(rskf.split(table, y)):
        jobs.append({"cv_scheme": "repeated_5fold_x100", "repeat": i // 5, "fold": i % 5, "train_idx": train_idx, "test_idx": test_idx})
    return jobs


def run_one(job: dict, table_records: list[dict]) -> tuple[dict, pd.DataFrame]:
    table = pd.DataFrame.from_records(table_records)
    y = table["target"].astype(int).to_numpy()
    train_idx = np.asarray(job["train_idx"], dtype=int)
    test_idx = np.asarray(job["test_idx"], dtype=int)
    train = table.iloc[train_idx].copy()
    test = table.iloc[test_idx].copy()

    service_med, service_scale, refs = fit_service_reference(train)
    x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
    x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]

    clf = model(1940000 + int(job["repeat"]) * 10 + int(job["fold"]))
    clf.fit(x_train, y[train_idx])
    train_prob = clf.predict_proba(x_train)[:, 1]
    threshold = threshold_for_sens(y[train_idx], train_prob, TARGET_SENSITIVITY)
    test_prob = clf.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metric = calc_metrics(y[test_idx], test_prob, test_pred)
    metric.update(
        {
            "cv_scheme": job["cv_scheme"],
            "repeat": int(job["repeat"]),
            "fold": int(job["fold"]),
            "threshold": threshold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
    )
    pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
    pred["cv_scheme"] = job["cv_scheme"]
    pred["repeat"] = int(job["repeat"])
    pred["fold"] = int(job["fold"])
    pred["probability_impaired"] = test_prob
    pred["prediction"] = test_pred
    return metric, pred


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("cv_scheme")
        .agg(
            n_splits=("fold", "count"),
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
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
            threshold_median=("threshold", "median"),
        )
        .reset_index()
        .sort_values("cv_scheme")
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    table.to_csv(OUT_DIR / "final_cv_subject_table.csv", index=False, encoding="utf-8-sig")
    jobs = split_jobs(table)
    workers = max(1, os.cpu_count() or 1)
    table_records = table.to_dict("records")
    metrics = []
    preds = []
    print(f"jobs={len(jobs)} workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, job, table_records) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            metric, pred = future.result()
            metrics.append(metric)
            preds.append(pred)
            if i % 50 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)}")

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(preds, ignore_index=True)
    summary = summarize(metrics_df)
    dataset_summary = (
        preds_df.groupby(["cv_scheme", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("prediction", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )
    metrics_df.to_csv(OUT_DIR / "final_cv_metrics_by_split.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "final_cv_predictions_by_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "final_cv_metrics_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(OUT_DIR / "final_cv_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print("\nsummary")
    print(summary.to_string(index=False))
    print("\ndataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
