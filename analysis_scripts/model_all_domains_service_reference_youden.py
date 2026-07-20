from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "all_domains_to_service_reference_youden"
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from model_all_domains_to_service_reference import (  # noqa: E402
    FEATURES,
    SAMPLE_TABLE,
    SUBJECT_TABLE,
    align_to_service,
    domain_z_to_service,
    fit_service_reference,
    model,
)
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402


def threshold_youden(y: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) <= 1:
        return float(vals[0]) if len(vals) else 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-9]
    best_t = 0.5
    best_j = -np.inf
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


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
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1810000)
    modes = {
        "raw_no_alignment": "raw",
        "all_domains_to_service_reference": "aligned",
        "all_domains_service_z": "service_z",
    }
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        service_med, service_scale, refs = fit_service_reference(train)
        for name, mode_name in modes.items():
            if mode_name == "raw":
                x_train = train[FEATURES]
                x_test = test[FEATURES]
            elif mode_name == "aligned":
                x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
                x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]
            else:
                x_train = domain_z_to_service(train, refs)
                x_test = domain_z_to_service(test, refs)
            clf = model(1820000 + repeat)
            clf.fit(x_train, y[train_idx])
            train_prob = clf.predict_proba(x_train)[:, 1]
            threshold = threshold_youden(y[train_idx], train_prob)
            test_prob = clf.predict_proba(x_test)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(y[test_idx], test_prob, test_pred)
            row.update({"model_set": name, "repeat": repeat, "threshold": threshold})
            metrics_rows.append(row)
            pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
            pred["model_set"] = name
            pred["repeat"] = repeat
            pred["probability_impaired"] = test_prob
            pred["prediction"] = test_pred
            pred_rows.append(pred)
    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(OUT_DIR / "youden_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "youden_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics.groupby("model_set")
        .agg(
            n_repeats=("repeat", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
            threshold_median=("threshold", "median"),
        )
        .reset_index()
        .sort_values(["auc_mean", "acc_mean"], ascending=[False, False])
    )
    summary.to_csv(OUT_DIR / "youden_metrics_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary = (
        preds.groupby(["model_set", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("prediction", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUT_DIR / "youden_dataset_summary.csv", index=False, encoding="utf-8-sig")

    samples = pd.read_csv(SAMPLE_TABLE)
    service_med, service_scale, refs = fit_service_reference(table)
    sample_rows = []
    for name, mode_name in modes.items():
        if mode_name == "raw":
            x_train = table[FEATURES]
            x_sample = samples[FEATURES]
        elif mode_name == "aligned":
            x_train = align_to_service(table, service_med, service_scale, refs)[FEATURES]
            sample_tmp = samples.copy()
            sample_tmp["dataset"] = "OUR_SAMPLE"
            x_sample = align_to_service(sample_tmp, service_med, service_scale, refs, fallback_dataset="OUR_SAMPLE")[FEATURES]
        else:
            x_train = domain_z_to_service(table, refs)
            sample_tmp = samples.copy()
            sample_tmp["dataset"] = "OUR_SAMPLE"
            x_sample = domain_z_to_service(sample_tmp, refs, fallback_dataset="OUR_SAMPLE")
        clf = model(1830000 + len(name))
        clf.fit(x_train, y)
        train_prob = clf.predict_proba(x_train)[:, 1]
        threshold = threshold_youden(y, train_prob)
        sample_prob = clf.predict_proba(x_sample)[:, 1]
        for idx, sample in samples.iterrows():
            sample_rows.append(
                {
                    "model_set": name,
                    "source_id": sample["source_id"],
                    "quality_score": sample.get("quality_score", np.nan),
                    "probability_impaired": float(sample_prob[idx]),
                    "threshold": threshold,
                    "prediction": int(sample_prob[idx] >= threshold),
                }
            )
    sample_out = pd.DataFrame(sample_rows)
    sample_out.to_csv(OUT_DIR / "youden_sample_predictions.csv", index=False, encoding="utf-8-sig")
    print("summary")
    print(summary.to_string(index=False))
    print("\ndataset")
    print(dataset_summary.to_string(index=False))
    print("\nsample")
    print(sample_out.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    run()
