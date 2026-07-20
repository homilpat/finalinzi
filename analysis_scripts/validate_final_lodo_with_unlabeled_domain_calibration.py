from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_lodo_unlabeled_domain_calibration"
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from model_all_domains_to_service_reference import (  # noqa: E402
    FEATURES,
    SUBJECT_TABLE,
    align_to_service,
    fit_service_reference,
    model,
    robust_params,
    threshold_for_sens,
)


TARGET_SENSITIVITY = 0.80
RANDOM_STATE = 240719


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
    return table.dropna(subset=FEATURES, how="all").reset_index(drop=True)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    y = table["target"].to_numpy()
    groups = table["dataset"].astype(str).to_numpy()
    rows = []
    preds = []
    oof_prob = np.full(len(table), np.nan)
    oof_pred = np.full(len(table), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(table, y, groups), start=1):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        left_out = str(test["dataset"].iloc[0])

        service_med, service_scale, refs = fit_service_reference(train)
        x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]

        # Unlabeled domain calibration: labels are not used, but the held-out domain feature
        # distribution is used once and applied identically to every held-out row.
        test_refs = dict(refs)
        test_refs[left_out] = robust_params(test, FEATURES)
        x_test = align_to_service(test, service_med, service_scale, test_refs)[FEATURES]

        clf = model(RANDOM_STATE + fold)
        clf.fit(x_train, y[train_idx])
        train_prob = clf.predict_proba(x_train)[:, 1]
        threshold = threshold_for_sens(y[train_idx], train_prob, TARGET_SENSITIVITY)
        test_prob = clf.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        oof_prob[test_idx] = test_prob
        oof_pred[test_idx] = test_pred

        train_metric = calc_metrics(y[train_idx], train_prob, (train_prob >= threshold).astype(int))
        train_metric.update({
            "fold": fold,
            "left_out_dataset": left_out,
            "set": "train_apparent",
            "threshold": threshold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        rows.append(train_metric)

        test_metric = calc_metrics(y[test_idx], test_prob, test_pred)
        test_metric.update({
            "fold": fold,
            "left_out_dataset": left_out,
            "set": "test_domain_calibrated",
            "threshold": threshold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        rows.append(test_metric)

        pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
        pred["fold"] = fold
        pred["left_out_dataset"] = left_out
        pred["probability_impaired"] = test_prob
        pred["threshold"] = threshold
        pred["prediction"] = test_pred
        preds.append(pred)

    aggregate = calc_metrics(y, oof_prob, oof_pred)
    aggregate.update({
        "fold": "all",
        "left_out_dataset": "all",
        "set": "test_domain_calibrated_aggregate",
        "threshold": np.nan,
        "n_train": np.nan,
        "n_test": len(table),
    })
    rows.append(aggregate)

    metrics = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    dataset_summary = (
        pred_df.groupby(["dataset", "target"], dropna=False)
        .agg(
            n_predictions=("prediction", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )
    summary = (
        metrics[metrics["fold"].ne("all")]
        .groupby("set")
        .agg(
            n_folds=("fold", "count"),
            auc_mean=("auc", "mean"),
            acc_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
    )

    metrics.to_csv(OUT_DIR / "metrics_by_domain_fold.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "predictions_by_domain_fold.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    print("aggregate")
    print(metrics[metrics["fold"].eq("all")].to_string(index=False))
    print("\nsummary")
    print(summary.to_string(index=False))
    print("\ndataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    run()
