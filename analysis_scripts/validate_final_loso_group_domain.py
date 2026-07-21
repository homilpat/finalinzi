from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_loso_group_domain"
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
RANDOM_STATE = 230719


def load_table() -> pd.DataFrame:
    table = pd.read_csv(SUBJECT_TABLE)
    table = table[table["target"].notna()].copy()
    table["target"] = table["target"].astype(int)
    table = table.dropna(subset=FEATURES, how="all").reset_index(drop=True)
    return table


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


def run_logo(table: pd.DataFrame, group_col: str, validation_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = table["target"].to_numpy()
    groups = table[group_col].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    rows = []
    preds = []
    oof_prob = np.full(len(table), np.nan)
    oof_pred = np.full(len(table), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(table, y, groups), start=1):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        try:
            service_med, service_scale, refs = fit_service_reference(train)
        except ValueError as exc:
            rows.append({
                "validation": validation_name,
                "fold": fold,
                "left_out": str(test[group_col].iloc[0]),
                "set": "skipped",
                "error": str(exc),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })
            continue

        x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
        x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]
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
            "validation": validation_name,
            "fold": fold,
            "left_out": str(test[group_col].iloc[0]),
            "set": "train_apparent",
            "threshold": threshold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "error": "",
        })
        rows.append(train_metric)

        if len(np.unique(y[test_idx])) == 2:
            test_metric = calc_metrics(y[test_idx], test_prob, test_pred)
        else:
            test_metric = {
                "auc": np.nan,
                "accuracy": accuracy_score(y[test_idx], test_pred),
                "sensitivity": np.nan,
                "specificity": np.nan,
                "f1": f1_score(y[test_idx], test_pred, zero_division=0),
                "tn": int(np.sum((y[test_idx] == 0) & (test_pred == 0))),
                "fp": int(np.sum((y[test_idx] == 0) & (test_pred == 1))),
                "fn": int(np.sum((y[test_idx] == 1) & (test_pred == 0))),
                "tp": int(np.sum((y[test_idx] == 1) & (test_pred == 1))),
            }
        test_metric.update({
            "validation": validation_name,
            "fold": fold,
            "left_out": str(test[group_col].iloc[0]),
            "set": "test",
            "threshold": threshold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "error": "",
        })
        rows.append(test_metric)

        pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
        pred["validation"] = validation_name
        pred["fold"] = fold
        pred["left_out"] = str(test[group_col].iloc[0])
        pred["threshold"] = threshold
        pred["probability_impaired"] = test_prob
        pred["prediction"] = test_pred
        preds.append(pred)

    valid = np.isfinite(oof_prob) & (oof_pred >= 0)
    aggregate = calc_metrics(y[valid], oof_prob[valid], oof_pred[valid])
    aggregate.update({
        "validation": validation_name,
        "fold": "all",
        "left_out": "all",
        "set": "test_aggregate",
        "threshold": np.nan,
        "n_train": np.nan,
        "n_test": int(valid.sum()),
        "error": "",
    })
    rows.append(aggregate)

    row_df = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fold_summary = (
        row_df[row_df["set"].isin(["train_apparent", "test"])]
        .groupby(["validation", "set"], dropna=False)
        .agg(
            n_folds=("fold", "count"),
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
            tn_sum=("tn", "sum"),
            fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"),
            tp_sum=("tp", "sum"),
        )
        .reset_index()
    )
    return row_df, pred_df, fold_summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    table.to_csv(OUT_DIR / "subject_table_used.csv", index=False, encoding="utf-8-sig")

    subject_rows, subject_preds, subject_summary = run_logo(table, "group_id", "loso_group")
    domain_rows, domain_preds, domain_summary = run_logo(table, "dataset", "leave_one_domain_out")
    rows = pd.concat([subject_rows, domain_rows], ignore_index=True)
    preds = pd.concat([subject_preds, domain_preds], ignore_index=True)
    summary = pd.concat([subject_summary, domain_summary], ignore_index=True)
    aggregate = rows[rows["set"].eq("test_aggregate")].copy()

    dataset_summary = (
        preds.groupby(["validation", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("prediction", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )

    rows.to_csv(OUT_DIR / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "predictions_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "metrics_fold_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUT_DIR / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    print("aggregate")
    print(aggregate.to_string(index=False))
    print("\nfold summary")
    print(summary.to_string(index=False))
    print("\ndataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
