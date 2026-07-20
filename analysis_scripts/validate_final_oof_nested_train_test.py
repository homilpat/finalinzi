from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_oof_nested_train_test"
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
RANDOM_STATE = 210719


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


def metric_row(kind: str, fold: int | str, y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    out = calc_metrics(y, prob, pred)
    out.update({"validation": kind, "fold": fold, "threshold": float(threshold), "n": int(len(y))})
    return out


def run_oof_5fold(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []
    oof_prob = np.full(len(table), np.nan)
    oof_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(table, y), start=1):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        service_med, service_scale, refs = fit_service_reference(train)
        x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
        x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]

        clf = model(RANDOM_STATE + fold)
        clf.fit(x_train, y[train_idx])
        train_prob = clf.predict_proba(x_train)[:, 1]
        threshold = threshold_for_sens(y[train_idx], train_prob, TARGET_SENSITIVITY)
        test_prob = clf.predict_proba(x_test)[:, 1]

        oof_prob[test_idx] = test_prob
        oof_thresholds.append(threshold)
        rows.append(metric_row("oof_5fold_train_apparent", fold, y[train_idx], train_prob, threshold))
        rows.append(metric_row("oof_5fold_test_fold", fold, y[test_idx], test_prob, threshold))

        pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
        pred["validation"] = "oof_5fold"
        pred["fold"] = fold
        pred["probability_impaired"] = test_prob
        pred["threshold"] = threshold
        pred["prediction"] = (test_prob >= threshold).astype(int)
        preds.append(pred)

    global_threshold = float(np.median(oof_thresholds))
    rows.append(metric_row("oof_5fold_test_aggregate", "all", y, oof_prob, global_threshold))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def inner_oof_threshold(train: pd.DataFrame, y_train: np.ndarray, outer_fold: int) -> float:
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + outer_fold * 100)
    inner_prob = np.full(len(train), np.nan)
    for inner_fold, (sub_idx, val_idx) in enumerate(inner.split(train, y_train), start=1):
        sub_train = train.iloc[sub_idx].copy()
        val = train.iloc[val_idx].copy()
        service_med, service_scale, refs = fit_service_reference(sub_train)
        x_sub = align_to_service(sub_train, service_med, service_scale, refs)[FEATURES]
        x_val = align_to_service(val, service_med, service_scale, refs)[FEATURES]
        clf = model(RANDOM_STATE + outer_fold * 1000 + inner_fold)
        clf.fit(x_sub, y_train[sub_idx])
        inner_prob[val_idx] = clf.predict_proba(x_val)[:, 1]
    return threshold_for_sens(y_train, inner_prob, TARGET_SENSITIVITY)


def run_nested_5fold(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []
    nested_prob = np.full(len(table), np.nan)
    nested_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(table, y), start=1):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        y_train = y[train_idx]
        threshold = inner_oof_threshold(train.reset_index(drop=True), y_train, fold)

        service_med, service_scale, refs = fit_service_reference(train)
        x_train = align_to_service(train, service_med, service_scale, refs)[FEATURES]
        x_test = align_to_service(test, service_med, service_scale, refs)[FEATURES]
        clf = model(RANDOM_STATE + 5000 + fold)
        clf.fit(x_train, y_train)
        train_prob = clf.predict_proba(x_train)[:, 1]
        test_prob = clf.predict_proba(x_test)[:, 1]

        nested_prob[test_idx] = test_prob
        nested_thresholds.append(threshold)
        rows.append(metric_row("nested_5fold_train_apparent", fold, y_train, train_prob, threshold))
        rows.append(metric_row("nested_5fold_outer_test", fold, y[test_idx], test_prob, threshold))

        pred = test[["dataset", "subject_id", "group_id", "target"]].copy()
        pred["validation"] = "nested_5fold"
        pred["fold"] = fold
        pred["probability_impaired"] = test_prob
        pred["threshold"] = threshold
        pred["prediction"] = (test_prob >= threshold).astype(int)
        preds.append(pred)

    global_threshold = float(np.median(nested_thresholds))
    rows.append(metric_row("nested_5fold_outer_test_aggregate", "all", y, nested_prob, global_threshold))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "f1", "threshold"]
    fold_rows = rows[rows["fold"].ne("all")].copy()
    summary = (
        fold_rows.groupby("validation")
        .agg(
            n_folds=("fold", "count"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
        )
        .reset_index()
    )
    aggregate = rows[rows["fold"].eq("all")].copy()
    aggregate["n_folds"] = 1
    for col in metric_cols:
        aggregate[f"{col}_mean"] = aggregate[col]
        aggregate[f"{col}_std"] = np.nan
    aggregate["tn_mean"] = aggregate["tn"]
    aggregate["fp_mean"] = aggregate["fp"]
    aggregate["fn_mean"] = aggregate["fn"]
    aggregate["tp_mean"] = aggregate["tp"]
    aggregate = aggregate[summary.columns]
    return pd.concat([summary, aggregate], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    table.to_csv(OUT_DIR / "subject_table_used.csv", index=False, encoding="utf-8-sig")
    oof_rows, oof_preds = run_oof_5fold(table)
    nested_rows, nested_preds = run_nested_5fold(table)
    rows = pd.concat([oof_rows, nested_rows], ignore_index=True)
    preds = pd.concat([oof_preds, nested_preds], ignore_index=True)
    summary = summarize(rows)

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
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "predictions_by_fold.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    print("summary")
    print(summary.to_string(index=False))
    print("\naggregate")
    print(rows[rows["fold"].eq("all")].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
