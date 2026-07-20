from __future__ import annotations

import sys
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
OUT_DIR = ROOT / "analysis_outputs" / "threshold_sensitivity80_candidates"
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from model_combined_fold_median_direction_iqr import (  # noqa: E402
    FEATURE_SETS,
    fit_fold_params,
    make_subject_table,
    transform_risk_z,
)


TARGET_SENSITIVITY = 0.80


def model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )


def threshold_for_min_sensitivity(y: np.ndarray, prob: np.ndarray, min_sens: float) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) == 0:
        return 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else [], vals.max() + 1e-9]
    best_t = float(candidates[0])
    best_spec = -np.inf
    best_sens = -np.inf
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_sens = sens
            best_t = float(t)
    if best_spec > -np.inf:
        return best_t
    # If the train fold cannot reach the target, pick max Youden as fallback.
    best_j = -np.inf
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best_j:
            best_j = sens + spec - 1
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


def run_scalar_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_features = list(dict.fromkeys(feature for features in FEATURE_SETS.values() for feature in features))
    table = make_subject_table(all_features)
    y = table["target"].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1410000)
    candidates = {
        "A_local_stable4__risk_z_features_plus_summary": (
            FEATURE_SETS["A_local_stable4"],
            "risk_z_features_plus_summary",
        ),
        "B_direction_shape6__risk_z_features": (
            FEATURE_SETS["B_direction_shape6"],
            "risk_z_features",
        ),
        "B_direction_shape6__risk_z_summary3": (
            FEATURE_SETS["B_direction_shape6"],
            "risk_z_summary3",
        ),
        "C_union9__risk_z_summary3": (
            FEATURE_SETS["C_union9"],
            "risk_z_summary3",
        ),
        "C_union9__risk_z_features_plus_summary": (
            FEATURE_SETS["C_union9"],
            "risk_z_features_plus_summary",
        ),
    }
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        for name, (features, mode) in candidates.items():
            normal_median, scale, direction = fit_fold_params(train, features)
            x_train = transform_risk_z(train, features, normal_median, scale, direction)
            x_test = transform_risk_z(test, features, normal_median, scale, direction)
            risk_cols = [f"{feature}__risk_z" for feature in features]
            if mode == "risk_z_features":
                cols = risk_cols
            elif mode == "risk_z_summary3":
                cols = ["risk_z_mean", "risk_z_max", "risk_z_count_pos"]
            else:
                cols = risk_cols + ["risk_z_mean", "risk_z_max", "risk_z_count_pos"]
            clf = model(1420000 + repeat)
            clf.fit(x_train[cols], y[train_idx])
            train_prob = clf.predict_proba(x_train[cols])[:, 1]
            threshold = threshold_for_min_sensitivity(y[train_idx], train_prob, TARGET_SENSITIVITY)
            test_prob = clf.predict_proba(x_test[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(y[test_idx], test_prob, test_pred)
            row.update({"candidate": name, "repeat": repeat, "threshold": threshold, "target_sensitivity": TARGET_SENSITIVITY})
            metrics_rows.append(row)
            pred = test[["group_id", "dataset", "label_group", "target"]].copy()
            pred["candidate"] = name
            pred["repeat"] = repeat
            pred["probability_impaired"] = test_prob
            pred["prediction"] = test_pred
            pred_rows.append(pred)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby("candidate")
        .agg(
            n_repeats=("repeat", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_p10=("sensitivity", lambda x: x.quantile(0.10)),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
            threshold_median=("threshold", "median"),
        )
        .reset_index()
        .sort_values(["sensitivity_mean", "specificity_mean", "auc_mean"], ascending=[False, False, False])
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df, preds_df = run_scalar_candidates()
    metrics_df.to_csv(OUT_DIR / "sens80_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "sens80_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = summarize(metrics_df)
    summary.to_csv(OUT_DIR / "sens80_metrics_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary = (
        preds_df.groupby(["candidate", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("group_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUT_DIR / "sens80_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print("summary")
    print(summary.to_string(index=False))
    print("\ndataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
