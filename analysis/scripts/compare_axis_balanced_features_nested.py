"""Nested comparison of the deployed V-axis features and AP/ML extensions.

This is an audit only. It does not modify the deployed model artifact.
All feature selection, collinearity checks, and sensitivity-constrained
threshold selection are fitted inside each outer-training fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))
sys.path.insert(0, str(ROOT / "backend" / "modeling"))

from analyze_20s_literature_feature_candidates import FS, SOURCE, STEP, WIN
from audit_gyro_turn_freewalk40 import read_interval
from gait_axis_aligned_core import window_features

FEATURE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "literature_20s_feature_candidate_analysis"
    / "feature_table.csv"
)
LABEL_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
OUT_DIR = ROOT / "analysis_outputs" / "axis_balanced_feature_comparison_nested"

CURRENT_V3 = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_acf_step_stride_symmetry_iqr",
]
TARGET_SENSITIVITY = 0.80
MAX_ABS_SPEARMAN = 0.85
MAX_VIF = 5.0
OUTER_SPLITS = 5
INNER_SPLITS = 3
TOP_PER_AXIS = 4
SELECTION_TOLERANCE = 0.002


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def current_acf_ratio_iqr(subject_id: str, start_sec: float) -> float:
    acc, _ = read_interval(subject_id, start_sec, duration_sec=20.0)
    values = []
    for start in range(0, len(acc) - WIN + 1, STEP):
        feature = window_features(acc[start : start + WIN]).get(
            "v_acf_step_stride_symmetry", np.nan
        )
        if np.isfinite(feature):
            values.append(float(feature))
    if not values:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def build_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    table = pd.read_csv(FEATURE_TABLE).drop(columns=["target"], errors="ignore")
    labels = (
        pd.read_csv(LABEL_TABLE)[["subject_id", "target"]]
        .drop_duplicates("subject_id")
    )
    chosen = pd.read_csv(SOURCE)[["subject_id", "start_sec"]]
    ratio_rows = []
    for row in chosen.itertuples(index=False):
        ratio_rows.append(
            {
                "subject_id": str(row.subject_id),
                "v_acf_step_stride_symmetry_iqr": current_acf_ratio_iqr(
                    str(row.subject_id), float(row.start_sec)
                ),
            }
        )
    ratio = pd.DataFrame(ratio_rows)
    table["subject_id"] = table["subject_id"].astype(str)
    labels["subject_id"] = labels["subject_id"].astype(str)
    table = (
        table.merge(ratio, on="subject_id", how="inner")
        .merge(labels, on="subject_id", how="inner")
    )

    excluded_suffixes = (
        "yaw_turn_event_count_qc",
        "n_windows",
    )
    pools = {}
    for axis in ("v", "ap", "ml"):
        pools[axis] = sorted(
            column
            for column in table.columns
            if column.startswith(f"{axis}_")
            and column not in excluded_suffixes
            and pd.to_numeric(table[column], errors="coerce").notna().sum() >= 10
        )
    for feature in CURRENT_V3:
        if feature not in pools["v"]:
            pools["v"].append(feature)
    return table.sort_values("subject_id").reset_index(drop=True), pools


def vif_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.shape[1] == 1:
        return {frame.columns[0]: 1.0}
    values = SimpleImputer(strategy="median").fit_transform(frame)
    values = RobustScaler().fit_transform(values)
    result = {}
    for index, feature in enumerate(frame.columns):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        if np.std(target) <= 1e-12:
            result[feature] = np.inf
            continue
        r2 = LinearRegression().fit(others, target).score(others, target)
        result[feature] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else np.inf
    return result


def diagnostics(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[float, float]:
    if len(features) == 1:
        return 0.0, 1.0
    part = frame[list(features)]
    correlation = part.corr(method="spearman").abs()
    upper = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    max_correlation = float(np.nanmax(upper.to_numpy()))
    max_vif = float(max(vif_values(part).values()))
    return max_correlation, max_vif


def inner_oof(
    frame: pd.DataFrame,
    y: np.ndarray,
    features: tuple[str, ...],
    seed: int,
) -> np.ndarray:
    probability = np.full(len(frame), np.nan)
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=seed
    )
    for train_index, valid_index in splitter.split(frame, y):
        model = make_model()
        model.fit(frame.iloc[train_index][list(features)], y[train_index])
        probability[valid_index] = model.predict_proba(
            frame.iloc[valid_index][list(features)]
        )[:, 1]
    return probability


def threshold_for_sensitivity(y: np.ndarray, probability: np.ndarray) -> float:
    best = (0.5, -1.0)
    for threshold in np.unique(np.concatenate(([0.0, 1.0], probability))):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= TARGET_SENSITIVITY and specificity > best[1]:
            best = (float(threshold), float(specificity))
    return best[0]


def rank_axis(
    train: pd.DataFrame, y: np.ndarray, candidates: list[str]
) -> list[str]:
    scores = []
    for feature in candidates:
        values = pd.to_numeric(train[feature], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) <= 1:
            continue
        filled = values.fillna(values.median())
        correlation = filled.corr(pd.Series(y, index=filled.index), method="spearman")
        scores.append((abs(float(correlation)) if np.isfinite(correlation) else 0.0, feature))
    return [feature for _, feature in sorted(scores, reverse=True)[:TOP_PER_AXIS]]


def candidate_combinations(
    strategy: str,
    train: pd.DataFrame,
    y: np.ndarray,
    pools: dict[str, list[str]],
) -> list[tuple[str, ...]]:
    if strategy == "current_v3":
        return [tuple(CURRENT_V3)]
    if strategy == "current_v3_plus_ap":
        return [tuple(CURRENT_V3 + [feature]) for feature in pools["ap"]]
    if strategy == "current_v3_plus_ml":
        return [tuple(CURRENT_V3 + [feature]) for feature in pools["ml"]]
    if strategy != "axis_balanced_3to4":
        raise ValueError(f"Unknown strategy: {strategy}")

    ranked = {
        axis: rank_axis(train, y, pools[axis])
        for axis in ("v", "ap", "ml")
    }
    all_ranked = sorted(set(ranked["v"] + ranked["ap"] + ranked["ml"]))
    output = set()
    for size in (3, 4):
        for combo in combinations(all_ranked, size):
            if all(any(feature.startswith(f"{axis}_") for feature in combo) for axis in ("v", "ap", "ml")):
                output.add(tuple(combo))
    return sorted(output)


def select_combination(
    strategy: str,
    train: pd.DataFrame,
    y: np.ndarray,
    pools: dict[str, list[str]],
    seed: int,
) -> tuple[tuple[str, ...], float, float, float, float, int]:
    results = []
    for features in candidate_combinations(strategy, train, y, pools):
        max_corr, max_vif = diagnostics(train, features)
        if max_corr > MAX_ABS_SPEARMAN or max_vif > MAX_VIF:
            continue
        probability = inner_oof(train, y, features, seed)
        results.append(
            {
                "features": features,
                "auc": float(roc_auc_score(y, probability)),
                "threshold": threshold_for_sensitivity(y, probability),
                "max_corr": max_corr,
                "max_vif": max_vif,
            }
        )
    if not results:
        raise RuntimeError(f"No admissible combination for {strategy}")
    best_auc = max(result["auc"] for result in results)
    near_best = [
        result for result in results
        if result["auc"] >= best_auc - SELECTION_TOLERANCE
    ]
    best = sorted(
        near_best,
        key=lambda result: (
            len(result["features"]),
            -result["auc"],
            result["features"],
        ),
    )[0]
    return (
        best["features"],
        best["auc"],
        best["threshold"],
        best["max_corr"],
        best["max_vif"],
        len(results),
    )


def evaluate_fold(
    strategy: str,
    repeat: int,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    table: pd.DataFrame,
    pools: dict[str, list[str]],
) -> tuple[dict, list[dict]]:
    train = table.iloc[train_index].reset_index(drop=True)
    test = table.iloc[test_index].reset_index(drop=True)
    y_train = train["target"].to_numpy(int)
    selected, inner_auc, threshold, max_corr, max_vif, n_candidates = select_combination(
        strategy,
        train,
        y_train,
        pools,
        seed=330000 + repeat * 10 + fold,
    )
    model = make_model().fit(train[list(selected)], y_train)
    probability = model.predict_proba(test[list(selected)])[:, 1]
    prediction = (probability >= threshold).astype(int)
    selection = {
        "strategy": strategy,
        "repeat": repeat,
        "fold": fold,
        "selected_features": "|".join(selected),
        "n_selected": len(selected),
        "inner_auc": inner_auc,
        "threshold": threshold,
        "max_abs_spearman": max_corr,
        "max_vif": max_vif,
        "n_admissible_candidates": n_candidates,
    }
    predictions = [
        {
            "strategy": strategy,
            "repeat": repeat,
            "fold": fold,
            "subject_id": subject_id,
            "target": int(target),
            "probability": float(probability_value),
            "prediction": int(prediction_value),
        }
        for subject_id, target, probability_value, prediction_value in zip(
            test["subject_id"], test["target"], probability, prediction
        )
    ]
    return selection, predictions


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (strategy, repeat), part in predictions.groupby(["strategy", "repeat"]):
        y = part["target"].to_numpy(int)
        probability = part["probability"].to_numpy(float)
        prediction = part["prediction"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "strategy": strategy,
                "repeat": repeat,
                "auc": roc_auc_score(y, probability),
                "sensitivity": recall_score(y, prediction, zero_division=0),
                "specificity": tn / (tn + fp),
                "precision": precision_score(y, prediction, zero_division=0),
                "accuracy": accuracy_score(y, prediction),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby("strategy")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
        .sort_values("auc_mean", ascending=False)
    )
    return metrics, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, pools = build_table()
    table.to_csv(OUT_DIR / "subject_feature_table.csv", index=False, encoding="utf-8-sig")

    strategies = [
        "current_v3",
        "current_v3_plus_ap",
        "current_v3_plus_ml",
        "axis_balanced_3to4",
    ]
    y = table["target"].to_numpy(int)
    tasks = []
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=220000 + repeat,
        )
        for fold, (train_index, test_index) in enumerate(splitter.split(table, y)):
            for strategy in strategies:
                tasks.append((strategy, repeat, fold, train_index, test_index))

    print(
        f"subjects={len(table)} normal={(y == 0).sum()} impaired={(y == 1).sum()} "
        f"tasks={len(tasks)}",
        flush=True,
    )
    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_fold)(
            strategy, repeat, fold, train_index, test_index, table, pools
        )
        for strategy, repeat, fold, train_index, test_index in tasks
    )
    selections = pd.DataFrame([output[0] for output in outputs])
    predictions = pd.DataFrame(
        [row for _, prediction_rows in outputs for row in prediction_rows]
    )
    metrics, summary = summarize(predictions)

    combo_counts = (
        selections.groupby(["strategy", "selected_features"])
        .size()
        .rename("selected_folds")
        .reset_index()
    )
    combo_counts["selection_rate_within_strategy"] = combo_counts.groupby("strategy")[
        "selected_folds"
    ].transform(lambda values: values / values.sum())
    combo_counts = combo_counts.sort_values(
        ["strategy", "selected_folds"], ascending=[True, False]
    )

    feature_counts = []
    for strategy, part in selections.groupby("strategy"):
        counter = Counter()
        for value in part["selected_features"]:
            counter.update(value.split("|"))
        for feature, count in counter.most_common():
            feature_counts.append(
                {
                    "strategy": strategy,
                    "feature": feature,
                    "selected_folds": count,
                    "selection_rate": count / len(part),
                }
            )

    selections.to_csv(OUT_DIR / "outer_fold_selections.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "outer_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    combo_counts.to_csv(OUT_DIR / "combination_frequency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_counts).to_csv(
        OUT_DIR / "feature_frequency.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "n_subjects": int(len(table)),
        "normal": int((y == 0).sum()),
        "impaired": int((y == 1).sum()),
        "source_condition": "one preselected 20-second segment per subject",
        "label_source": str(LABEL_TABLE),
        "label_definition": "revised clinical OR label without s3_velocity",
        "current_model_modified": False,
        "outer_cv": f"StratifiedKFold(5) x {args.repeats} repeats",
        "inner_cv": "StratifiedKFold(3), all selection on outer-training only",
        "target_sensitivity": TARGET_SENSITIVITY,
        "max_abs_spearman": MAX_ABS_SPEARMAN,
        "max_vif": MAX_VIF,
        "top_candidates_per_axis_for_balanced_search": TOP_PER_AXIS,
        "axis_pool_sizes": {axis: len(features) for axis, features in pools.items()},
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nNested outer-CV summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMost frequent combinations")
    print(combo_counts.groupby("strategy").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
