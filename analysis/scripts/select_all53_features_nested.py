"""Leakage-safe nested selection from all 53 axis-expanded gait features.

The deployed model is not modified. Candidate filtering, univariate ranking,
correlation pruning, VIF checks, combination selection, preprocessing, and
threshold selection are all learned from each outer-training fold only.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))

import compare_axis_balanced_features_nested as axis_audit

OUT_DIR = ROOT / "analysis_outputs" / "all53_feature_selection_nested"
CURRENT_V3 = tuple(axis_audit.CURRENT_V3)
OUTER_SPLITS = 5
INNER_SPLITS = axis_audit.INNER_SPLITS
MAX_MISSING_RATE = 0.20
MAX_ABS_SPEARMAN = axis_audit.MAX_ABS_SPEARMAN
MAX_VIF = axis_audit.MAX_VIF
TOP_K = 10
MAX_FEATURES = 4
SELECTION_TOLERANCE = 0.002


def univariate_rank(
    train: pd.DataFrame,
    y: np.ndarray,
    candidates: list[str],
    seed: int,
) -> list[tuple[str, float]]:
    ranked = []
    for index, feature in enumerate(candidates):
        probability = axis_audit.inner_oof(
            train,
            y,
            (feature,),
            seed + index,
        )
        ranked.append((feature, float(roc_auc_score(y, probability))))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def correlation_prune(
    train: pd.DataFrame,
    ranked: list[tuple[str, float]],
) -> list[str]:
    kept: list[str] = []
    correlation = train[[feature for feature, _ in ranked]].corr(
        method="spearman"
    )
    for feature, _ in ranked:
        if all(
            not np.isfinite(correlation.loc[feature, existing])
            or abs(correlation.loc[feature, existing]) <= MAX_ABS_SPEARMAN
            for existing in kept
        ):
            kept.append(feature)
        if len(kept) == TOP_K:
            break
    return kept


def select_from_all53(
    train: pd.DataFrame,
    y: np.ndarray,
    candidates: list[str],
    seed: int,
) -> tuple[tuple[str, ...], float, float, float, float, dict]:
    missing_rate = train[candidates].isna().mean()
    eligible = [
        feature
        for feature in candidates
        if missing_rate[feature] <= MAX_MISSING_RATE
        and train[feature].notna().sum() >= 10
        and train[feature].nunique(dropna=True) > 1
    ]
    ranked = univariate_rank(train, y, eligible, seed)
    screened = correlation_prune(train, ranked)

    results = []
    n_vif_rejected = 0
    for size in range(1, min(MAX_FEATURES, len(screened)) + 1):
        for combo in combinations(screened, size):
            max_corr, max_vif = axis_audit.diagnostics(train, combo)
            if max_corr > MAX_ABS_SPEARMAN or max_vif > MAX_VIF:
                n_vif_rejected += 1
                continue
            probability = axis_audit.inner_oof(train, y, combo, seed + 1000)
            results.append(
                {
                    "features": combo,
                    "auc": float(roc_auc_score(y, probability)),
                    "probability": probability,
                    "max_corr": max_corr,
                    "max_vif": max_vif,
                }
            )
    if not results:
        raise RuntimeError("No all53 combination passed the nested filters.")

    best_auc = max(result["auc"] for result in results)
    near_best = [
        result
        for result in results
        if result["auc"] >= best_auc - SELECTION_TOLERANCE
    ]
    selected = sorted(
        near_best,
        key=lambda result: (
            len(result["features"]),
            -result["auc"],
            result["features"],
        ),
    )[0]
    threshold = axis_audit.threshold_for_sensitivity(
        y, selected["probability"]
    )
    audit = {
        "n_initial": len(candidates),
        "n_eligible": len(eligible),
        "n_after_correlation_topk": len(screened),
        "screened_features": "|".join(screened),
        "screened_univariate_auc": "|".join(
            f"{feature}:{auc:.6f}"
            for feature, auc in ranked
            if feature in screened
        ),
        "n_admissible_combinations": len(results),
        "n_vif_rejected": n_vif_rejected,
    }
    return (
        selected["features"],
        selected["auc"],
        threshold,
        selected["max_corr"],
        selected["max_vif"],
        audit,
    )


def evaluate_fold(
    strategy: str,
    repeat: int,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    table: pd.DataFrame,
    candidates: list[str],
) -> tuple[dict, list[dict]]:
    train = table.iloc[train_index].reset_index(drop=True)
    test = table.iloc[test_index].reset_index(drop=True)
    y_train = train["target"].to_numpy(int)
    seed = 530000 + repeat * 10 + fold

    if strategy == "current_v3_fixed":
        features = CURRENT_V3
        probability_inner = axis_audit.inner_oof(
            train, y_train, features, seed
        )
        inner_auc = float(roc_auc_score(y_train, probability_inner))
        threshold = axis_audit.threshold_for_sensitivity(
            y_train, probability_inner
        )
        max_corr, max_vif = axis_audit.diagnostics(train, features)
        audit = {
            "n_initial": len(features),
            "n_eligible": len(features),
            "n_after_correlation_topk": len(features),
            "screened_features": "|".join(features),
            "screened_univariate_auc": "",
            "n_admissible_combinations": 1,
            "n_vif_rejected": 0,
        }
    elif strategy == "all53_nested_selection":
        (
            features,
            inner_auc,
            threshold,
            max_corr,
            max_vif,
            audit,
        ) = select_from_all53(train, y_train, candidates, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    model = axis_audit.make_model().fit(train[list(features)], y_train)
    probability = model.predict_proba(test[list(features)])[:, 1]
    prediction = (probability >= threshold).astype(int)
    selection = {
        "strategy": strategy,
        "repeat": repeat,
        "fold": fold,
        "selected_features": "|".join(features),
        "n_selected": len(features),
        "inner_oof_auc": inner_auc,
        "threshold": threshold,
        "selected_max_abs_spearman": max_corr,
        "selected_max_vif": max_vif,
        **audit,
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
            test["subject_id"],
            test["target"],
            probability,
            prediction,
        )
    ]
    return selection, predictions


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (strategy, repeat), part in predictions.groupby(
        ["strategy", "repeat"]
    ):
        y = part["target"].to_numpy(int)
        probability = part["probability"].to_numpy(float)
        prediction = part["prediction"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "strategy": strategy,
                "repeat": repeat,
                "auc": roc_auc_score(y, probability),
                "sensitivity": recall_score(
                    y, prediction, zero_division=0
                ),
                "specificity": tn / (tn + fp),
                "precision": precision_score(
                    y, prediction, zero_division=0
                ),
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
    table, pools = axis_audit.build_table()
    candidates = sorted(set(pools["v"] + pools["ap"] + pools["ml"]))
    if len(candidates) != 53:
        raise ValueError(f"Expected 53 candidates, found {len(candidates)}")
    table.to_csv(
        OUT_DIR / "subject_feature_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y = table["target"].to_numpy(int)
    tasks = []
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=440000 + repeat,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(table, y)
        ):
            for strategy in (
                "current_v3_fixed",
                "all53_nested_selection",
            ):
                tasks.append(
                    (
                        strategy,
                        repeat,
                        fold,
                        train_index,
                        test_index,
                    )
                )

    print(
        f"subjects={len(table)} normal={(y == 0).sum()} "
        f"impaired={(y == 1).sum()} candidates={len(candidates)} "
        f"tasks={len(tasks)}",
        flush=True,
    )
    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_fold)(
            strategy,
            repeat,
            fold,
            train_index,
            test_index,
            table,
            candidates,
        )
        for strategy, repeat, fold, train_index, test_index in tasks
    )
    selections = pd.DataFrame([output[0] for output in outputs])
    predictions = pd.DataFrame(
        [row for _, prediction_rows in outputs for row in prediction_rows]
    )
    metrics, summary = summarize(predictions)

    selected_part = selections[
        selections["strategy"].eq("all53_nested_selection")
    ]
    feature_counter = Counter()
    for value in selected_part["selected_features"]:
        feature_counter.update(value.split("|"))
    feature_frequency = pd.DataFrame(
        [
            {
                "feature": feature,
                "selected_folds": feature_counter.get(feature, 0),
                "selection_rate": feature_counter.get(feature, 0)
                / len(selected_part),
            }
            for feature in candidates
        ]
    ).sort_values(["selection_rate", "feature"], ascending=[False, True])
    combination_frequency = (
        selected_part.groupby("selected_features")
        .size()
        .rename("selected_folds")
        .reset_index()
        .sort_values("selected_folds", ascending=False)
    )
    combination_frequency["selection_rate"] = (
        combination_frequency["selected_folds"] / len(selected_part)
    )

    selections.to_csv(
        OUT_DIR / "outer_fold_selections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUT_DIR / "outer_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUT_DIR / "metrics_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_frequency.to_csv(
        OUT_DIR / "feature_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combination_frequency.to_csv(
        OUT_DIR / "combination_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "n_subjects": int(len(table)),
        "normal": int((y == 0).sum()),
        "impaired": int((y == 1).sum()),
        "label_definition": "revised clinical OR label without s3_velocity",
        "n_initial_candidates": len(candidates),
        "candidate_axes": {
            axis: len(features) for axis, features in pools.items()
        },
        "outer_cv": f"StratifiedKFold(5) x {args.repeats} repeats",
        "inner_cv": "StratifiedKFold(3)",
        "selection_inside_outer_training": [
            "missingness and variance eligibility",
            "univariate inner-OOF AUC ranking",
            "Spearman redundancy pruning",
            f"top {TOP_K} retained candidates",
            "enumerate 1-4 feature combinations",
            "VIF <= 5",
            "inner-OOF AUC selection",
            "inner-OOF threshold for sensitivity >= 0.80",
        ],
        "current_model_modified": False,
        "source_condition": "one preselected 20-second segment per subject",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nNested outer-CV summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop selected features")
    print(feature_frequency.head(15).to_string(index=False))
    print("\nTop selected combinations")
    print(combination_frequency.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
