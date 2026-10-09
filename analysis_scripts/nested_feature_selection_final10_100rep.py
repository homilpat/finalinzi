"""Fully nested selection of 1-4 features from the final 10 candidates.

Feature filtering, combination selection, preprocessing, and the screening
threshold are learned from each outer-training fold only. The deployed model
is not modified.
"""
from __future__ import annotations

import argparse
import json
import warnings
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "nested_feature_selection_final10_100rep"
)
SOURCE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "daily_subwindow_median_iqr"
    / "subwindow_median_iqr_table.csv"
)
LABEL_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)

LEGACY_TO_CANONICAL = {
    "v_harmonic_ratio_median": (
        "v_acf_step_stride_symmetry_median"
    ),
    "v_harmonic_ratio_iqr": "v_acf_step_stride_symmetry_iqr",
    "ap_harmonic_ratio_median": (
        "ap_acf_step_stride_symmetry_median"
    ),
    "ap_harmonic_ratio_iqr": "ap_acf_step_stride_symmetry_iqr",
}
CANDIDATES = [
    "v_acf_step_stride_symmetry_median",
    "v_acf_step_stride_symmetry_iqr",
    "ap_acf_step_stride_symmetry_median",
    "ap_acf_step_stride_symmetry_iqr",
    "v_stride_freq_hz_median",
    "v_stride_freq_hz_iqr",
    "ap_spec_entropy_median",
    "ap_spec_entropy_iqr",
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
]
TARGET_SENSITIVITY = 0.80
MAX_MISSING_RATE = 0.20
MAX_ABS_SPEARMAN = 0.85
MAX_VIF = 5.0
INNER_SPLITS = 3
OUTER_SPLITS = 5
SELECTION_TOLERANCE = 0.002


def build_subject_table() -> pd.DataFrame:
    segments = pd.read_csv(SOURCE_TABLE).rename(
        columns=LEGACY_TO_CANONICAL
    )
    missing = sorted(set(CANDIDATES) - set(segments.columns))
    if missing:
        raise ValueError(f"Missing candidate columns: {missing}")
    subject = (
        segments.groupby("subject_id", as_index=False)[CANDIDATES]
        .median()
    )
    labels = (
        pd.read_csv(LABEL_TABLE)[["subject_id", "target"]]
        .drop_duplicates("subject_id")
    )
    subject = subject.merge(labels, on="subject_id", how="inner")
    if subject["target"].nunique() != 2:
        raise ValueError("Both target classes are required.")
    return subject.sort_values("subject_id").reset_index(drop=True)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def vif_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.shape[1] == 1:
        return {frame.columns[0]: 1.0}
    values = SimpleImputer(strategy="median").fit_transform(frame)
    values = RobustScaler().fit_transform(values)
    result = {}
    for index, feature in enumerate(frame.columns):
        others = np.delete(values, index, axis=1)
        target = values[:, index]
        if np.nanstd(target) <= 1e-12:
            result[feature] = np.inf
            continue
        r2 = LinearRegression().fit(others, target).score(
            others, target
        )
        result[feature] = (
            float(1.0 / (1.0 - r2)) if r2 < 1.0 else np.inf
        )
    return result


def combo_diagnostics(
    frame: pd.DataFrame, combo: tuple[str, ...]
) -> tuple[float, float]:
    if len(combo) == 1:
        return 0.0, 1.0
    part = frame.loc[:, combo]
    correlation = part.corr(method="spearman").abs()
    upper = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    max_correlation = float(np.nanmax(upper.to_numpy()))
    max_vif = float(max(vif_values(part).values()))
    return max_correlation, max_vif


def inner_oof_probability(
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


def threshold_for_sensitivity(
    y: np.ndarray, probability: np.ndarray
) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0]),
                probability[np.isfinite(probability)],
            ]
        )
    )
    best_threshold = 0.5
    best_specificity = -1.0
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if (
            sensitivity >= TARGET_SENSITIVITY
            and specificity > best_specificity
        ):
            best_specificity = specificity
            best_threshold = float(threshold)
    return best_threshold


def select_features(
    train: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> tuple[tuple[str, ...], float, float, float, float, int]:
    missing_rate = train[CANDIDATES].isna().mean()
    eligible = [
        feature
        for feature in CANDIDATES
        if missing_rate[feature] <= MAX_MISSING_RATE
        and train[feature].notna().sum() >= 10
        and train[feature].nunique(dropna=True) > 1
    ]
    if not eligible:
        raise RuntimeError("No candidates passed the missingness filter.")

    results = []
    for size in range(1, min(4, len(eligible)) + 1):
        for combo in combinations(eligible, size):
            max_correlation, max_vif = combo_diagnostics(train, combo)
            if (
                max_correlation > MAX_ABS_SPEARMAN
                or max_vif > MAX_VIF
            ):
                continue
            probability = inner_oof_probability(
                train, y, combo, seed
            )
            auc = roc_auc_score(y, probability)
            results.append(
                {
                    "features": combo,
                    "auc": float(auc),
                    "max_abs_spearman": max_correlation,
                    "max_vif": max_vif,
                    "probability": probability,
                }
            )
    if not results:
        raise RuntimeError("No feature combination passed correlation/VIF.")

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
    threshold = threshold_for_sensitivity(
        y, selected["probability"]
    )
    return (
        selected["features"],
        selected["auc"],
        threshold,
        selected["max_abs_spearman"],
        selected["max_vif"],
        len(results),
    )


def evaluate_outer_fold(
    repeat: int,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    subject: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    train = subject.iloc[train_index].reset_index(drop=True)
    test = subject.iloc[test_index].reset_index(drop=True)
    y_train = train["target"].to_numpy(int)
    features, inner_auc, threshold, max_corr, max_vif, n_combos = (
        select_features(
            train,
            y_train,
            seed=920000 + repeat * 10 + fold,
        )
    )
    model = make_model()
    model.fit(train[list(features)], y_train)
    probability = model.predict_proba(test[list(features)])[:, 1]
    prediction = (probability >= threshold).astype(int)
    selection = {
        "repeat": repeat,
        "fold": fold,
        "n_train": len(train),
        "n_test": len(test),
        "selected_features": "|".join(features),
        "n_selected": len(features),
        "inner_oof_auc": inner_auc,
        "threshold": threshold,
        "selected_max_abs_spearman": max_corr,
        "selected_max_vif": max_vif,
        "n_admissible_combinations": n_combos,
    }
    predictions = [
        {
            "repeat": repeat,
            "fold": fold,
            "subject_id": subject_id,
            "target": int(target),
            "probability": float(probability_value),
            "prediction": int(prediction_value),
            "threshold": threshold,
            "selected_features": "|".join(features),
        }
        for subject_id, target, probability_value, prediction_value in zip(
            test["subject_id"],
            test["target"],
            probability,
            prediction,
        )
    ]
    return selection, predictions


def metrics_by_repeat(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, part in predictions.groupby("repeat", sort=True):
        y = part["target"].to_numpy(int)
        probability = part["probability"].to_numpy(float)
        prediction = part["prediction"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        rows.append(
            {
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
                "f1": f1_score(y, prediction, zero_division=0),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    return pd.DataFrame(rows)


def descriptive_diagnostics(subject: pd.DataFrame) -> None:
    correlation = subject[CANDIDATES].corr(method="spearman")
    correlation.to_csv(
        OUT_DIR / "descriptive_spearman_all_subjects.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"feature": feature, "vif": value}
            for feature, value in vif_values(subject[CANDIDATES]).items()
        ]
    ).to_csv(
        OUT_DIR / "descriptive_vif_all_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "feature": CANDIDATES,
            "missing_rate": subject[CANDIDATES].isna().mean().to_numpy(),
            "n_finite": subject[CANDIDATES].notna().sum().to_numpy(),
        }
    ).to_csv(
        OUT_DIR / "descriptive_missingness_all_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject = build_subject_table()
    subject.to_csv(
        OUT_DIR / "subject_candidate_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    descriptive_diagnostics(subject)

    tasks = []
    y = subject["target"].to_numpy(int)
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=810000 + repeat,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(subject, y)
        ):
            tasks.append(
                (repeat, fold, train_index, test_index)
            )
    print(
        f"subjects={len(subject)} normal={(y == 0).sum()} "
        f"impaired={(y == 1).sum()} outer_tasks={len(tasks)}",
        flush=True,
    )
    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_outer_fold)(
            repeat, fold, train_index, test_index, subject
        )
        for repeat, fold, train_index, test_index in tasks
    )
    selections = pd.DataFrame([output[0] for output in outputs])
    predictions = pd.DataFrame(
        [
            row
            for _, prediction_rows in outputs
            for row in prediction_rows
        ]
    )
    metrics = metrics_by_repeat(predictions)
    selections.to_csv(
        OUT_DIR / "outer_fold_feature_selections.csv",
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

    feature_frequency = pd.DataFrame(
        [
            {
                "feature": feature,
                "selected_folds": int(
                    selections["selected_features"]
                    .str.split("|", regex=False)
                    .apply(lambda values: feature in values)
                    .sum()
                ),
                "selection_rate": float(
                    selections["selected_features"]
                    .str.split("|", regex=False)
                    .apply(lambda values: feature in values)
                    .mean()
                ),
            }
            for feature in CANDIDATES
        ]
    ).sort_values("selection_rate", ascending=False)
    feature_frequency.to_csv(
        OUT_DIR / "feature_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combination_frequency = (
        selections.groupby(
            ["selected_features", "n_selected"], as_index=False
        )
        .agg(
            selected_folds=("fold", "size"),
            inner_auc_mean=("inner_oof_auc", "mean"),
        )
        .sort_values(
            ["selected_folds", "inner_auc_mean"],
            ascending=[False, False],
        )
    )
    combination_frequency.to_csv(
        OUT_DIR / "combination_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "experimental_feature_selection_deployed_model_unchanged",
        "subjects": len(subject),
        "normal_subjects": int((y == 0).sum()),
        "impaired_subjects": int((y == 1).sum()),
        "outer_cv": (
            f"StratifiedKFold({OUTER_SPLITS}) x {args.repeats} repeats"
        ),
        "inner_cv": f"StratifiedKFold({INNER_SPLITS})",
        "candidate_count": len(CANDIDATES),
        "combination_sizes": "1-4",
        "missing_rate_max": MAX_MISSING_RATE,
        "max_abs_spearman": MAX_ABS_SPEARMAN,
        "max_vif": MAX_VIF,
        "selection_tolerance_auc": SELECTION_TOLERANCE,
        "threshold_rule": (
            "inner OOF sensitivity >= 0.80 then maximum specificity"
        ),
        "auc_mean": float(metrics["auc"].mean()),
        "auc_std": float(metrics["auc"].std()),
        "auc_ci_2p5": float(metrics["auc"].quantile(0.025)),
        "auc_ci_97p5": float(metrics["auc"].quantile(0.975)),
        "sensitivity_mean": float(metrics["sensitivity"].mean()),
        "specificity_mean": float(metrics["specificity"].mean()),
        "accuracy_mean": float(metrics["accuracy"].mean()),
        "f1_mean": float(metrics["f1"].mean()),
        "deployed_model_modified": False,
        "aggregation_note": (
            "Each 20s segment contains six 10s windows at 2s steps; "
            "window features are summarized by median/IQR, then each "
            "subject is represented by the median across available segments."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFeature selection frequency")
    print(feature_frequency.to_string(index=False))
    print("\nTop combinations")
    print(combination_frequency.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
