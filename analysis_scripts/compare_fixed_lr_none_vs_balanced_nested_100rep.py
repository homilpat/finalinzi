"""Paired nested-CV comparison of fixed LR class-weight settings.

The only model difference is class_weight=None versus "balanced". Thresholds
are selected separately inside each outer-training fold from 3-fold inner-OOF
probabilities: sensitivity >= 0.80, then maximum specificity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_acf_step_stride_symmetry_iqr",
]
ALIASES = {"v_harmonic_ratio_iqr": "v_acf_step_stride_symmetry_iqr"}
N_REPEATS = 100
N_SPLITS = 5
TARGET_SENSITIVITY = float(os.environ.get("TARGET_SENSITIVITY", "0.80"))
OUTPUT_SUFFIX = (
    "with_s3_fixed_lr_none_vs_balanced_nested_100rep"
    if TARGET_SENSITIVITY == 0.80
    else (
        "with_s3_fixed_lr_none_vs_balanced_nested_100rep_"
        f"sens{round(TARGET_SENSITIVITY * 100):02d}"
    )
)
OUT_DIR = ROOT / "analysis_outputs" / OUTPUT_SUFFIX
BASE_OUTER_SEED = 810000
BASE_MODEL_SEED = 900000


def make_lr(class_weight: str | None, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def inner_oof(
    model: Pipeline, x: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x, y):
        fitted = clone(model)
        fitted.fit(x[train_idx], y[train_idx])
        probability[valid_idx] = fitted.predict_proba(x[valid_idx])[:, 1]
    return probability


def select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    values = np.unique(probability)
    mids = (values[:-1] + values[1:]) / 2
    candidates = np.r_[values.min() - 1e-9, mids, values.max() + 1e-9]
    best_threshold = float(candidates[0])
    best_specificity = -1.0
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        if (
            sensitivity >= TARGET_SENSITIVITY
            and specificity > best_specificity
        ):
            best_threshold = float(threshold)
            best_specificity = float(specificity)
    return best_threshold


def metrics(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "sensitivity": float(recall_score(y, prediction)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE_PATH).rename(columns=ALIASES)
    x = table[FEATURES].to_numpy(float)
    y = table["target"].to_numpy(int)
    subject_ids = table["subject_id"].astype(str).to_numpy()
    configurations = {"None LR": None, "Balanced LR": "balanced"}
    metric_rows = []
    prediction_rows = []

    for repeat in range(N_REPEATS):
        outer = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=BASE_OUTER_SEED + repeat,
        )
        folds = list(outer.split(x, y))
        for name, class_weight in configurations.items():
            oof_probability = np.full(len(y), np.nan)
            oof_prediction = np.zeros(len(y), dtype=int)
            oof_threshold = np.full(len(y), np.nan)
            for fold, (train_idx, test_idx) in enumerate(folds):
                seed = BASE_MODEL_SEED + repeat * 100 + fold
                model = make_lr(class_weight, seed)
                inner_probability = inner_oof(
                    model, x[train_idx], y[train_idx], seed + 50000
                )
                threshold = select_threshold(
                    y[train_idx], inner_probability
                )
                model.fit(x[train_idx], y[train_idx])
                probability = model.predict_proba(x[test_idx])[:, 1]
                oof_probability[test_idx] = probability
                oof_prediction[test_idx] = probability >= threshold
                oof_threshold[test_idx] = threshold

            metric_rows.append(
                {
                    "model": name,
                    "repeat": repeat,
                    "threshold": float(np.median(oof_threshold)),
                    **metrics(y, oof_probability, oof_prediction),
                }
            )
            prediction_rows.extend(
                {
                    "model": name,
                    "repeat": repeat,
                    "subject_id": subject_id,
                    "target": int(target),
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "threshold": float(threshold),
                }
                for subject_id, target, probability, prediction, threshold in zip(
                    subject_ids,
                    y,
                    oof_probability,
                    oof_prediction,
                    oof_threshold,
                )
            )
        if (repeat + 1) % 10 == 0:
            print(f"repeat {repeat + 1}/{N_REPEATS}", flush=True)

    metric_table = pd.DataFrame(metric_rows)
    summary_rows = []
    for name, part in metric_table.groupby("model", sort=False):
        row = {"model": name, "n_repeats": int(len(part))}
        for column in [
            "auc",
            "sensitivity",
            "specificity",
            "precision",
            "accuracy",
            "f1",
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ]:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    metric_table.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        OUT_DIR / "oof_predictions.csv", index=False
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    metadata = {
        "subjects": int(len(y)),
        "normal": int((y == 0).sum()),
        "impaired": int((y == 1).sum()),
        "label": "with-s3 clinical OR label",
        "outer_cv": "StratifiedKFold(5) x 100 paired repeats",
        "inner_cv": "3-fold OOF within each outer-training fold",
        "threshold": (
            f"sensitivity >= {TARGET_SENSITIVITY:.2f}, "
            "then maximum specificity"
        ),
        "fixed_parameters": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "compared_parameter": "class_weight=None versus balanced",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
