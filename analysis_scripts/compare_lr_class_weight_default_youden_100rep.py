from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "with_s3_lr_class_weight_default_youden_100rep"
)
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
REPEATS = 100
N_SPLITS = 5
BASE_SEED = 20260728


def make_model(class_weight: str | None, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    index = int(np.argmax(tpr[finite] - fpr[finite]))
    return (
        float(thresholds[finite][index]),
        float((tpr[finite] - fpr[finite])[index]),
    )


def calculate_metrics(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, probability),
        "sensitivity": recall_score(y, prediction, zero_division=0),
        "specificity": tn / (tn + fp),
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE_PATH).sort_values("cache_index").reset_index(drop=True)
    x = table[FEATURES].to_numpy(float)
    y = table["target"].to_numpy(int)

    configurations = {"None": None, "Balanced": "balanced"}
    repeat_rows = []
    fold_threshold_rows = []

    for repeat in range(REPEATS):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=BASE_SEED + repeat,
        )
        splits = list(splitter.split(x, y))

        for name, class_weight in configurations.items():
            oof_probability = np.full(len(y), np.nan)
            predictions = {
                "fixed_0.5": np.full(len(y), -1, dtype=int),
                "train_youden": np.full(len(y), -1, dtype=int),
            }
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                model = make_model(
                    class_weight, BASE_SEED + repeat * 10 + fold
                )
                model.fit(x[train_idx], y[train_idx])
                train_probability = model.predict_proba(x[train_idx])[:, 1]
                test_probability = model.predict_proba(x[test_idx])[:, 1]
                threshold, train_youden_j = youden_threshold(
                    y[train_idx], train_probability
                )
                oof_probability[test_idx] = test_probability
                predictions["fixed_0.5"][test_idx] = test_probability >= 0.5
                predictions["train_youden"][test_idx] = (
                    test_probability >= threshold
                )
                fold_threshold_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "class_weight": name,
                        "youden_threshold": threshold,
                        "train_youden_j": train_youden_j,
                    }
                )

            for threshold_rule, prediction in predictions.items():
                repeat_rows.append(
                    {
                        "repeat": repeat + 1,
                        "class_weight": name,
                        "threshold_rule": threshold_rule,
                        **calculate_metrics(y, oof_probability, prediction),
                    }
                )

    repeat_table = pd.DataFrame(repeat_rows)
    threshold_table = pd.DataFrame(fold_threshold_rows)
    summary = (
        repeat_table.groupby(["class_weight", "threshold_rule"], sort=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    threshold_summary = (
        threshold_table.groupby("class_weight", sort=False)
        .agg(
            youden_threshold_mean=("youden_threshold", "mean"),
            youden_threshold_std=("youden_threshold", "std"),
            train_youden_j_mean=("train_youden_j", "mean"),
            train_youden_j_std=("train_youden_j", "std"),
        )
        .reset_index()
    )
    summary = summary.merge(threshold_summary, on="class_weight", how="left")
    repeat_table.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)
    threshold_table.to_csv(OUT_DIR / "fold_thresholds.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
