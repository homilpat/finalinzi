"""Paired nested-CV validation of the deployed no-s3 LR and Optuna LR.

Hyperparameters and the sensitivity-constrained decision threshold are selected
using only each outer-training fold. The same outer folds are used for both
models so their repeat-level performance can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon
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
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
)
TABLE_PATH = SOURCE_DIR / "subject_feature_table.csv"
OUT_DIR = ROOT / "analysis_outputs" / "no_s3_lr_optuna_paired_nested_100rep"

FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_acf_step_stride_symmetry_iqr",
]
ALIASES = {"v_harmonic_ratio_iqr": "v_acf_step_stride_symmetry_iqr"}
TARGET_SENSITIVITY = 0.80
N_SPLITS = 5
BASE_OUTER_SEED = 810000
BASE_MODEL_SEED = 900000


def make_lr(
    *,
    c: float = 1.0,
    penalty: str = "l2",
    class_weight: str | None = None,
    seed: int = 0,
) -> Pipeline:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    penalty=penalty,
                    class_weight=class_weight,
                    solver=solver,
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def inner_oof(model: Pipeline, x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x, y):
        fitted = clone(model)
        fitted.fit(x[train_idx], y[train_idx])
        probability[valid_idx] = fitted.predict_proba(x[valid_idx])[:, 1]
    return probability


def threshold_for_sensitivity(
    y_true: np.ndarray, probability: np.ndarray
) -> float:
    values = np.unique(probability[np.isfinite(probability)])
    if not len(values):
        return 0.5
    mids = (
        (values[:-1] + values[1:]) / 2
        if len(values) > 1
        else np.asarray([], dtype=float)
    )
    candidates = np.r_[values.min() - 1e-9, mids, values.max() + 1e-9]
    best_threshold = float(candidates[0])
    best_specificity = -1.0
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if (
            sensitivity >= TARGET_SENSITIVITY
            and specificity > best_specificity
        ):
            best_threshold = float(threshold)
            best_specificity = specificity
    return best_threshold


def tune_lr(
    x: np.ndarray, y: np.ndarray, seed: int, n_trials: int
) -> tuple[dict, float]:
    splitter = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y)
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "c": trial.suggest_float("c", 1e-4, 1e3, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
        model = make_lr(**params, seed=seed)
        scores = []
        for train_idx, valid_idx in splitter:
            fitted = clone(model)
            fitted.fit(x[train_idx], y[train_idx])
            probability = fitted.predict_proba(x[valid_idx])[:, 1]
            scores.append(roc_auc_score(y[valid_idx], probability))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=False,
    )
    return dict(study.best_params), float(study.best_value)


def prediction_metrics(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in metrics.groupby("model", sort=False):
        row = {"model": model, "n_repeats": int(len(part))}
        for column in [
            "inner_auc",
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
        row["auc_ci_lo"] = float(part["auc"].quantile(0.025))
        row["auc_ci_hi"] = float(part["auc"].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="repeat", columns="model")
    rows = []
    for metric in [
        "auc",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "f1",
    ]:
        difference = (
            wide[metric]["Optuna LR"] - wide[metric]["Fixed LR"]
        ).to_numpy()
        try:
            p_value = float(wilcoxon(difference).pvalue)
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "metric": metric,
                "mean_difference_optuna_minus_fixed": float(
                    np.mean(difference)
                ),
                "difference_ci_lo": float(np.quantile(difference, 0.025)),
                "difference_ci_hi": float(np.quantile(difference, 0.975)),
                "wins": int(np.sum(difference > 0)),
                "ties": int(np.sum(difference == 0)),
                "losses": int(np.sum(difference < 0)),
                "wilcoxon_p_exploratory": p_value,
            }
        )
    return pd.DataFrame(rows)


def save_consensus_plots(predictions: pd.DataFrame) -> None:
    for model, part in predictions.groupby("model", sort=False):
        subject = (
            part.groupby(["subject_id", "target"], as_index=False)
            .agg(
                probability=("probability", "mean"),
                positive_votes=("prediction", "sum"),
                n_votes=("prediction", "size"),
            )
        )
        subject["prediction"] = (
            subject["positive_votes"] > subject["n_votes"] / 2
        ).astype(int)
        y = subject["target"].to_numpy(int)
        probability = subject["probability"].to_numpy(float)
        prediction = subject["prediction"].to_numpy(int)

        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
        ax.set_title(f"{model}: subject-consensus ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"{model.lower().replace(' ', '_')}_consensus_roc.png",
            dpi=180,
        )
        plt.close(fig)

        cm = confusion_matrix(y, prediction, labels=[0, 1])
        row_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{row_percent[row, column]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if row_percent[row, column] >= 50 else "black",
                    fontsize=13,
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Impaired"],
            yticklabels=["Normal", "Impaired"],
            xlabel="Predicted",
            ylabel="Actual",
        )
        ax.set_title(f"{model}: subject-consensus confusion matrix")
        fig.colorbar(image, ax=ax, label="Row percentage")
        fig.tight_layout()
        fig.savefig(
            OUT_DIR
            / f"{model.lower().replace(' ', '_')}_consensus_confusion_percent.png",
            dpi=180,
        )
        plt.close(fig)


def run(repeats: int, n_trials: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    table = pd.read_csv(TABLE_PATH).rename(columns=ALIASES)
    missing = sorted(set(FEATURES + ["subject_id", "target"]) - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    table = table.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    x = table[FEATURES].to_numpy(float)
    y = table["target"].to_numpy(int)
    subject_ids = table["subject_id"].astype(str).to_numpy()

    metric_rows = []
    prediction_rows = []
    parameter_rows = []
    for repeat in range(repeats):
        if repeat % 10 == 0:
            print(f"repeat {repeat}/{repeats}", flush=True)
        outer = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=BASE_OUTER_SEED + repeat,
        )
        storage = {
            name: {
                "probability": np.full(len(y), np.nan),
                "prediction": np.zeros(len(y), dtype=int),
                "threshold": np.full(len(y), np.nan),
                "inner_auc": [],
            }
            for name in ["Fixed LR", "Optuna LR"]
        }
        for fold, (train_idx, test_idx) in enumerate(outer.split(x, y)):
            seed = BASE_MODEL_SEED + repeat * 100 + fold
            inner_seed = seed + 50000

            fixed = make_lr(seed=seed)
            fixed_inner = inner_oof(fixed, x[train_idx], y[train_idx], inner_seed)
            fixed_threshold = threshold_for_sensitivity(
                y[train_idx], fixed_inner
            )
            fixed.fit(x[train_idx], y[train_idx])
            fixed_probability = fixed.predict_proba(x[test_idx])[:, 1]
            storage["Fixed LR"]["probability"][test_idx] = fixed_probability
            storage["Fixed LR"]["prediction"][test_idx] = (
                fixed_probability >= fixed_threshold
            )
            storage["Fixed LR"]["threshold"][test_idx] = fixed_threshold
            storage["Fixed LR"]["inner_auc"].append(
                roc_auc_score(y[train_idx], fixed_inner)
            )

            params, _ = tune_lr(
                x[train_idx], y[train_idx], inner_seed, n_trials
            )
            tuned = make_lr(**params, seed=seed)
            tuned_inner = inner_oof(tuned, x[train_idx], y[train_idx], inner_seed)
            tuned_threshold = threshold_for_sensitivity(
                y[train_idx], tuned_inner
            )
            tuned.fit(x[train_idx], y[train_idx])
            tuned_probability = tuned.predict_proba(x[test_idx])[:, 1]
            storage["Optuna LR"]["probability"][test_idx] = tuned_probability
            storage["Optuna LR"]["prediction"][test_idx] = (
                tuned_probability >= tuned_threshold
            )
            storage["Optuna LR"]["threshold"][test_idx] = tuned_threshold
            storage["Optuna LR"]["inner_auc"].append(
                roc_auc_score(y[train_idx], tuned_inner)
            )
            parameter_rows.append(
                {"repeat": repeat, "fold": fold, **params}
            )

        for model_name, values in storage.items():
            probability = values["probability"]
            prediction = values["prediction"]
            metric_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "inner_auc": float(np.mean(values["inner_auc"])),
                    "threshold": float(np.median(values["threshold"])),
                    **prediction_metrics(y, probability, prediction),
                }
            )
            prediction_rows.extend(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "subject_id": subject_id,
                    "target": int(target),
                    "probability": float(prob),
                    "prediction": int(pred),
                    "threshold": float(threshold),
                }
                for subject_id, target, prob, pred, threshold in zip(
                    subject_ids,
                    y,
                    probability,
                    prediction,
                    values["threshold"],
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    summary = summarize(metrics)
    paired = paired_summary(metrics)
    metrics.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    parameters.to_csv(OUT_DIR / "optuna_best_params_by_fold.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_differences.csv", index=False)
    save_consensus_plots(predictions)

    auc_difference = float(
        paired.loc[
            paired["metric"].eq("auc"),
            "mean_difference_optuna_minus_fixed",
        ].iloc[0]
    )
    metadata = {
        "features": FEATURES,
        "subjects": int(len(table)),
        "normal_subjects": int((y == 0).sum()),
        "impaired_subjects": int((y == 1).sum()),
        "outer_cv": f"StratifiedKFold(5) x {repeats} paired repeats",
        "inner_cv": "3-fold StratifiedKFold within each outer-training fold",
        "optuna_trials_per_outer_fold": n_trials,
        "optuna_search_space": {
            "C": "log-uniform [1e-4, 1e3]",
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
        },
        "optuna_objective": "mean inner-CV ROC AUC",
        "threshold_selection": (
            "inner OOF sensitivity >= 0.80, then maximum specificity"
        ),
        "paired_auc_mean_difference": auc_difference,
        "note": (
            "Wilcoxon p-values are exploratory because repeated CV estimates "
            "are correlated."
        ),
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nPaired differences")
    print(paired.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    run(args.repeats, args.trials)
