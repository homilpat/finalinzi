"""Nested Optuna comparison of LR, GBM, RF, and SVM on the no-s3 cohort.

Every model uses the same subject-level outer folds. Hyperparameters, decision
thresholds, and the optional model-family selection are determined exclusively
inside each outer-training fold.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
)
TABLE_PATH = SOURCE_DIR / "subject_feature_table.csv"
OUT_DIR = ROOT / "analysis_outputs" / "no_s3_optuna_4model_nested_100rep"

FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_acf_step_stride_symmetry_iqr",
]
ALIASES = {"v_harmonic_ratio_iqr": "v_acf_step_stride_symmetry_iqr"}
MODEL_NAMES = ["LR", "GBM", "RF", "SVM"]
TARGET_SENSITIVITY = 0.80
N_SPLITS = 5
BASE_OUTER_SEED = 810000
BASE_MODEL_SEED = 910000


def make_pipeline(model_name: str, params: dict, seed: int) -> Pipeline:
    if model_name == "LR":
        penalty = params.get("penalty", "l2")
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=penalty,
            class_weight=params.get("class_weight"),
            solver="liblinear" if penalty == "l1" else "lbfgs",
            max_iter=3000,
            random_state=seed,
        )
    elif model_name == "GBM":
        estimator = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            subsample=float(params.get("subsample", 0.7)),
            max_features=params.get("max_features"),
            random_state=seed,
        )
    elif model_name == "RF":
        max_depth = params.get("max_depth", 0)
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if int(max_depth) == 0 else int(max_depth),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=1,
            random_state=seed,
        )
    elif model_name == "SVM":
        estimator = SVC(
            C=float(params.get("C", 1.0)),
            gamma=float(params.get("gamma", 0.1)),
            class_weight=params.get("class_weight", "balanced"),
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", estimator),
        ]
    )


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "LR":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if model_name == "GBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 180),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 2, 15
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 200, step=25
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [0, 2, 3, 4, 6, 8, 12]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 12
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    raise ValueError(f"Unknown model: {model_name}")


def inner_splits(x: np.ndarray, y: np.ndarray, seed: int) -> list:
    return list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y)
    )


def oof_probability(
    model: Pipeline, x: np.ndarray, y: np.ndarray, splits: list
) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    for train_idx, valid_idx in splits:
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


def tune_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    n_trials: int,
) -> tuple[dict, float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        model = make_pipeline(model_name, params, seed)
        scores = []
        for train_idx, valid_idx in splits:
            fitted = clone(model)
            fitted.fit(x[train_idx], y[train_idx])
            probability = fitted.predict_proba(x[valid_idx])[:, 1]
            scores.append(roc_auc_score(y[valid_idx], probability))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
        gc_after_trial=False,
    )
    return dict(study.best_params), float(study.best_value)


def metrics(
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


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in frame.groupby("model", sort=False):
        row = {"model": name, "n_repeats": int(len(part))}
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
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def paired_against_lr(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index="repeat", columns="model")
    rows = []
    for name in ["GBM", "RF", "SVM", "AutoSelect"]:
        for metric_name in [
            "auc",
            "sensitivity",
            "specificity",
            "precision",
            "accuracy",
            "f1",
        ]:
            difference = (
                wide[metric_name][name] - wide[metric_name]["LR"]
            ).to_numpy()
            try:
                p_value = float(wilcoxon(difference).pvalue)
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "model": name,
                    "metric": metric_name,
                    "mean_difference_vs_lr": float(np.mean(difference)),
                    "difference_ci_lo": float(
                        np.quantile(difference, 0.025)
                    ),
                    "difference_ci_hi": float(
                        np.quantile(difference, 0.975)
                    ),
                    "wins": int(np.sum(difference > 0)),
                    "ties": int(np.sum(difference == 0)),
                    "losses": int(np.sum(difference < 0)),
                    "wilcoxon_p_exploratory": p_value,
                }
            )
    return pd.DataFrame(rows)


def save_consensus_plots(predictions: pd.DataFrame) -> pd.DataFrame:
    consensus_rows = []
    for name, part in predictions.groupby("model", sort=False):
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
        result = metrics(y, probability, prediction)
        consensus_rows.append({"model": name, **result})

        fpr, tpr, _ = roc_curve(y, probability)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {result['auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"{name}: subject-consensus ROC",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        stem = name.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{stem}_consensus_roc.png", dpi=180)
        plt.close(fig)

        cm = confusion_matrix(y, prediction, labels=[0, 1])
        row_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
        for row in range(2):
            for column in range(2):
                value = row_percent[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value >= 50 else "black",
                    fontsize=13,
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Impaired"],
            yticklabels=["Normal", "Impaired"],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{name}: subject-consensus confusion matrix",
        )
        fig.colorbar(image, ax=ax, label="Row percentage")
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"{stem}_consensus_confusion_percent.png", dpi=180
        )
        plt.close(fig)
    result = pd.DataFrame(consensus_rows).sort_values(
        "auc", ascending=False
    )
    result.to_csv(OUT_DIR / "subject_consensus_metrics.csv", index=False)
    return result


def run(repeats: int, trials: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    table = pd.read_csv(TABLE_PATH).rename(columns=ALIASES)
    required = FEATURES + ["subject_id", "target"]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    table = table.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    x = table[FEATURES].to_numpy(float)
    y = table["target"].to_numpy(int)
    subject_ids = table["subject_id"].astype(str).to_numpy()

    metric_rows = []
    prediction_rows = []
    parameter_rows = []
    selection_rows = []
    for repeat in range(repeats):
        if repeat % 5 == 0:
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
            for name in MODEL_NAMES + ["AutoSelect"]
        }

        for fold, (train_idx, test_idx) in enumerate(outer.split(x, y)):
            seed = BASE_MODEL_SEED + repeat * 100 + fold
            splits = inner_splits(x[train_idx], y[train_idx], seed + 50000)
            candidates = {}
            for model_offset, name in enumerate(MODEL_NAMES):
                model_seed = seed + model_offset * 1000000
                params, tuning_auc = tune_model(
                    name,
                    x[train_idx],
                    y[train_idx],
                    splits,
                    model_seed,
                    trials,
                )
                model = make_pipeline(name, params, model_seed)
                inner_probability = oof_probability(
                    model, x[train_idx], y[train_idx], splits
                )
                threshold = threshold_for_sensitivity(
                    y[train_idx], inner_probability
                )
                inner_prediction = (
                    inner_probability >= threshold
                ).astype(int)
                inner_result = metrics(
                    y[train_idx], inner_probability, inner_prediction
                )
                model.fit(x[train_idx], y[train_idx])
                test_probability = model.predict_proba(x[test_idx])[:, 1]
                test_prediction = (test_probability >= threshold).astype(int)

                storage[name]["probability"][test_idx] = test_probability
                storage[name]["prediction"][test_idx] = test_prediction
                storage[name]["threshold"][test_idx] = threshold
                storage[name]["inner_auc"].append(inner_result["auc"])
                candidates[name] = {
                    "probability": test_probability,
                    "prediction": test_prediction,
                    "threshold": threshold,
                    "inner_auc": inner_result["auc"],
                    "inner_specificity": inner_result["specificity"],
                }
                parameter_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": name,
                        "tuning_cv_auc": tuning_auc,
                        "inner_oof_auc": inner_result["auc"],
                        "inner_oof_sensitivity": inner_result["sensitivity"],
                        "inner_oof_specificity": inner_result["specificity"],
                        "threshold": threshold,
                        "params_json": json.dumps(
                            params, ensure_ascii=False, sort_keys=True
                        ),
                    }
                )

            selected_name = max(
                MODEL_NAMES,
                key=lambda name: (
                    candidates[name]["inner_specificity"],
                    candidates[name]["inner_auc"],
                    -MODEL_NAMES.index(name),
                ),
            )
            selected = candidates[selected_name]
            storage["AutoSelect"]["probability"][test_idx] = selected[
                "probability"
            ]
            storage["AutoSelect"]["prediction"][test_idx] = selected[
                "prediction"
            ]
            storage["AutoSelect"]["threshold"][test_idx] = selected[
                "threshold"
            ]
            storage["AutoSelect"]["inner_auc"].append(selected["inner_auc"])
            selection_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "selected_model": selected_name,
                    "inner_specificity": selected["inner_specificity"],
                    "inner_auc": selected["inner_auc"],
                }
            )

        for name, values in storage.items():
            result = metrics(
                y, values["probability"], values["prediction"]
            )
            metric_rows.append(
                {
                    "model": name,
                    "repeat": repeat,
                    "inner_auc": float(np.mean(values["inner_auc"])),
                    "threshold": float(np.median(values["threshold"])),
                    **result,
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
                    values["probability"],
                    values["prediction"],
                    values["threshold"],
                )
            )

    metric_frame = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    parameter_frame = pd.DataFrame(parameter_rows)
    selection_frame = pd.DataFrame(selection_rows)
    summary = summarize(metric_frame)
    paired = paired_against_lr(metric_frame)
    consensus = save_consensus_plots(prediction_frame)

    metric_frame.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)
    prediction_frame.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    parameter_frame.to_csv(
        OUT_DIR / "optuna_best_params_by_outer_fold.csv", index=False
    )
    selection_frame.to_csv(
        OUT_DIR / "inner_model_selections.csv", index=False
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_vs_lr.csv", index=False)

    metadata = {
        "features": FEATURES,
        "subjects": int(len(table)),
        "normal_subjects": int((y == 0).sum()),
        "impaired_subjects": int((y == 1).sum()),
        "models": MODEL_NAMES,
        "repeats": repeats,
        "outer_cv": "subject-level StratifiedKFold(5), identical paired folds",
        "inner_cv": "3-fold StratifiedKFold within outer-training only",
        "optuna_trials_per_model_per_outer_fold": trials,
        "optuna_objective": "mean inner-CV ROC AUC",
        "threshold": (
            "inner OOF sensitivity >= 0.80, then maximum specificity"
        ),
        "auto_select": (
            "within outer-training: maximum thresholded inner specificity, "
            "then inner AUC"
        ),
        "auto_select_counts": dict(
            Counter(selection_frame["selected_model"])
        ),
        "note": (
            "Individual tuned-family results are paired descriptive estimates. "
            "AutoSelect is the fully nested estimate of model-family selection. "
            "Repeated-CV Wilcoxon p-values are exploratory because repeats are "
            "correlated."
        ),
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nSubject consensus")
    print(consensus.to_string(index=False))
    print("\nAutoSelect counts")
    print(metadata["auto_select_counts"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    arguments = parse_args()
    run(arguments.repeats, arguments.trials)
