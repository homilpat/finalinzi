from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis_outputs" / "final_training_conditions_nested_model_comparison_100rep"
TABLE_PATH = SRC / "subject_feature_table.csv"
CACHE_PATH = SRC / "subject_all_segments_final3_cache.npz"
OUT_DIR = ROOT / "analysis_outputs" / "with_s3_six_model_library_defaults_100rep"

FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
MODEL_ORDER = ["LR", "SVM", "RF", "XGB", "CNN1D", "LSTM"]
N_SPLITS = 5
REPEATS = 100
BASE_SEED = 20260728
DL_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_ml_models(seed: int) -> dict[str, Pipeline]:
    # Only probability/random_state/n_jobs are operational additions.
    return {
        "LR": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(random_state=seed)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", SVC(probability=True, random_state=seed)),
            ]
        ),
        "RF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=seed)),
            ]
        ),
        "XGB": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    xgb.XGBClassifier(
                        random_state=seed,
                        eval_metric="logloss",
                        verbosity=0,
                    ),
                ),
            ]
        ),
    }


class MinimalCNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return self.fc(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class MinimalLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1]).squeeze(-1)


def fit_sequence_preprocessor(
    x_seq: np.ndarray, lengths: np.ndarray, train_idx: np.ndarray
) -> tuple[SimpleImputer, RobustScaler]:
    valid_rows = np.concatenate(
        [x_seq[i, : int(lengths[i])] for i in train_idx], axis=0
    )
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    scaler.fit(imputer.fit_transform(valid_rows))
    return imputer, scaler


def transform_sequences(
    x_seq: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    imputer: SimpleImputer,
    scaler: RobustScaler,
) -> np.ndarray:
    result = np.zeros((len(indices), x_seq.shape[1], x_seq.shape[2]), dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        length = int(lengths[src_i])
        valid = x_seq[src_i, :length]
        result[out_i, :length] = scaler.transform(imputer.transform(valid)).astype(
            np.float32
        )
    return result


def train_dl_model(
    model_name: str,
    x_seq: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    imputer, scaler = fit_sequence_preprocessor(x_seq, lengths, train_idx)
    x_train = transform_sequences(x_seq, lengths, train_idx, imputer, scaler)
    x_test = transform_sequences(x_seq, lengths, test_idx, imputer, scaler)

    model = MinimalCNN1D() if model_name == "CNN1D" else MinimalLSTM()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    train_x = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    train_len = torch.tensor(lengths[train_idx], dtype=torch.long, device=DEVICE)
    train_y = torch.tensor(y[train_idx], dtype=torch.float32, device=DEVICE)

    model.train()
    for _ in range(DL_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x, train_len)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(train_x, train_len)).cpu().numpy()
        test_prob = torch.sigmoid(
            model(
                torch.tensor(x_test, dtype=torch.float32, device=DEVICE),
                torch.tensor(lengths[test_idx], dtype=torch.long, device=DEVICE),
            )
        ).cpu().numpy()
    return train_prob, test_prob


def youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def metric_row(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "auc": roc_auc_score(y_true, probability),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_default_settings() -> None:
    models = make_ml_models(BASE_SEED)
    selected = {
        "LR": {
            key: models["LR"].named_steps["clf"].get_params()[key]
            for key in ["C", "penalty", "solver", "class_weight", "max_iter"]
        },
        "SVM": {
            key: models["SVM"].named_steps["clf"].get_params()[key]
            for key in ["C", "kernel", "gamma", "class_weight", "probability"]
        },
        "RF": {
            key: models["RF"].named_steps["clf"].get_params()[key]
            for key in [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "class_weight",
            ]
        },
        "XGB": {
            key: models["XGB"].named_steps["clf"].get_params().get(key)
            for key in [
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ]
        },
        "CNN1D": {
            "status": "minimal baseline; no library-defined default architecture",
            "architecture": "Conv1d(3,32,kernel=3,padding=1)-ReLU-AvgPool-Linear",
            "optimizer": "Adam defaults",
            "loss": "BCEWithLogitsLoss defaults",
            "epochs": DL_EPOCHS,
        },
        "LSTM": {
            "status": "minimal baseline; no library-defined default architecture",
            "architecture": "LSTM(input=3,hidden=32,layers=1)-Linear",
            "optimizer": "Adam defaults",
            "loss": "BCEWithLogitsLoss defaults",
            "epochs": DL_EPOCHS,
        },
        "shared_evaluation": {
            "split": f"subject-level StratifiedKFold({N_SPLITS}) x {REPEATS}",
            "preprocessing": "median imputation; RobustScaler for LR/SVM/CNN/LSTM, fit on training fold only",
            "threshold_rules": ["fixed_0.5", "train_youden"],
            "device": str(DEVICE),
        },
    }
    (OUT_DIR / "default_model_settings.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def plot_summary(summary: pd.DataFrame) -> None:
    fixed = summary[summary["threshold_rule"] == "fixed_0.5"].copy()
    fixed = fixed.sort_values("auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(fixed["model"], fixed["auc_mean"], color="#8EA4B8")
    for index, row in fixed.reset_index(drop=True).iterrows():
        ax.text(
            row["auc_mean"] + 0.003,
            index,
            f'{row["auc_mean"]:.3f}',
            va="center",
            fontsize=10,
        )
    ax.set_xlim(max(0.5, fixed["auc_mean"].min() - 0.06), 1.0)
    ax.set_xlabel("Mean subject-level OOF AUC (100 repeats)")
    ax.set_ylabel("")
    ax.set_title("Library-default baseline model comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "default_models_auc_comparison.png", dpi=220)
    fig.savefig(OUT_DIR / "default_models_auc_comparison.pdf")
    plt.close(fig)


def plot_consensus(
    y: np.ndarray,
    subjects: np.ndarray,
    probability_sum: dict[str, np.ndarray],
    prediction_votes: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    consensus_rows: list[dict[str, object]] = []
    fig_roc, ax_roc = plt.subplots(figsize=(7.2, 6.2))
    for model_name in MODEL_ORDER:
        mean_prob = probability_sum[model_name] / REPEATS
        fpr, tpr, _ = roc_curve(y, mean_prob)
        auc_value = roc_auc_score(y, mean_prob)
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{model_name} ({auc_value:.3f})")
        for rule in ["fixed_0.5", "train_youden"]:
            pred = (prediction_votes[(model_name, rule)] >= (REPEATS / 2)).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row = metric_row(y, mean_prob, 0.5)
            row.update(
                {
                    "model": model_name,
                    "threshold_rule": rule,
                    "consensus_tn": int(tn),
                    "consensus_fp": int(fp),
                    "consensus_fn": int(fn),
                    "consensus_tp": int(tp),
                    "consensus_sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                    "consensus_specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "consensus_accuracy": (tn + tp) / len(y),
                    "consensus_f1": 2 * tp / (2 * tp + fp + fn)
                    if 2 * tp + fp + fn
                    else np.nan,
                }
            )
            consensus_rows.append(row)

            matrix = np.array([[tn, fp], [fn, tp]])
            fig, ax = plt.subplots(figsize=(4.3, 3.8))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Normal", "Impaired"],
                yticklabels=["Normal", "Impaired"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name} - {rule}")
            fig.tight_layout()
            fig.savefig(
                OUT_DIR / f"confusion_{model_name.lower()}_{rule}.png", dpi=220
            )
            plt.close(fig)

        pd.DataFrame(
            {
                "subject": subjects,
                "label": y,
                "mean_oof_probability": mean_prob,
            }
        ).to_csv(OUT_DIR / f"consensus_probability_{model_name.lower()}.csv", index=False)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Mean OOF ROC curves over 100 repeats")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.2)
    fig_roc.tight_layout()
    fig_roc.savefig(OUT_DIR / "all_models_mean_oof_roc.png", dpi=220)
    fig_roc.savefig(OUT_DIR / "all_models_mean_oof_roc.pdf")
    plt.close(fig_roc)
    return pd.DataFrame(consensus_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE_PATH).sort_values("cache_index").reset_index(drop=True)
    cache = np.load(CACHE_PATH, allow_pickle=True)

    x_agg = table[FEATURES].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=int)
    subjects = table["subject_id"].astype(str).to_numpy()
    x_seq = cache["x_seq"].astype(float)
    lengths = np.minimum(table["n_segments"].to_numpy(dtype=int), x_seq.shape[1])

    print(
        f"Subjects={len(y)} normal={(y == 0).sum()} impaired={(y == 1).sum()} "
        f"device={DEVICE}"
    )
    save_default_settings()

    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    probability_sum = {name: np.zeros(len(y), dtype=float) for name in MODEL_ORDER}
    prediction_votes = {
        (name, rule): np.zeros(len(y), dtype=int)
        for name in MODEL_ORDER
        for rule in ["fixed_0.5", "train_youden"]
    }

    for repeat in range(REPEATS):
        split_seed = BASE_SEED + repeat
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=split_seed
        )
        splits = list(splitter.split(x_agg, y))
        repeat_prob = {name: np.full(len(y), np.nan) for name in MODEL_ORDER}
        repeat_pred = {
            (name, rule): np.full(len(y), -1, dtype=int)
            for name in MODEL_ORDER
            for rule in ["fixed_0.5", "train_youden"]
        }

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            ml_models = make_ml_models(split_seed * 10 + fold)
            fold_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for model_name, estimator in ml_models.items():
                fitted = clone(estimator).fit(x_agg[train_idx], y[train_idx])
                fold_probabilities[model_name] = (
                    fitted.predict_proba(x_agg[train_idx])[:, 1],
                    fitted.predict_proba(x_agg[test_idx])[:, 1],
                )

            for model_name in ["CNN1D", "LSTM"]:
                fold_probabilities[model_name] = train_dl_model(
                    model_name,
                    x_seq,
                    lengths,
                    y,
                    train_idx,
                    test_idx,
                    split_seed * 100 + fold,
                )

            for model_name, (train_prob, test_prob) in fold_probabilities.items():
                thresholds = {
                    "fixed_0.5": 0.5,
                    "train_youden": youden_threshold(y[train_idx], train_prob),
                }
                repeat_prob[model_name][test_idx] = test_prob
                for rule, threshold in thresholds.items():
                    pred = (test_prob >= threshold).astype(int)
                    repeat_pred[(model_name, rule)][test_idx] = pred
                    metrics = metric_row(y[test_idx], test_prob, threshold)
                    fold_rows.append(
                        {
                            "repeat": repeat + 1,
                            "fold": fold,
                            "model": model_name,
                            "threshold_rule": rule,
                            "threshold": threshold,
                            **metrics,
                        }
                    )

        for model_name in MODEL_ORDER:
            probability = repeat_prob[model_name]
            probability_sum[model_name] += probability
            for rule in ["fixed_0.5", "train_youden"]:
                pred = repeat_pred[(model_name, rule)]
                prediction_votes[(model_name, rule)] += pred
                tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                repeat_rows.append(
                    {
                        "repeat": repeat + 1,
                        "model": model_name,
                        "threshold_rule": rule,
                        "auc": roc_auc_score(y, probability),
                        "sensitivity": tp / (tp + fn),
                        "specificity": tn / (tn + fp),
                        "accuracy": (tn + tp) / len(y),
                        "precision": tp / (tp + fp) if tp + fp else 0.0,
                        "f1": 2 * tp / (2 * tp + fp + fn)
                        if 2 * tp + fp + fn
                        else 0.0,
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                )
                for subject_idx in range(len(y)):
                    oof_rows.append(
                        {
                            "repeat": repeat + 1,
                            "model": model_name,
                            "threshold_rule": rule,
                            "subject": subjects[subject_idx],
                            "label": y[subject_idx],
                            "probability": probability[subject_idx],
                            "prediction": pred[subject_idx],
                        }
                    )

        if (repeat + 1) % 10 == 0:
            print(f"Completed {repeat + 1}/{REPEATS} repeats", flush=True)

    repeat_df = pd.DataFrame(repeat_rows)
    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    summary = (
        repeat_df.groupby(["model", "threshold_rule"])
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
        .sort_values(["threshold_rule", "auc_mean"], ascending=[True, False])
    )

    repeat_df.to_csv(OUT_DIR / "repeat_level_metrics.csv", index=False)
    fold_df.to_csv(OUT_DIR / "fold_level_metrics.csv", index=False)
    oof_df.to_csv(OUT_DIR / "subject_oof_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    consensus = plot_consensus(
        y, subjects, probability_sum, prediction_votes
    )
    consensus.to_csv(OUT_DIR / "consensus_confusion_metrics.csv", index=False)
    plot_summary(summary)

    print("\nLibrary-default baseline summary")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
