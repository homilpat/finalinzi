"""Leakage-safe 100-repeat model comparison for one 60s free-walk measurement.

The deployed model is never modified. All fitted candidates and evaluation
artifacts are written under analysis_outputs/ only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
except Exception:
    TORCH_OK = False
    DEVICE = None


OUT_DIR = ROOT / "analysis_outputs" / "freewalk_60s_final3_model_comparison_100rep"
CACHE_PATH = OUT_DIR / "freewalk_60s_top3_cache.npz"
ASSIGNMENT_PATH = OUT_DIR / "repeat_segment_assignments.csv"
CANDIDATE_DIR = OUT_DIR / "candidate_models"
MODEL_RESULT_DIR = OUT_DIR / "model_results"
ROC_DIR = OUT_DIR / "roc_curves"
CM_DIR = OUT_DIR / "confusion_matrices"

GAIT_PROJECT = next(
    path for path in ROOT.parent.iterdir() if (path / "physionet_AWS").exists()
)
RAW_DIR = GAIT_PROJECT / "physionet_AWS"
V2_CSV = (
    RAW_DIR
    / "strict_preprocessing_runs"
    / "strict_preprocessed_accgyro_v2"
    / "gait_features_strict_20s_accgyro_v2.csv"
)
LEGACY_CACHE_PATH = (
    ROOT
    / "analysis_outputs"
    / "single_20s_segment_model_comparison_100rep"
    / "single_20s_segment_feature_cache.npz"
)
FS = 100
DURATION_SAMPLES = 60 * FS
WINDOW_SAMPLES = 10 * FS
WINDOW_STEP = 2 * FS
N_WINDOWS = 26
N_REPEATS = 100
N_SPLITS = 5
TARGET_SENSITIVITY = 0.80
BASE_SEED = 20260724

FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
SEQUENCE_FEATURES = ["v_jerk_rms", "v_harmonic_ratio"]
ML_MODELS = ["LR", "SVM", "RF", "GBM", "XGB", "Voting", "Stacking"]
DL_MODELS = ["CNN1D", "LSTM"]
ALL_MODELS = ML_MODELS + DL_MODELS


def parse_hea(subject_id: str) -> dict:
    lines = (RAW_DIR / f"{subject_id}.hea").read_text(
        encoding="utf-8"
    ).splitlines()
    parts = lines[0].split()
    sample_count, channel_count = int(parts[3]), int(parts[1])
    gains, baselines = [], []
    for line in lines[1 : 1 + channel_count]:
        match = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2)))
    return {
        "n": sample_count,
        "ch": channel_count,
        "gains": np.asarray(gains[:3]),
        "baselines": np.asarray(baselines[:3]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def make_rf(seed: int, n_estimators: int = 300) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def make_xgb(seed: int, n_estimators: int = 100) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def base_steps() -> list:
    return [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]


def make_model(name: str, seed: int) -> Pipeline:
    if name == "LR":
        estimator = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )
    elif name == "SVM":
        estimator = SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
    elif name == "RF":
        estimator = make_rf(seed)
    elif name == "GBM":
        estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            subsample=0.7,
            random_state=seed,
        )
    elif name == "XGB":
        estimator = make_xgb(seed)
    elif name == "Voting":
        estimator = VotingClassifier(
            voting="soft",
            n_jobs=1,
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
                ("rf", make_rf(seed, 200)),
                ("xgb", make_xgb(seed)),
            ],
        )
    elif name == "Stacking":
        estimator = StackingClassifier(
            cv=3,
            n_jobs=1,
            final_estimator=LogisticRegression(
                max_iter=3000, solver="liblinear", class_weight="balanced"
            ),
            estimators=[
                ("rf", make_rf(seed, 200)),
                ("xgb", make_xgb(seed)),
                (
                    "svm",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ],
        )
    else:
        raise ValueError(f"Unknown ML model: {name}")
    return Pipeline([*base_steps(), ("model", estimator)])


def strict_60s_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = np.load(LEGACY_CACHE_PATH, allow_pickle=True)
    labels = (
        pd.DataFrame(legacy["meta"].tolist())[["subject_id", "target"]]
        .drop_duplicates("subject_id")
    )
    valid20 = pd.read_csv(V2_CSV, usecols=["subject_id", "bout_idx", "start_sec"])
    valid20 = valid20.merge(labels, on="subject_id", how="inner")
    bouts = pd.read_csv(
        RAW_DIR
        / "strict_preprocessing_runs"
        / "strict_preprocessed_accgyro_v2"
        / "strict_bouts_all.csv"
    )
    bout_lookup = bouts.set_index(["subject_id", "bout_idx"])

    candidate_rows = []
    excluded_rows = []
    for subject_id, part in valid20.groupby("subject_id", sort=True):
        subject_candidates = []
        for bout_idx, bout_part in part.groupby("bout_idx"):
            bout = bout_lookup.loc[(subject_id, bout_idx)]
            bout_start = float(bout["start_sec"])
            bout_end = float(bout["end_sec"])
            starts = set(np.round(bout_part["start_sec"].to_numpy(float), 3))
            for start_sec in starts:
                strict_coverage = all(
                    round(start_sec + delta, 3) in starts for delta in range(0, 41, 5)
                )
                fits_bout = start_sec >= bout_start and start_sec + 60.0 <= bout_end
                if strict_coverage and fits_bout:
                    subject_candidates.append(
                        {
                            "subject_id": str(subject_id),
                            "target": int(part["target"].iloc[0]),
                            "bout_idx": int(bout_idx),
                            "start_sec": float(start_sec),
                        }
                    )
        if subject_candidates:
            candidate_rows.extend(subject_candidates)
        else:
            excluded_rows.append(
                {
                    "subject_id": str(subject_id),
                    "target": int(part["target"].iloc[0]),
                    "reason": "no_contiguous_strict_qc_60s_candidate",
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(excluded_rows)


def make_assignments(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(BASE_SEED + repeat)
        for subject_id, part in candidates.groupby("subject_id", sort=True):
            chosen = part.iloc[int(rng.integers(0, len(part)))]
            rows.append(
                {
                    "repeat": repeat,
                    "subject_id": str(subject_id),
                    "target": int(chosen["target"]),
                    "bout_idx": int(chosen["bout_idx"]),
                    "start_sec": float(chosen["start_sec"]),
                    "segment_id": (
                        f"{subject_id}__b{int(chosen['bout_idx'])}"
                        f"__s{float(chosen['start_sec']):.3f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def extract_segment(raw: np.memmap, header: dict, start_sec: float) -> tuple[np.ndarray, np.ndarray]:
    from gait_axis_aligned_core import window_features

    start = int(round(start_sec * FS))
    end = start + DURATION_SAMPLES
    acc = raw[start:end, :3].astype(float)
    acc = (acc - header["baselines"]) / header["gains"]
    sequence = []
    for offset in range(0, DURATION_SAMPLES - WINDOW_SAMPLES + 1, WINDOW_STEP):
        feature = window_features(acc[offset : offset + WINDOW_SAMPLES])
        sequence.append(
            [
                float(feature["v_jerk_rms"]),
                float(feature["v_harmonic_ratio"]),
            ]
        )
    sequence_array = np.asarray(sequence, dtype=np.float32)
    if sequence_array.shape != (N_WINDOWS, len(SEQUENCE_FEATURES)):
        raise ValueError(f"Unexpected sequence shape: {sequence_array.shape}")
    jerk = sequence_array[:, 0]
    harmonic_ratio = sequence_array[:, 1]
    aggregate = np.asarray(
        [
            np.nanmedian(jerk),
            np.nanpercentile(jerk, 75) - np.nanpercentile(jerk, 25),
            np.nanpercentile(harmonic_ratio, 75)
            - np.nanpercentile(harmonic_ratio, 25),
        ],
        dtype=np.float32,
    )
    return aggregate, sequence_array


def build_or_load_cache() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and ASSIGNMENT_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        return (
            pd.read_csv(ASSIGNMENT_PATH),
            cached["x_agg"].astype(np.float32),
            cached["x_seq"].astype(np.float32),
        )

    candidates, excluded = strict_60s_candidates()
    assignments = make_assignments(candidates)
    unique = (
        assignments[["subject_id", "target", "bout_idx", "start_sec", "segment_id"]]
        .drop_duplicates("segment_id")
        .sort_values(["subject_id", "start_sec"])
        .reset_index(drop=True)
    )
    unique["cache_index"] = np.arange(len(unique))
    lookup = unique.set_index("segment_id")["cache_index"]
    assignments["cache_index"] = assignments["segment_id"].map(lookup).astype(int)

    x_agg = np.full((len(unique), len(FEATURES)), np.nan, dtype=np.float32)
    x_seq = np.full(
        (len(unique), N_WINDOWS, len(SEQUENCE_FEATURES)), np.nan, dtype=np.float32
    )
    for subject_number, (subject_id, part) in enumerate(
        unique.groupby("subject_id", sort=True), start=1
    ):
        header = parse_hea(str(subject_id))
        raw = np.memmap(
            RAW_DIR / f"{subject_id}.dat",
            dtype="<i2",
            mode="r",
            shape=(header["n"], header["ch"]),
        )
        for _, row in part.iterrows():
            aggregate, sequence = extract_segment(raw, header, float(row["start_sec"]))
            index = int(row["cache_index"])
            x_agg[index] = aggregate
            x_seq[index] = sequence
        print(
            f"[cache] {subject_number:02d}/{unique['subject_id'].nunique()} "
            f"{subject_id}: {len(part)} unique 60s segments",
            flush=True,
        )

    print(
        f"[cache] aggregate_missing={int((~np.isfinite(x_agg)).sum())} "
        f"sequence_missing={int((~np.isfinite(x_seq)).sum())}",
        flush=True,
    )
    assignments.to_csv(ASSIGNMENT_PATH, index=False, encoding="utf-8-sig")
    unique.to_csv(OUT_DIR / "unique_cached_segments.csv", index=False, encoding="utf-8-sig")
    candidates.groupby(["subject_id", "target"]).size().rename("n_candidates").reset_index().to_csv(
        OUT_DIR / "candidate_counts_by_subject.csv", index=False, encoding="utf-8-sig"
    )
    excluded.to_csv(OUT_DIR / "excluded_subjects.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(CACHE_PATH, x_agg=x_agg, x_seq=x_seq)
    return assignments, x_agg, x_seq


def threshold_for_min_sensitivity(y_true: np.ndarray, probability: np.ndarray) -> float:
    values = np.unique(probability[np.isfinite(probability)])
    if not len(values):
        return 0.5
    mids = (values[:-1] + values[1:]) / 2 if len(values) > 1 else np.array([])
    candidates = np.r_[values.min() - 1e-9, mids, values.max() + 1e-9]
    best_threshold = float(candidates[0])
    best_specificity = -1.0
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= TARGET_SENSITIVITY and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = specificity
    return best_threshold


def metrics_from_predictions(
    y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def inner_oof_ml(
    model: Pipeline, x: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, valid_idx in inner.split(x, y):
        fitted = clone(model)
        fitted.fit(x[train_idx], y[train_idx])
        probability[valid_idx] = fitted.predict_proba(x[valid_idx])[:, 1]
    return probability


if TORCH_OK:
    class CNN1DNet(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(n_features, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.output = nn.Sequential(nn.Dropout(0.35), nn.Linear(16, 1))

        def forward(self, x):
            return self.output(self.network(x.transpose(1, 2)).squeeze(-1)).squeeze(1)


    class LSTMNet(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 16, batch_first=True)
            self.output = nn.Sequential(nn.Dropout(0.35), nn.Linear(16, 1))

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.output(hidden[-1]).squeeze(1)


def normalize_sequences(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    flat_train = x_train.reshape(-1, x_train.shape[-1])
    scaler.fit(imputer.fit_transform(flat_train))

    def transform(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(-1, values.shape[-1])
        return scaler.transform(imputer.transform(flat)).reshape(values.shape).astype(np.float32)

    return transform(x_train), transform(x_test)


def train_dl(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    return_state: bool = False,
) -> tuple[np.ndarray, dict | None]:
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not available.")
    set_seed(seed)
    model_class = CNN1DNet if model_name == "CNN1D" else LSTMNet
    model = model_class(x_train.shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)

    indices = np.arange(len(y_train))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=max(0.2, 2 / len(indices)),
        stratify=y_train,
        random_state=seed,
    )
    positives = max(1, int((y_train[train_idx] == 1).sum()))
    negatives = max(1, int((y_train[train_idx] == 0).sum()))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=DEVICE)
    )

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, dtype=dtype, device=DEVICE)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(60):
        model.train()
        optimizer.zero_grad()
        loss = criterion(
            model(tensor(x_train[train_idx])),
            tensor(y_train[train_idx].astype(np.float32)),
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = nn.BCEWithLogitsLoss()(
                model(tensor(x_train[valid_idx])),
                tensor(y_train[valid_idx].astype(np.float32)),
            ).item()
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
        elif epoch - best_epoch >= 8:
            break

    model.load_state_dict({name: value.to(DEVICE) for name, value in best_state.items()})
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor(x_test))).cpu().numpy()
    state = best_state if return_state else None
    return probability, state


def inner_oof_dl(
    model_name: str, x: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for inner_fold, (train_idx, valid_idx) in enumerate(inner.split(x, y)):
        x_train, x_valid = normalize_sequences(x[train_idx], x[valid_idx])
        probability[valid_idx], _ = train_dl(
            model_name,
            x_train,
            y[train_idx],
            x_valid,
            seed + inner_fold,
        )
    return probability


def selected_repeat_data(
    assignments: pd.DataFrame,
    x_agg_cache: np.ndarray,
    x_seq_cache: np.ndarray,
    repeat: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    chosen = (
        assignments[assignments["repeat"].eq(repeat)]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    indices = chosen["cache_index"].to_numpy(int)
    return chosen, x_agg_cache[indices], x_seq_cache[indices]


def evaluate_model(
    model_name: str,
    assignments: pd.DataFrame,
    x_agg_cache: np.ndarray,
    x_seq_cache: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for repeat in range(N_REPEATS):
        if repeat % 10 == 0:
            print(f"[{model_name}] repeat {repeat}/{N_REPEATS}", flush=True)
        chosen, x_agg, x_seq = selected_repeat_data(
            assignments, x_agg_cache, x_seq_cache, repeat
        )
        y = chosen["target"].to_numpy(int)
        subjects = chosen["subject_id"].astype(str).to_numpy()
        outer = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=810000 + repeat
        )
        oof_probability = np.full(len(y), np.nan)
        oof_prediction = np.zeros(len(y), dtype=int)
        oof_threshold = np.full(len(y), np.nan)
        train_auc = []

        for fold, (train_idx, test_idx) in enumerate(outer.split(x_agg, y)):
            if set(subjects[train_idx]) & set(subjects[test_idx]):
                raise AssertionError("Subject leakage detected.")
            seed = 900000 + repeat * 100 + fold
            if model_name in ML_MODELS:
                model = make_model(model_name, seed)
                inner_probability = inner_oof_ml(
                    model, x_agg[train_idx], y[train_idx], seed + 50000
                )
                threshold = threshold_for_min_sensitivity(
                    y[train_idx], inner_probability
                )
                model.fit(x_agg[train_idx], y[train_idx])
                test_probability = model.predict_proba(x_agg[test_idx])[:, 1]
            else:
                inner_probability = inner_oof_dl(
                    model_name, x_seq[train_idx], y[train_idx], seed + 50000
                )
                threshold = threshold_for_min_sensitivity(
                    y[train_idx], inner_probability
                )
                scaled_train, scaled_test = normalize_sequences(
                    x_seq[train_idx], x_seq[test_idx]
                )
                test_probability, _ = train_dl(
                    model_name,
                    scaled_train,
                    y[train_idx],
                    scaled_test,
                    seed,
                )

            train_auc.append(roc_auc_score(y[train_idx], inner_probability))
            oof_probability[test_idx] = test_probability
            oof_prediction[test_idx] = (test_probability >= threshold).astype(int)
            oof_threshold[test_idx] = threshold

        metric_rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "train_inner_oof_auc": float(np.mean(train_auc)),
                "threshold_median": float(np.median(oof_threshold)),
                **metrics_from_predictions(y, oof_probability, oof_prediction),
            }
        )
        repeat_predictions = chosen[
            ["repeat", "subject_id", "target", "segment_id", "start_sec"]
        ].copy()
        repeat_predictions["model"] = model_name
        repeat_predictions["probability"] = oof_probability
        repeat_predictions["threshold"] = oof_threshold
        repeat_predictions["prediction"] = oof_prediction
        prediction_rows.append(repeat_predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(
        MODEL_RESULT_DIR / f"{model_name}_metrics.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        MODEL_RESULT_DIR / f"{model_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics, predictions


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    for model_name in ALL_MODELS:
        metric_path = MODEL_RESULT_DIR / f"{model_name}_metrics.csv"
        prediction_path = MODEL_RESULT_DIR / f"{model_name}_predictions.csv"
        if metric_path.exists() and prediction_path.exists():
            metrics.append(pd.read_csv(metric_path))
            predictions.append(pd.read_csv(prediction_path))
    if not metrics:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("model")
        .agg(
            n_repeats=("repeat", "count"),
            train_inner_oof_auc_mean=("train_inner_oof_auc", "mean"),
            test_auc_mean=("auc", "mean"),
            test_auc_std=("auc", "std"),
            auc_ci_lo=("auc", lambda values: values.quantile(0.025)),
            auc_ci_hi=("auc", lambda values: values.quantile(0.975)),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            threshold_median=("threshold_median", "median"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
        )
        .reset_index()
    )
    summary["gap"] = (
        summary["train_inner_oof_auc_mean"] - summary["test_auc_mean"]
    )
    summary["eligible_sensitivity_ge_0p80"] = (
        summary["sensitivity_mean"] >= TARGET_SENSITIVITY
    )
    summary["eligible_auc_rank"] = np.nan
    eligible = summary["eligible_sensitivity_ge_0p80"]
    summary.loc[eligible, "eligible_auc_rank"] = (
        summary.loc[eligible, "test_auc_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary.sort_values(
        ["sensitivity_mean", "test_auc_mean", "specificity_mean"],
        ascending=[False, False, False],
    )


def save_plots(predictions: pd.DataFrame) -> None:
    ROC_DIR.mkdir(exist_ok=True)
    CM_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    confusion_rows = []
    for model_name, part in predictions.groupby("model"):
        y = part["target"].to_numpy(int)
        probability = part["probability"].to_numpy(float)
        prediction = part["prediction"].to_numpy(int)
        fpr, tpr, _ = roc_curve(y, probability)
        pooled_auc = roc_auc_score(y, probability)
        ax.plot(fpr, tpr, lw=1.8, label=f"{model_name} AUC={pooled_auc:.3f}")

        individual_fig, individual_ax = plt.subplots(figsize=(6, 5))
        individual_ax.plot(fpr, tpr, lw=2, label=f"AUC={pooled_auc:.3f}")
        individual_ax.plot([0, 1], [0, 1], "k--", lw=1)
        individual_ax.set_xlabel("False Positive Rate")
        individual_ax.set_ylabel("True Positive Rate")
        individual_ax.set_title(f"{model_name}: pooled ROC over 100 repeats")
        individual_ax.grid(alpha=0.25)
        individual_ax.legend(loc="lower right")
        individual_fig.tight_layout()
        individual_fig.savefig(
            ROC_DIR / f"roc_{model_name}_pooled_100rep.png", dpi=180
        )
        plt.close(individual_fig)

        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        confusion_rows.append(
            {"model": model_name, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        )
        matrix = np.asarray([[tn, fp], [fn, tp]])
        cm_fig, cm_ax = plt.subplots(figsize=(4.5, 4))
        cm_ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                cm_ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
        cm_ax.set_xticks([0, 1], ["Pred normal", "Pred impaired"])
        cm_ax.set_yticks([0, 1], ["True normal", "True impaired"])
        cm_ax.set_title(f"{model_name}: pooled confusion matrix")
        cm_fig.tight_layout()
        cm_fig.savefig(
            CM_DIR / f"confusion_matrix_{model_name}_pooled_100rep.png", dpi=180
        )
        plt.close(cm_fig)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("60s free-walk: pooled ROC over 100 subject-level repeats")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_DIR / "roc_all_models_pooled_100rep.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(confusion_rows).to_csv(
        OUT_DIR / "confusion_matrix_pooled_100rep.csv",
        index=False,
        encoding="utf-8-sig",
    )


def fit_candidate_models(
    assignments: pd.DataFrame,
    x_agg_cache: np.ndarray,
    summary: pd.DataFrame,
) -> None:
    CANDIDATE_DIR.mkdir(exist_ok=True)
    indices = assignments["cache_index"].to_numpy(int)
    x = x_agg_cache[indices]
    y = assignments["target"].to_numpy(int)
    for model_name in ML_MODELS:
        if model_name not in set(summary["model"]):
            continue
        model = make_model(model_name, BASE_SEED)
        model.fit(x, y)
        threshold = float(
            summary.loc[summary["model"].eq(model_name), "threshold_median"].iloc[0]
        )
        artifact = {
            "pipeline": model,
            "features": FEATURES,
            "threshold": threshold,
            "model_name": model_name,
            "status": "experimental_60s_candidate_not_deployed",
            "protocol": {
                "measurement_seconds": 60,
                "window_seconds": 10,
                "window_step_seconds": 2,
                "n_windows": N_WINDOWS,
                "aggregation": {
                    "v_jerk_rms": ["median", "iqr"],
                    "v_harmonic_ratio": ["iqr"],
                },
            },
        }
        joblib.dump(
            artifact, CANDIDATE_DIR / f"gait_60s_top3_{model_name}_candidate.joblib"
        )


def finalize(
    assignments: pd.DataFrame,
    x_agg_cache: np.ndarray,
) -> None:
    metrics, predictions = collect_results()
    if metrics.empty:
        return
    summary = summarize(metrics)
    metrics.to_csv(OUT_DIR / "metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        OUT_DIR / "predictions_by_repeat.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    save_plots(predictions)
    fit_candidate_models(assignments, x_agg_cache, summary)

    metadata = {
        "status": "experimental_only_existing_deployed_model_unchanged",
        "features": FEATURES,
        "sequence_features_for_dl": SEQUENCE_FEATURES,
        "subjects": int(assignments["subject_id"].nunique()),
        "normal_subjects": int(
            (
                assignments.drop_duplicates("subject_id")["target"]
                == 0
            ).sum()
        ),
        "impaired_subjects": int(
            (
                assignments.drop_duplicates("subject_id")["target"]
                == 1
            ).sum()
        ),
        "repeats": N_REPEATS,
        "outer_cv": "subject-level StratifiedKFold(5), one 60s segment per subject per repeat",
        "threshold": "inner 3-fold OOF train probabilities; sensitivity >= 0.80 then maximum specificity",
        "preprocessing": "fit within each training fold only",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCurrent completed summary")
    print(
        summary[
            [
                "model",
                "test_auc_mean",
                "test_auc_std",
                "auc_ci_lo",
                "auc_ci_hi",
                "sensitivity_mean",
                "specificity_mean",
                "recall_mean",
                "precision_mean",
                "accuracy_mean",
                "f1_mean",
                "gap",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build the 60s feature cache and stop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in [OUT_DIR, MODEL_RESULT_DIR, ROC_DIR, CM_DIR, CANDIDATE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    assignments, x_agg_cache, x_seq_cache = build_or_load_cache()
    print(
        f"assignments={len(assignments)} subjects={assignments['subject_id'].nunique()} "
        f"unique_segments={len(x_agg_cache)} device={DEVICE}",
        flush=True,
    )
    if args.cache_only:
        return

    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    for model_name in requested:
        if model_name in DL_MODELS and not TORCH_OK:
            print(f"[skip] {model_name}: PyTorch unavailable", flush=True)
            continue
        metric_path = MODEL_RESULT_DIR / f"{model_name}_metrics.csv"
        prediction_path = MODEL_RESULT_DIR / f"{model_name}_predictions.csv"
        if metric_path.exists() and prediction_path.exists():
            print(f"[reuse] {model_name}", flush=True)
            continue
        evaluate_model(
            model_name,
            assignments,
            x_agg_cache,
            x_seq_cache,
        )
        finalize(assignments, x_agg_cache)
    finalize(assignments, x_agg_cache)


if __name__ == "__main__":
    main()
