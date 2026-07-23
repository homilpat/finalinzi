"""One-repeat paired smoke test for two single-measurement ACC representations."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from gait_axis_aligned_core import window_features
from compare_single_20s_segment_100rep import (
    AGG_FEATURES,
    LEGACY_CACHE_PATH,
    add_acc_qc,
    aggregate_sequence,
    parse_hea,
    read_20s,
)


OUT_DIR = ROOT / "analysis_outputs" / "smoke_top4_subwindows_vs_single19s"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SENSITIVITY = 0.80
SEED = 20260724

SUBWINDOW_TOP4 = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "ap_spec_entropy_median",
    "v_stride_freq_hz_median",
]
SINGLE19_TOP4 = [
    "v_jerk_rms",
    "ap_spec_entropy",
    "v_stride_freq_hz",
    "v_harmonic_ratio",
]


def make_lr() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler()),
        ("lr", LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def threshold_for_sensitivity(y: np.ndarray, probability: np.ndarray) -> float:
    values = np.unique(probability[np.isfinite(probability)])
    candidates = np.r_[values.min() - 1e-9, (values[:-1] + values[1:]) / 2, values.max() + 1e-9]
    best_threshold, best_specificity = float(candidates[0]), -1.0
    for threshold in candidates:
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        if sensitivity >= TARGET_SENSITIVITY and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = float(specificity)
    return best_threshold


def inner_oof_threshold(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    probability = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    model = make_lr()
    for train_idx, valid_idx in cv.split(x, y):
        fitted = clone(model).fit(x[train_idx], y[train_idx])
        probability[valid_idx] = fitted.predict_proba(x[valid_idx])[:, 1]
    return threshold_for_sensitivity(y, probability)


def metric_row(name: str, y: np.ndarray, probability: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "representation": name,
        "auc": roc_auc_score(y, probability),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate(name: str, x: np.ndarray, y: np.ndarray, splits: list) -> tuple[dict, pd.DataFrame]:
    probability = np.full(len(y), np.nan)
    prediction = np.zeros(len(y), dtype=int)
    thresholds = np.full(len(y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(splits):
        threshold = inner_oof_threshold(x[train_idx], y[train_idx], SEED + fold)
        model = make_lr().fit(x[train_idx], y[train_idx])
        probability[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        prediction[test_idx] = probability[test_idx] >= threshold
        thresholds[test_idx] = threshold
    row = metric_row(name, y, probability, prediction)
    row["threshold_median"] = float(np.median(thresholds))
    pred = pd.DataFrame({
        "representation": name,
        "target": y,
        "probability": probability,
        "threshold": thresholds,
        "prediction": prediction,
    })
    return row, pred


def main() -> None:
    cached = np.load(LEGACY_CACHE_PATH, allow_pickle=True)
    meta = pd.DataFrame(cached["meta"].tolist())
    x_seq = cached["x_seq"].astype(np.float32)
    x_agg = aggregate_sequence(x_seq)
    meta = add_acc_qc(meta, x_agg)

    eligible = meta.index[meta["acc_qc_pass"]].to_numpy()
    eligible_meta = meta.loc[eligible]
    rng = np.random.default_rng(SEED)
    chosen = np.array([
        rng.choice(part.index.to_numpy())
        for _, part in eligible_meta.groupby("subject_id", sort=True)
    ])
    chosen_meta = meta.loc[chosen].reset_index(drop=True)

    agg_idx = [AGG_FEATURES.index(name) for name in SUBWINDOW_TOP4]
    x_subwindow = x_agg[chosen][:, agg_idx]

    single_rows = []
    for _, row in chosen_meta.iterrows():
        header = parse_hea(str(row["subject_id"]))
        segment = read_20s(str(row["subject_id"]), float(row["start_sec"]), header)
        feature = window_features(segment[100:])
        single_rows.append([float(feature[name]) for name in SINGLE19_TOP4])
    x_single19 = np.asarray(single_rows, dtype=np.float32)

    y = chosen_meta["target"].to_numpy(int)
    splits = list(StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    ).split(x_subwindow, y))

    rows, predictions = [], []
    for name, x in [
        ("20s_10s_subwindows_top3", x_subwindow[:, :3]),
        ("20s_10s_subwindows_top4", x_subwindow),
        ("trim1s_single19s_top4", x_single19),
    ]:
        row, pred = evaluate(name, x, y, splits)
        rows.append(row)
        pred["subject_id"] = chosen_meta["subject_id"]
        pred["segment_id"] = chosen_meta["segment_id"]
        predictions.append(pred)

    metrics = pd.DataFrame(rows)
    pred_table = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(OUT_DIR / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(OUT_DIR / "smoke_predictions.csv", index=False, encoding="utf-8-sig")
    chosen_meta.to_csv(OUT_DIR / "chosen_segments.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, part in pred_table.groupby("representation"):
        fpr, tpr, _ = roc_curve(part["target"], part["probability"])
        auc = roc_auc_score(part["target"], part["probability"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Paired one-repeat smoke test")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_roc.png", dpi=180)
    plt.close(fig)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Written: {OUT_DIR}")


if __name__ == "__main__":
    main()
