"""Paired 20s smoke test: ACC top3 versus literature-style gyro gait features."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from gait_axis_aligned_core import acf, peak_in_range, spec_entropy, window_features
from audit_gyro_turn_freewalk40 import event_mask, read_interval
from smoke_compare_freewalk_20s_30s_40s import FS, SEED, evaluate


SOURCE = ROOT / "analysis_outputs" / "smoke_top4_subwindows_vs_single19s" / "chosen_segments.csv"
OUT_DIR = ROOT / "analysis_outputs" / "smoke_20s_acc_vs_pitch_roll_gyro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN = 10 * FS
STEP = 2 * FS


def gyro_window_features(gyro_window: np.ndarray) -> dict:
    pitch = detrend(np.asarray(gyro_window[:, 1], dtype=float), type="linear")
    roll = detrend(np.asarray(gyro_window[:, 2], dtype=float), type="linear")
    sos = butter(4, [0.5 / (0.5 * FS), 5.0 / (0.5 * FS)], btype="bandpass", output="sos")
    pitch_band = sosfiltfilt(sos, pitch)
    roll_band = sosfiltfilt(sos, roll)

    regularity = []
    entropy = []
    for signal in (pitch_band, roll_band):
        _, stride_peak, _ = peak_in_range(acf(signal), FS, 0.80, 1.70)
        _, signal_entropy = spec_entropy(signal, FS)
        regularity.append(stride_peak)
        entropy.append(signal_entropy)

    return {
        "pitch_band_rms": float(np.sqrt(np.mean(np.square(pitch_band)))),
        "roll_band_rms": float(np.sqrt(np.mean(np.square(roll_band)))),
        "gyro_pr_stride_regularity": float(np.nanmean(regularity)),
        "gyro_pr_spec_entropy": float(np.nanmean(entropy)),
        "gyro_pr_band_rms": float(np.sqrt(np.mean(pitch_band**2 + roll_band**2))),
    }


def median_iqr(table: pd.DataFrame, column: str) -> tuple[float, float]:
    values = table[column].dropna()
    return (
        float(values.median()),
        float(values.quantile(0.75) - values.quantile(0.25)),
    )


def extract_features(acc: np.ndarray, gyro: np.ndarray) -> dict:
    rows = []
    for start in range(0, len(acc) - WIN + 1, STEP):
        acc_feature = window_features(acc[start : start + WIN])
        gyro_window = gyro[start : start + WIN]
        rows.append({
            "v_jerk_rms": float(acc_feature["v_jerk_rms"]),
            "ap_spec_entropy": float(acc_feature["ap_spec_entropy"]),
            **gyro_window_features(gyro_window),
        })
    table = pd.DataFrame(rows)
    jerk_median, jerk_iqr = median_iqr(table, "v_jerk_rms")
    gyro_reg_median, gyro_reg_iqr = median_iqr(table, "gyro_pr_stride_regularity")
    gyro_entropy_median, _ = median_iqr(table, "gyro_pr_spec_entropy")
    _, gyro_rms_iqr = median_iqr(table, "gyro_pr_band_rms")
    yaw = gyro[:, 0] - np.median(gyro[:, 0])
    _, turn_count = event_mask(yaw)
    return {
        "v_jerk_rms_median": jerk_median,
        "v_jerk_rms_iqr": jerk_iqr,
        "ap_spec_entropy_median": float(table["ap_spec_entropy"].median()),
        "pitch_band_rms_median": float(table["pitch_band_rms"].median()),
        "roll_band_rms_median": float(table["roll_band_rms"].median()),
        "gyro_pr_stride_regularity_median": gyro_reg_median,
        "gyro_pr_stride_regularity_iqr": gyro_reg_iqr,
        "gyro_pr_spec_entropy_median": gyro_entropy_median,
        "gyro_pr_band_rms_iqr": gyro_rms_iqr,
        "yaw_turn_event_count_qc": int(turn_count),
    }


def main() -> None:
    chosen = pd.read_csv(SOURCE)
    rows = []
    for _, item in chosen.iterrows():
        subject_id = str(item["subject_id"])
        acc, gyro = read_interval(subject_id, float(item["start_sec"]), duration_sec=20.0)
        rows.append({
            "subject_id": subject_id,
            "target": int(item["target"]),
            "segment_id": item["segment_id"],
            **extract_features(acc, gyro),
        })
    feature_table = pd.DataFrame(rows)

    acc_cols = [
        "v_jerk_rms_median",
        "v_jerk_rms_iqr",
        "ap_spec_entropy_median",
    ]
    configurations = {
        "acc_top3": acc_cols,
        "acc_top3_plus_pitch": [*acc_cols, "pitch_band_rms_median"],
        "acc_top3_plus_pitch_roll": [
            *acc_cols,
            "pitch_band_rms_median",
            "roll_band_rms_median",
        ],
        "gyro_literature4": [
            "gyro_pr_stride_regularity_median",
            "gyro_pr_stride_regularity_iqr",
            "gyro_pr_spec_entropy_median",
            "gyro_pr_band_rms_iqr",
        ],
        "acc_top3_plus_gyro_literature4": [
            *acc_cols,
            "gyro_pr_stride_regularity_median",
            "gyro_pr_stride_regularity_iqr",
            "gyro_pr_spec_entropy_median",
            "gyro_pr_band_rms_iqr",
        ],
    }

    y = feature_table["target"].to_numpy(int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        np.zeros((len(y), 1)), y
    ))
    metrics, predictions = [], []
    for name, columns in configurations.items():
        x = feature_table[columns].to_numpy(float)
        metric, pred = evaluate(name, x, y, splits)
        metric["n_features"] = len(columns)
        metric["features"] = "|".join(columns)
        metrics.append(metric)
        pred["subject_id"] = feature_table["subject_id"]
        predictions.append(pred)

    metrics_table = pd.DataFrame(metrics)
    pred_table = pd.concat(predictions, ignore_index=True)
    feature_table.to_csv(OUT_DIR / "feature_table.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(OUT_DIR / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(OUT_DIR / "smoke_predictions.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, part in pred_table.groupby("representation"):
        fpr, tpr, _ = roc_curve(part["target"], part["probability"])
        auc = roc_auc_score(part["target"], part["probability"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Paired 20s ACC vs literature-style gyro features")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_roc.png", dpi=180)
    plt.close(fig)

    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
