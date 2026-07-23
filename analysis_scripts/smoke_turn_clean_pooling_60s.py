"""Paired smoke test for gyro turn-clean pooling over a 60s daily-walk interval."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from gait_axis_aligned_core import window_features
from audit_gyro_turn_freewalk40 import event_mask, read_interval
from smoke_compare_freewalk_20s_30s_40s import FS, SEED, evaluate


SOURCE_DIR = ROOT / "analysis_outputs" / "smoke_freewalk_20s_30s_40s"
OUT_DIR = ROOT / "analysis_outputs" / "smoke_turn_clean_pooling_60s"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN = 10 * FS
STEP = 2 * FS
TURN_BUFFER = int(round(0.5 * FS))
MIN_CLEAN_WINDOWS = 8


def feature_windows(acc: np.ndarray) -> pd.DataFrame:
    rows = []
    for start in range(0, len(acc) - WIN + 1, STEP):
        feature = window_features(acc[start : start + WIN])
        rows.append({
            "start_sample": start,
            "v_jerk_rms": float(feature["v_jerk_rms"]),
            "ap_spec_entropy": float(feature["ap_spec_entropy"]),
        })
    return pd.DataFrame(rows)


def aggregate_top3(windows: pd.DataFrame) -> np.ndarray:
    jerk = windows["v_jerk_rms"].dropna()
    entropy = windows["ap_spec_entropy"].dropna()
    return np.asarray([
        jerk.median(),
        jerk.quantile(0.75) - jerk.quantile(0.25),
        entropy.median(),
    ], dtype=np.float32)


def main() -> None:
    intervals = pd.read_csv(SOURCE_DIR / "chosen_bouts.csv")
    intervals = intervals[intervals["duration_sec"].eq(40)].copy()

    rows = []
    feature_sets = {
        "central40_all16_top3": [],
        "freewalk60_all26_top3": [],
        "freewalk60_turnclean_top3": [],
    }
    labels, subjects = [], []

    for _, interval in intervals.iterrows():
        subject_id = str(interval["subject_id"])
        acc, gyro = read_interval(subject_id, float(interval["freewalk_start_sec"]), duration_sec=60.0)
        windows60 = feature_windows(acc)

        yaw = gyro[:, 0] - np.median(gyro[:, 0])
        turn_mask, turn_count = event_mask(yaw)
        buffered_turn = binary_dilation(turn_mask, iterations=TURN_BUFFER)
        affected = []
        for start in windows60["start_sample"].to_numpy(int):
            affected.append(bool(buffered_turn[start : start + WIN].any()))
        windows60["turn_affected"] = affected
        clean = windows60.loc[~windows60["turn_affected"]].copy()

        central40 = windows60[
            windows60["start_sample"].between(10 * FS, 40 * FS, inclusive="both")
        ].copy()
        included = len(central40) == 16 and len(clean) >= MIN_CLEAN_WINDOWS
        rows.append({
            "subject_id": subject_id,
            "target": int(interval["target"]),
            "turn_event_count": turn_count,
            "all_window_count": len(windows60),
            "turn_affected_window_count": int(windows60["turn_affected"].sum()),
            "clean_window_count": len(clean),
            "clean_window_rate": len(clean) / len(windows60),
            "included_in_paired_test": included,
        })
        if not included:
            continue

        feature_sets["central40_all16_top3"].append(aggregate_top3(central40))
        feature_sets["freewalk60_all26_top3"].append(aggregate_top3(windows60))
        feature_sets["freewalk60_turnclean_top3"].append(aggregate_top3(clean))
        labels.append(int(interval["target"]))
        subjects.append(subject_id)

    y = np.asarray(labels, dtype=int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        np.zeros((len(y), 1)), y
    ))

    metrics, predictions = [], []
    for name, values in feature_sets.items():
        x = np.asarray(values, dtype=np.float32)
        metric, pred = evaluate(name, x, y, splits)
        metric["n_subjects"] = len(y)
        metrics.append(metric)
        pred["subject_id"] = subjects
        predictions.append(pred)

    metrics_table = pd.DataFrame(metrics)
    pred_table = pd.concat(predictions, ignore_index=True)
    qc_table = pd.DataFrame(rows)
    metrics_table.to_csv(OUT_DIR / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(OUT_DIR / "smoke_predictions.csv", index=False, encoding="utf-8-sig")
    qc_table.to_csv(OUT_DIR / "clean_window_counts.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, part in pred_table.groupby("representation"):
        fpr, tpr, _ = roc_curve(part["target"], part["probability"])
        auc = roc_auc_score(part["target"], part["probability"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("60s gyro turn-clean pooling smoke test")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_roc.png", dpi=180)
    plt.close(fig)

    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nClean-window availability:")
    print(qc_table[["clean_window_count", "clean_window_rate"]].describe().to_string(
        float_format=lambda value: f"{value:.4f}"
    ))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
