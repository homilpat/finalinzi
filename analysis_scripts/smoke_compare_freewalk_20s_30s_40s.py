"""Paired one-repeat LR smoke test using 20s, 30s, and 40s from one 60s daily-walk bout."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from gait_axis_aligned_core import TARGET_FS_HZ, window_features
from compare_single_20s_segment_100rep import LEGACY_CACHE_PATH, RAW_DIR, V2_CSV, parse_hea
from smoke_compare_top4_subwindows_vs_single19s import evaluate


OUT_DIR = ROOT / "analysis_outputs" / "smoke_freewalk_20s_30s_40s"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260724
FS = int(TARGET_FS_HZ)
SUBWINDOW_SAMPLES = 10 * FS
SUBWINDOW_STEP = 2 * FS
DURATIONS = (20, 30, 40)
TOP3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "ap_spec_entropy_median"]


def read_acc(raw: np.memmap, header: dict, start_sec: float, duration_sec: int) -> np.ndarray:
    start = int(round(start_sec * FS))
    end = start + duration_sec * FS
    segment = raw[start:end, :3].astype(float)
    return (segment - header["baselines"]) / header["gains"]


def top3_features(segment: np.ndarray) -> tuple[np.ndarray, int]:
    rows = []
    for start in range(0, len(segment) - SUBWINDOW_SAMPLES + 1, SUBWINDOW_STEP):
        feature = window_features(segment[start : start + SUBWINDOW_SAMPLES])
        rows.append({
            "v_jerk_rms": float(feature["v_jerk_rms"]),
            "ap_spec_entropy": float(feature["ap_spec_entropy"]),
        })
    table = pd.DataFrame(rows)
    jerk = table["v_jerk_rms"].dropna()
    entropy = table["ap_spec_entropy"].dropna()
    values = np.array([
        jerk.median(),
        jerk.quantile(0.75) - jerk.quantile(0.25),
        entropy.median(),
    ], dtype=np.float32)
    return values, len(table)


def main() -> None:
    cached = np.load(LEGACY_CACHE_PATH, allow_pickle=True)
    labels = (
        pd.DataFrame(cached["meta"].tolist())[["subject_id", "target"]]
        .drop_duplicates("subject_id")
    )
    bouts = pd.read_csv(
        RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "strict_bouts_all.csv"
    )
    bout_lookup = bouts.set_index(["subject_id", "bout_idx"])
    valid20 = pd.read_csv(V2_CSV, usecols=["subject_id", "bout_idx", "start_sec"])
    valid20 = valid20.merge(labels, on="subject_id", how="inner")

    rng = np.random.default_rng(SEED)
    chosen_bouts = []
    features = {duration: [] for duration in DURATIONS}
    kept_labels = []
    kept_subjects = []

    for subject_id, part in valid20.groupby("subject_id", sort=True):
        candidates = []
        for bout_idx, bout_part in part.groupby("bout_idx"):
            bout = bout_lookup.loc[(subject_id, bout_idx)]
            bout_start = float(bout["start_sec"])
            bout_end = float(bout["end_sec"])
            starts = set(np.round(bout_part["start_sec"].to_numpy(float), 3))
            for start40 in starts:
                has_contiguous_qc = all(round(start40 + delta, 3) in starts for delta in (0, 5, 10, 15, 20))
                fits_centered_60s = start40 - 10 >= bout_start and start40 + 50 <= bout_end
                if has_contiguous_qc and fits_centered_60s:
                    candidates.append((int(bout_idx), float(start40)))
        if not candidates:
            continue
        bout_idx, start40 = candidates[int(rng.integers(0, len(candidates)))]
        freewalk_start = start40 - 10.0

        header = parse_hea(str(subject_id))
        raw = np.memmap(
            RAW_DIR / f"{subject_id}.dat",
            dtype="<i2",
            mode="r",
            shape=(header["n"], header["ch"]),
        )
        subject_features = {}
        valid = True
        for duration in DURATIONS:
            interval_start = freewalk_start + (60.0 - duration) / 2.0
            try:
                values, count = top3_features(read_acc(raw, header, interval_start, duration))
            except Exception:
                valid = False
                break
            if not np.isfinite(values).all():
                valid = False
                break
            subject_features[duration] = (values, count, interval_start)
        if not valid:
            continue

        kept_subjects.append(subject_id)
        target = int(part["target"].iloc[0])
        kept_labels.append(target)
        for duration in DURATIONS:
            values, count, interval_start = subject_features[duration]
            features[duration].append(values)
            chosen_bouts.append({
                "subject_id": subject_id,
                "target": target,
                "bout_idx": bout_idx,
                "freewalk_start_sec": freewalk_start,
                "duration_sec": duration,
                "interval_start_sec": interval_start,
                "interval_end_sec": interval_start + duration,
                "n_10s_subwindows": count,
            })

    y = np.asarray(kept_labels, dtype=int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        np.zeros((len(y), 1)), y
    ))

    rows, predictions = [], []
    for duration in DURATIONS:
        name = f"freewalk_{duration}s_top3"
        x = np.asarray(features[duration], dtype=np.float32)
        row, pred = evaluate(name, x, y, splits)
        row["n_subjects"] = len(y)
        row["n_subwindows"] = (duration - 10) // 2 + 1
        rows.append(row)
        pred["subject_id"] = kept_subjects
        predictions.append(pred)

    metrics = pd.DataFrame(rows)
    pred_table = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(OUT_DIR / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(OUT_DIR / "smoke_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chosen_bouts).to_csv(OUT_DIR / "chosen_bouts.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, part in pred_table.groupby("representation"):
        fpr, tpr, _ = roc_curve(part["target"], part["probability"])
        auc = roc_auc_score(part["target"], part["probability"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Paired 60s daily-walk smoke test")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_roc.png", dpi=180)
    plt.close(fig)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Written: {OUT_DIR}")


if __name__ == "__main__":
    main()
