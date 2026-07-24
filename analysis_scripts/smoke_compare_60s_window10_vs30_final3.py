"""Paired smoke test of 10s vs 30s windows inside one 60s measurement."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from compare_60s_freewalk_models_100rep import (
    FS,
    RAW_DIR,
    TARGET_SENSITIVITY,
    inner_oof_ml,
    make_model,
    metrics_from_predictions,
    parse_hea,
    threshold_for_min_sensitivity,
)
from gait_axis_aligned_core import acf, bandpass, peak_in_range


SOURCE_DIR = ROOT / "analysis_outputs" / "freewalk_60s_model_comparison_100rep"
ASSIGNMENT_PATH = SOURCE_DIR / "repeat_segment_assignments.csv"
OUT_DIR = ROOT / "analysis_outputs" / "smoke_60s_window10_vs30_final3"
REPEATS = 10
STEP_SECONDS = 2
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]


def vertical_window_features(vertical: np.ndarray) -> tuple[float, float]:
    filtered = bandpass(vertical, float(FS))
    correlation = acf(filtered)
    stride_lag, stride_peak, _ = peak_in_range(
        correlation, float(FS), 0.80, 1.70
    )
    harmonic_ratio = np.nan
    if (
        np.isfinite(stride_lag)
        and stride_lag > 0
        and np.isfinite(stride_peak)
        and stride_peak > 1e-6
    ):
        half = stride_lag / 2.0
        _, step_peak, _ = peak_in_range(
            correlation, float(FS), half * 0.6, half * 1.4
        )
        if np.isfinite(step_peak):
            harmonic_ratio = float(step_peak / stride_peak)
    jerk_rms = float(np.sqrt(np.nanmean(np.diff(filtered) ** 2)) * FS)
    return jerk_rms, harmonic_ratio


def aggregate_60s(acc: np.ndarray, window_seconds: int) -> np.ndarray:
    window = window_seconds * FS
    step = STEP_SECONDS * FS
    rows = []
    for start in range(0, len(acc) - window + 1, step):
        rows.append(vertical_window_features(acc[start : start + window, 0]))
    values = np.asarray(rows, dtype=float)
    jerk = values[:, 0]
    harmonic_ratio = values[:, 1]
    return np.asarray(
        [
            np.nanmedian(jerk),
            np.nanpercentile(jerk, 75) - np.nanpercentile(jerk, 25),
            np.nanpercentile(harmonic_ratio, 75)
            - np.nanpercentile(harmonic_ratio, 25),
        ],
        dtype=np.float32,
    )


def extract_features(assignments: pd.DataFrame) -> pd.DataFrame:
    unique = (
        assignments[
            ["segment_id", "subject_id", "target", "bout_idx", "start_sec"]
        ]
        .drop_duplicates("segment_id")
        .sort_values(["subject_id", "start_sec"])
    )
    rows = []
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
        for _, item in part.iterrows():
            start = int(round(float(item["start_sec"]) * FS))
            acc = raw[start : start + 60 * FS, :3].astype(float)
            acc = (acc - header["baselines"]) / header["gains"]
            row = {
                "segment_id": item["segment_id"],
                "subject_id": subject_id,
                "target": int(item["target"]),
            }
            for window_seconds in (10, 30):
                values = aggregate_60s(acc, window_seconds)
                for feature, value in zip(FEATURES, values):
                    row[f"{feature}__w{window_seconds}"] = float(value)
            rows.append(row)
        print(
            f"[features] {subject_number:02d}/{unique['subject_id'].nunique()} "
            f"{subject_id}: {len(part)}",
            flush=True,
        )
    return pd.DataFrame(rows)


def evaluate(
    assignments: pd.DataFrame, feature_table: pd.DataFrame
) -> pd.DataFrame:
    merged = assignments.merge(
        feature_table.drop(columns=["subject_id", "target"]),
        on="segment_id",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for repeat in range(REPEATS):
        part = (
            merged[merged["repeat"].eq(repeat)]
            .sort_values("subject_id")
            .reset_index(drop=True)
        )
        y = part["target"].to_numpy(int)
        outer = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=810000 + repeat
        )
        splits = list(outer.split(np.zeros((len(y), 1)), y))
        for window_seconds in (10, 30):
            columns = [f"{feature}__w{window_seconds}" for feature in FEATURES]
            x = part[columns].to_numpy(np.float32)
            oof_probability = np.full(len(y), np.nan)
            oof_prediction = np.zeros(len(y), dtype=int)
            thresholds = []
            for fold, (train_idx, test_idx) in enumerate(splits):
                model = make_model("LR", 900000 + repeat * 100 + fold)
                inner_probability = inner_oof_ml(
                    model,
                    x[train_idx],
                    y[train_idx],
                    950000 + repeat * 100 + fold,
                )
                threshold = threshold_for_min_sensitivity(
                    y[train_idx], inner_probability
                )
                model.fit(x[train_idx], y[train_idx])
                probability = model.predict_proba(x[test_idx])[:, 1]
                oof_probability[test_idx] = probability
                oof_prediction[test_idx] = (
                    probability >= threshold
                ).astype(int)
                thresholds.append(threshold)
            rows.append(
                {
                    "window_seconds": window_seconds,
                    "repeat": repeat,
                    "n_windows": (60 - window_seconds) // STEP_SECONDS + 1,
                    "threshold_median": float(np.median(thresholds)),
                    **metrics_from_predictions(
                        y, oof_probability, oof_prediction
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(ASSIGNMENT_PATH)
    assignments = assignments[assignments["repeat"].lt(REPEATS)].copy()
    feature_path = OUT_DIR / "paired_feature_table.csv"
    if feature_path.exists():
        feature_table = pd.read_csv(feature_path)
    else:
        feature_table = extract_features(assignments)
        feature_table.to_csv(feature_path, index=False, encoding="utf-8-sig")

    metrics = evaluate(assignments, feature_table)
    metrics.to_csv(
        OUT_DIR / "paired_metrics_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        metrics.groupby("window_seconds")
        .agg(
            repeats=("repeat", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            threshold_median=("threshold_median", "median"),
        )
        .reset_index()
    )
    pivot = metrics.pivot(
        index="repeat", columns="window_seconds", values="auc"
    )
    delta = pivot[30] - pivot[10]
    comparison = {
        "auc_delta_30s_minus_10s_mean": float(delta.mean()),
        "auc_delta_median": float(delta.median()),
        "auc_delta_min": float(delta.min()),
        "auc_delta_max": float(delta.max()),
        "paired_wilcoxon_p": float(wilcoxon(delta).pvalue),
        "target_sensitivity": TARGET_SENSITIVITY,
    }
    summary.to_csv(
        OUT_DIR / "paired_summary.csv", index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / "paired_comparison.json").write_text(
        pd.Series(comparison).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n", comparison)
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
