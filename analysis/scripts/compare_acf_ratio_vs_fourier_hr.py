"""Compare the deployed ACF symmetry ratio with classical Fourier HR.

Both features are extracted from the same vertical acceleration, 10-second
windows, and 20-second source segments. The deployed model is not modified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "modeling"))

from gait_axis_aligned_core import acf, bandpass, peak_in_range


FS = 100
WINDOW = 10 * FS
STEP = 2 * FS
MAX_SEGMENTS = 100
OUT_DIR = ROOT / "analysis_outputs" / "acf_ratio_vs_fourier_hr"
SOURCE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "daily_subwindow_median_iqr"
    / "subwindow_median_iqr_table.csv"
)
LABEL_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
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


def parse_header(subject_id: str) -> dict:
    lines = (RAW_DIR / f"{subject_id}.hea").read_text(
        encoding="utf-8"
    ).splitlines()
    parts = lines[0].split()
    gains, baselines = [], []
    for line in lines[1 : 1 + int(parts[1])]:
        match = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2)))
    return {
        "n": int(parts[3]),
        "ch": int(parts[1]),
        "gains": np.asarray(gains[:3]),
        "baselines": np.asarray(baselines[:3]),
    }


def fourier_hr(vertical: np.ndarray) -> tuple[float, float]:
    filtered = bandpass(vertical, float(FS))
    correlation = acf(filtered)
    stride_lag, stride_peak, _ = peak_in_range(
        correlation, float(FS), 0.80, 1.70
    )
    if not np.isfinite(stride_lag) or stride_lag <= 0:
        return np.nan, np.nan

    half = stride_lag / 2.0
    _, step_peak, _ = peak_in_range(
        correlation, float(FS), half * 0.6, half * 1.4
    )
    acf_ratio = (
        float(step_peak / stride_peak)
        if np.isfinite(step_peak)
        and np.isfinite(stride_peak)
        and stride_peak > 1e-6
        else np.nan
    )

    signal = filtered - np.nanmean(filtered)
    amplitude = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / FS)
    stride_frequency = 1.0 / stride_lag
    harmonics = np.arange(1, 21, dtype=float)
    harmonic_frequencies = harmonics * stride_frequency
    valid = harmonic_frequencies <= frequencies[-1]
    if valid.sum() < 4:
        return acf_ratio, np.nan
    harmonic_amplitudes = np.interp(
        harmonic_frequencies[valid], frequencies, amplitude
    )
    harmonic_numbers = harmonics[valid].astype(int)
    even_sum = harmonic_amplitudes[harmonic_numbers % 2 == 0].sum()
    odd_sum = harmonic_amplitudes[harmonic_numbers % 2 == 1].sum()
    fft_hr = float(even_sum / odd_sum) if odd_sum > 1e-12 else np.nan
    return acf_ratio, fft_hr


def summarize(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return np.nan, np.nan
    return (
        float(np.median(finite)),
        float(np.percentile(finite, 75) - np.percentile(finite, 25)),
    )


def extract_segments() -> pd.DataFrame:
    source = pd.read_csv(V2_CSV)
    subjects = (
        source[["subject_id", "group"]]
        .drop_duplicates()
        .sort_values("subject_id")
    )
    rows = []
    for subject_number, item in enumerate(
        subjects.itertuples(index=False), start=1
    ):
        subject_id = str(item.subject_id)
        segments = source[source["subject_id"].eq(subject_id)]
        if len(segments) > MAX_SEGMENTS:
            segments = segments.sample(MAX_SEGMENTS, random_state=42)
        header = parse_header(subject_id)
        raw = np.memmap(
            RAW_DIR / f"{subject_id}.dat",
            dtype="<i2",
            mode="r",
            shape=(header["n"], header["ch"]),
        )
        for segment_order, segment in enumerate(
            segments.itertuples(index=False)
        ):
            start_sec = float(segment.start_sec)
            start = int(round(start_sec * FS))
            acc = raw[start : start + 20 * FS, :3].astype(float)
            if len(acc) < 20 * FS:
                continue
            acc = (acc - header["baselines"]) / header["gains"]
            acf_values, fft_values = [], []
            for offset in range(0, len(acc) - WINDOW + 1, STEP):
                acf_value, fft_value = fourier_hr(
                    acc[offset : offset + WINDOW, 0]
                )
                acf_values.append(acf_value)
                fft_values.append(fft_value)
            acf_median, acf_iqr = summarize(acf_values)
            fft_median, fft_iqr = summarize(fft_values)
            rows.append(
                {
                    "subject_id": subject_id,
                    "segment_order": segment_order,
                    "start_sec": start_sec,
                    "acf_ratio_median": acf_median,
                    "acf_ratio_iqr": acf_iqr,
                    "fourier_hr_median": fft_median,
                    "fourier_hr_iqr": fft_iqr,
                    "n_windows": len(acf_values),
                }
            )
        if subject_number % 10 == 0:
            print(
                f"extracted {subject_number}/{len(subjects)} subjects",
                flush=True,
            )
    return pd.DataFrame(rows)


def repeated_lr_auc(
    table: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> pd.DataFrame:
    rows = []
    y = table["target"].to_numpy(int)
    for repeat in range(100):
        splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=810000 + repeat
        )
        for name, features in feature_sets.items():
            probability = np.full(len(table), np.nan)
            for train_index, test_index in splitter.split(table, y):
                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=1.0, max_iter=1000, random_state=0
                            ),
                        ),
                    ]
                )
                model.fit(table.iloc[train_index][features], y[train_index])
                probability[test_index] = model.predict_proba(
                    table.iloc[test_index][features]
                )[:, 1]
            rows.append(
                {
                    "repeat": repeat,
                    "feature_set": name,
                    "auc": roc_auc_score(y, probability),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "segment_feature_comparison.csv"
    if cache_path.exists():
        segments = pd.read_csv(cache_path)
    else:
        segments = extract_segments()
        original = pd.read_csv(SOURCE_TABLE)
        if len(original) != len(segments):
            raise RuntimeError(
                f"Row mismatch: original={len(original)}, new={len(segments)}"
            )
        segments["v_jerk_rms_median"] = original[
            "v_jerk_rms_median"
        ].to_numpy()
        segments["v_jerk_rms_iqr"] = original[
            "v_jerk_rms_iqr"
        ].to_numpy()
        segments["stored_acf_ratio_iqr"] = original[
            "v_harmonic_ratio_iqr"
        ].to_numpy()
        segments.to_csv(cache_path, index=False, encoding="utf-8-sig")

    labels = pd.read_csv(LABEL_TABLE)[
        ["subject_id", "target"]
    ].drop_duplicates("subject_id")
    subject = (
        segments.groupby("subject_id", as_index=False)
        .median(numeric_only=True)
        .merge(labels, on="subject_id", how="inner")
    )
    feature_sets = {
        "acf_ratio_iqr": [
            "v_jerk_rms_median",
            "v_jerk_rms_iqr",
            "acf_ratio_iqr",
        ],
        "fourier_hr_iqr": [
            "v_jerk_rms_median",
            "v_jerk_rms_iqr",
            "fourier_hr_iqr",
        ],
        "fourier_hr_median": [
            "v_jerk_rms_median",
            "v_jerk_rms_iqr",
            "fourier_hr_median",
        ],
        "fourier_hr_median_iqr": [
            "v_jerk_rms_median",
            "v_jerk_rms_iqr",
            "fourier_hr_median",
            "fourier_hr_iqr",
        ],
    }
    auc = repeated_lr_auc(subject, feature_sets)
    auc.to_csv(
        OUT_DIR / "lr_auc_by_repeat.csv", index=False, encoding="utf-8-sig"
    )
    auc_summary = (
        auc.groupby("feature_set")["auc"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    auc_summary.to_csv(
        OUT_DIR / "lr_auc_summary.csv", index=False, encoding="utf-8-sig"
    )

    finite = segments[
        ["acf_ratio_iqr", "fourier_hr_iqr"]
    ].dropna()
    segment_rho, segment_p = spearmanr(
        finite["acf_ratio_iqr"], finite["fourier_hr_iqr"]
    )
    subject_rho, subject_p = spearmanr(
        subject["acf_ratio_iqr"], subject["fourier_hr_iqr"]
    )
    acf_auc = auc[auc["feature_set"].eq("acf_ratio_iqr")].sort_values(
        "repeat"
    )["auc"]
    fft_auc = auc[auc["feature_set"].eq("fourier_hr_iqr")].sort_values(
        "repeat"
    )["auc"]
    paired = wilcoxon(fft_auc.to_numpy(), acf_auc.to_numpy())
    report = {
        "segments": len(segments),
        "subjects": len(subject),
        "windows_per_segment": 6,
        "stored_vs_recomputed_acf_iqr_max_abs_error": float(
            np.nanmax(
                np.abs(
                    segments["stored_acf_ratio_iqr"]
                    - segments["acf_ratio_iqr"]
                )
            )
        ),
        "segment_iqr_spearman_rho": float(segment_rho),
        "segment_iqr_spearman_p": float(segment_p),
        "subject_iqr_spearman_rho": float(subject_rho),
        "subject_iqr_spearman_p": float(subject_p),
        "acf_ratio_iqr_median": float(
            segments["acf_ratio_iqr"].median()
        ),
        "fourier_hr_iqr_median": float(
            segments["fourier_hr_iqr"].median()
        ),
        "acf_ratio_median_median": float(
            segments["acf_ratio_median"].median()
        ),
        "fourier_hr_median_median": float(
            segments["fourier_hr_median"].median()
        ),
        "fourier_minus_acf_auc_wilcoxon_statistic": float(
            paired.statistic
        ),
        "fourier_minus_acf_auc_wilcoxon_p": float(paired.pvalue),
        "deployed_model_modified": False,
    }
    (OUT_DIR / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    subject.to_csv(
        OUT_DIR / "subject_feature_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(auc_summary.to_string(index=False))


if __name__ == "__main__":
    main()
