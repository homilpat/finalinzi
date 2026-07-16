from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, sosfiltfilt


REQUIRED_COLUMNS = {
    "Timestamp_ns",
    "Acc_X",
    "Acc_Y",
    "Gyro_Clean_X",
}
GRAVITY_MPS2 = 9.80665
TARGET_FS_HZ = 100.0
TRIM_EDGE_SEC = 3.0


def _find_header_line(text: str) -> int:
    for idx, line in enumerate(text.splitlines()):
        if line.startswith("Timestamp_ns"):
            return idx
    raise ValueError("CSV header row starting with Timestamp_ns was not found.")


def load_gait_csv(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig")
        else:
            text = raw
    else:
        text = Path(source).read_text(encoding="utf-8-sig")

    header_line = _find_header_line(text)
    df = pd.read_csv(StringIO("\n".join(text.splitlines()[header_line:])))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).sort_values("Timestamp_ns").reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Not enough gait samples.")
    return df


def _sampling_rate(timestamp_ns: pd.Series) -> float:
    dt = np.diff(timestamp_ns.to_numpy(dtype=float)) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Timestamp interval could not be calculated.")
    fs = 1.0 / float(np.median(dt))
    if not np.isfinite(fs) or fs < 10:
        raise ValueError(f"Sampling rate is too low or invalid: {fs:.2f} Hz")
    return fs


def _resample_to_uniform_hz(df: pd.DataFrame, target_fs: float = TARGET_FS_HZ) -> tuple[pd.DataFrame, float, float]:
    start_ns = float(df["Timestamp_ns"].iloc[0])
    elapsed = (df["Timestamp_ns"].to_numpy(dtype=float) - start_ns) / 1e9
    duration_sec = float(elapsed[-1])
    if duration_sec <= 0:
        raise ValueError("CSV duration is invalid.")

    step = 1.0 / target_fs
    uniform_t = np.arange(0.0, duration_sec + (step * 0.5), step)
    resampled = pd.DataFrame({"_elapsed_sec": uniform_t})
    resampled["Timestamp_ns"] = start_ns + (uniform_t * 1e9)

    numeric_cols = [
        col
        for col in df.columns
        if col != "Timestamp_ns" and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in numeric_cols:
        values = df[col].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(elapsed)
        if valid.sum() < 2:
            resampled[col] = np.nan
        else:
            resampled[col] = np.interp(uniform_t, elapsed[valid], values[valid])
    return resampled, duration_sec, target_fs


def _bandpass(values: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    low = max(low, 0.01)
    if high <= low:
        return values - np.nanmedian(values)
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    if len(values) < 30:
        return values - np.nanmedian(values)
    return sosfiltfilt(sos, values)


def _iqr(values: np.ndarray) -> float:
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def _stride_regularity(signal: np.ndarray, fs: float) -> float | None:
    centered = signal - np.nanmean(signal)
    denom = float(np.sum(centered * centered))
    if not np.isfinite(denom) or denom <= 0:
        return None

    acf = correlate(centered, centered, mode="full")[len(centered) - 1 :]
    acf = acf / denom
    min_lag = max(1, int(round(0.80 * fs)))
    max_lag = min(len(acf) - 1, int(round(1.60 * fs)))
    if max_lag <= min_lag:
        return None
    value = float(np.nanmax(acf[min_lag : max_lag + 1]))
    if not np.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def _extract_window_features(window: pd.DataFrame, fs: float) -> dict[str, float | None]:
    # Smartphone is worn vertically on the waist with the top toward the head.
    # In that placement, Android Y is the closest vertical axis and X is the
    # closest medio-lateral axis. Acceleration is converted from m/s^2 to g.
    v = _bandpass(window["Acc_Y"].to_numpy(dtype=float) / GRAVITY_MPS2, fs, low=0.6, high=3.0)
    ml = _bandpass(window["Acc_X"].to_numpy(dtype=float) / GRAVITY_MPS2, fs, low=0.6, high=3.0)
    # Sensor CSV stores gyroscope values in rad/s; the trained gait pipeline
    # expects the roll amplitude feature on the deg/s scale used in validation.
    roll_raw = np.rad2deg(window["Gyro_Clean_X"].to_numpy(dtype=float))
    roll = _bandpass(roll_raw - np.nanmedian(roll_raw), fs, low=0.5, high=5.0)
    return {
        "v_amp_pool_median": float(np.nanmedian(np.abs(v))),
        "ml_amp_pool_iqr": _iqr(np.abs(ml)),
        "base_v_stride_regularity": _stride_regularity(v, fs),
        "roll_amp_pool_iqr": _iqr(roll),
    }


def extract_gait_features_from_csv(
    source,
    selected_window_sec: float = 10.0,
    trim_edge_sec: float = TRIM_EDGE_SEC,
) -> dict:
    df = load_gait_csv(source)
    observed_fs = _sampling_rate(df["Timestamp_ns"])
    df, duration_sec, fs = _resample_to_uniform_hz(df)

    if duration_sec < selected_window_sec:
        raise ValueError(f"CSV is shorter than {selected_window_sec:.0f} seconds.")

    step_sec = 1.0
    analysis_start_sec = trim_edge_sec
    analysis_end_sec = duration_sec - trim_edge_sec
    if analysis_end_sec - analysis_start_sec < selected_window_sec:
        analysis_start_sec = 0.0
        analysis_end_sec = duration_sec

    candidates = []
    max_start = max(analysis_start_sec, analysis_end_sec - selected_window_sec)
    for start in np.arange(analysis_start_sec, max_start + 0.001, step_sec):
        end = start + selected_window_sec
        window = df[(df["_elapsed_sec"] >= start) & (df["_elapsed_sec"] < end)]
        if len(window) < int(fs * selected_window_sec * 0.75):
            continue
        features = _extract_window_features(window, fs)
        regularity = features.get("base_v_stride_regularity")
        score = regularity if regularity is not None else -1.0
        candidates.append({
            "score": score,
            "start_offset_sec": float(start),
            "end_offset_sec": float(end),
            "sample_count": int(len(window)),
            "features": features,
        })

    if not candidates:
        raise ValueError("No valid 10-second gait window found.")

    feature_names = sorted({key for item in candidates for key in item["features"]})
    median_features = {}
    for feature in feature_names:
        values = [
            item["features"].get(feature)
            for item in candidates
            if item["features"].get(feature) is not None and np.isfinite(item["features"].get(feature))
        ]
        median_features[feature] = float(np.nanmedian(values)) if values else None

    best = max(candidates, key=lambda item: item["score"])
    regularity_scores = [item["score"] for item in candidates if item["score"] >= 0]
    return {
        "features": median_features,
        "window": {
            "protocol": "csv_20s_multi10_median",
            "aggregation": "median_features",
            "trim_edge_sec": trim_edge_sec,
            "analysis_start_sec": round(float(analysis_start_sec), 3),
            "analysis_end_sec": round(float(analysis_end_sec), 3),
            "window_count": len(candidates),
            "window_step_sec": step_sec,
            "best_start_offset_sec": round(best["start_offset_sec"], 3),
            "best_end_offset_sec": round(best["end_offset_sec"], 3),
            "median_sample_count": int(np.nanmedian([item["sample_count"] for item in candidates])),
            "quality_score": None if best["score"] < 0 else round(float(best["score"]), 4),
            "median_quality_score": (
                None if not regularity_scores else round(float(np.nanmedian(regularity_scores)), 4)
            ),
            "sampling_rate_hz": round(float(fs), 3),
            "observed_sampling_rate_hz": round(float(observed_fs), 3),
            "resampled_to_hz": round(float(TARGET_FS_HZ), 3),
            "collected_sec": round(duration_sec, 3),
            "selected_sec": selected_window_sec,
        },
    }
