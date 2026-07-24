from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


DEFAULT_FS = 50.0


@dataclass
class ExerciseSeries:
    fs: float
    time_sec: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray | None
    axes: tuple[str, str, str]
    vertical_idx: int
    lateral_idx: int
    ap_idx: int


def _numeric_column(df: pd.DataFrame, names: list[str]) -> np.ndarray | None:
    lowered = {c.lower(): c for c in df.columns}
    for name in names:
        col = lowered.get(name.lower())
        if col is not None:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return None


def _read_sensor_csv(file_obj: str | BinaryIO) -> pd.DataFrame:
    return pd.read_csv(file_obj, comment="#")


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    timestamp_ns = _numeric_column(df, ["Timestamp_ns", "timestamp_ns"])
    if timestamp_ns is not None:
        t = timestamp_ns - np.nanmin(timestamp_ns)
        return t / 1e9

    timestamp_ms = _numeric_column(df, ["timestamp_ms", "timestamp"])
    if timestamp_ms is not None:
        t = timestamp_ms - np.nanmin(timestamp_ms)
        scale = 1000.0 if np.nanmax(t) > 1000 else 1.0
        return t / scale

    return np.arange(len(df), dtype=float) / DEFAULT_FS


def _pick_acc_columns(df: pd.DataFrame) -> tuple[np.ndarray, tuple[str, str, str]]:
    anatomical = ["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]
    if all(c in df.columns for c in anatomical):
        acc = df[anatomical].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return acc, ("V", "ML", "AP")

    raw = ["Acc_X", "Acc_Y", "Acc_Z"]
    if all(c in df.columns for c in raw):
        acc = df[raw].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # Raw APK files are usually m/s2. Convert to g when values look large.
        if np.nanmedian(np.linalg.norm(acc, axis=1)) > 3.0:
            acc = acc / 9.80665
        return acc, ("raw_X", "raw_Y", "raw_Z")

    raise ValueError("missing accelerometer columns")


def _pick_gyro_columns(df: pd.DataFrame) -> np.ndarray | None:
    clean = ["Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z"]
    raw = ["Gyro_Raw_X", "Gyro_Raw_Y", "Gyro_Raw_Z"]
    roll = ["Gyro_Roll_deg_s"]
    if all(c in df.columns for c in clean):
        return df[clean].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if all(c in df.columns for c in raw):
        return df[raw].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if all(c in df.columns for c in roll):
        vals = df[roll].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.column_stack([np.zeros(len(vals)), np.zeros(len(vals)), vals[:, 0]])
    return None


def _resample(t: np.ndarray, values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(t) & np.isfinite(values).all(axis=1)
    t = t[keep]
    values = values[keep]
    if len(t) < 4:
        raise ValueError("not enough valid sensor rows")
    order = np.argsort(t)
    t = t[order]
    values = values[order]
    unique = np.r_[True, np.diff(t) > 1e-6]
    t = t[unique]
    values = values[unique]
    grid = np.arange(0.0, float(t[-1] - t[0]), 1.0 / fs)
    if len(grid) < 10:
        raise ValueError("sensor duration is too short")
    shifted = t - t[0]
    out = np.column_stack([np.interp(grid, shifted, values[:, i]) for i in range(values.shape[1])])
    return grid, out


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    if len(x) < 30:
        return x
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    if high <= low:
        return x
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _lowpass(x: np.ndarray, fs: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 30:
        return x
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    sos = butter(3, high / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def load_exercise_series(file_obj: str | BinaryIO, fs: float = DEFAULT_FS) -> ExerciseSeries:
    df = _read_sensor_csv(file_obj)
    t = _time_seconds(df)
    acc_raw, axes = _pick_acc_columns(df)
    gyro_raw = _pick_gyro_columns(df)
    time_sec, acc = _resample(t, acc_raw, fs)
    gyro = None
    if gyro_raw is not None:
        _, gyro = _resample(t, gyro_raw, fs)

    if axes == ("V", "ML", "AP"):
        vertical_idx, lateral_idx, ap_idx = 0, 1, 2
    else:
        med = np.nanmedian(acc, axis=0)
        vertical_idx = int(np.nanargmax(np.abs(med)))
        remaining = [i for i in range(3) if i != vertical_idx]
        dyn_power = [float(np.nanstd(_bandpass(acc[:, i], fs, 0.4, 3.5))) for i in remaining]
        lateral_idx = remaining[int(np.nanargmax(dyn_power))]
        ap_idx = remaining[1 - int(np.nanargmax(dyn_power))]

    return ExerciseSeries(
        fs=fs,
        time_sec=time_sec,
        acc=acc,
        gyro=gyro,
        axes=axes,
        vertical_idx=vertical_idx,
        lateral_idx=lateral_idx,
        ap_idx=ap_idx,
    )


def _count_reps(signal: np.ndarray, fs: float, min_interval_sec: float, threshold_scale: float = 0.8) -> tuple[int, list[float], float]:
    sig = np.asarray(signal, dtype=float)
    height = max(np.nanmedian(sig) + threshold_scale * np.nanstd(sig), np.nanpercentile(sig, 70))
    distance = max(1, int(round(min_interval_sec * fs)))
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=max(1e-6, 0.35 * np.nanstd(sig)))
    return int(len(peaks)), (peaks / fs).round(2).tolist(), float(height)


def analyze_knee_raise(series: ExerciseSeries) -> dict:
    vertical = _bandpass(series.acc[:, series.vertical_idx], series.fs, 0.5, 4.0)
    rectified = np.abs(vertical)
    count, peaks, threshold = _count_reps(rectified, series.fs, 0.35, 0.7)
    return {"exercise": "standing_knee_raise", "count": count, "rep_times_sec": peaks, "threshold": threshold}


def analyze_jump_stop(series: ExerciseSeries) -> dict:
    vertical = _bandpass(series.acc[:, series.vertical_idx], series.fs, 0.7, 6.0)
    energy = _lowpass(np.abs(vertical), series.fs, 3.0)
    jumps, peaks, threshold = _count_reps(energy, series.fs, 0.45, 1.0)
    tail = energy[int(max(0, len(energy) - 2.0 * series.fs)) :]
    baseline = float(np.nanmedian(energy))
    stopped = bool(len(tail) > 0 and np.nanmedian(tail) < baseline + 0.35 * np.nanstd(energy))
    stop_time = float(series.time_sec[-1]) if stopped else None
    return {
        "exercise": "jump_then_stop",
        "count": jumps,
        "rep_times_sec": peaks,
        "stop_detected": stopped,
        "stop_time_sec": stop_time,
        "threshold": threshold,
    }


def analyze_side_walk(series: ExerciseSeries) -> dict:
    lateral = _bandpass(series.acc[:, series.lateral_idx], series.fs, 0.4, 3.5)
    count, peaks, threshold = _count_reps(np.abs(lateral), series.fs, 0.30, 0.65)
    drift = _lowpass(series.acc[:, series.lateral_idx] - np.nanmedian(series.acc[:, series.lateral_idx]), series.fs, 0.35)
    signed_area = float(np.trapz(drift, dx=1.0 / series.fs))
    direction = "right" if signed_area > 0 else "left"
    return {
        "exercise": "side_walk",
        "count": count,
        "rep_times_sec": peaks,
        "direction": direction,
        "signed_lateral_area": signed_area,
        "threshold": threshold,
    }


def analyze_seated_knee_extension(series: ExerciseSeries) -> dict:
    ap = _bandpass(series.acc[:, series.ap_idx], series.fs, 0.3, 3.0)
    count, peaks, threshold = _count_reps(np.abs(ap), series.fs, 0.45, 0.75)
    tilt_source = series.gyro[:, 2] if series.gyro is not None else series.acc[:, series.ap_idx]
    tilt = _lowpass(tilt_source - np.nanmedian(tilt_source), series.fs, 0.5)
    posterior_tilt_score = float(np.nanpercentile(-tilt, 95))
    posterior_tilt_detected = bool(posterior_tilt_score > max(0.15, np.nanstd(tilt)))
    return {
        "exercise": "seated_knee_extension",
        "count": count,
        "rep_times_sec": peaks,
        "posterior_pelvic_tilt_detected": posterior_tilt_detected,
        "posterior_pelvic_tilt_score": posterior_tilt_score,
        "threshold": threshold,
    }


ANALYZERS = {
    "knee_raise": analyze_knee_raise,
    "standing_knee_raise": analyze_knee_raise,
    "jump_stop": analyze_jump_stop,
    "jump_then_stop": analyze_jump_stop,
    "side_walk": analyze_side_walk,
    "seated_knee_extension": analyze_seated_knee_extension,
    "knee_extension": analyze_seated_knee_extension,
}


def analyze_exercise_csv(file_obj: str | BinaryIO, exercise_type: str, fs: float = DEFAULT_FS) -> dict:
    series = load_exercise_series(file_obj, fs=fs)
    key = (exercise_type or "").strip()
    analyzer = ANALYZERS.get(key)
    if analyzer is None:
        raise ValueError(f"unsupported exercise_type: {exercise_type}")
    result = analyzer(series)
    result.update(
        {
            "duration_sec": float(series.time_sec[-1]) if len(series.time_sec) else 0.0,
            "fs": series.fs,
            "axes": series.axes,
            "vertical_axis": series.axes[series.vertical_idx],
            "lateral_axis": series.axes[series.lateral_idx],
            "ap_axis": series.axes[series.ap_idx],
            "mode": "rule_based_v1_requires_sample_threshold_tuning",
        }
    )
    return result
