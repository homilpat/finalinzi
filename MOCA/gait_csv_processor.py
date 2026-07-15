from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, sosfiltfilt


REQUIRED_COLUMNS = {
    "Timestamp_ns",
    "Acc_X",
    "Acc_Z",
    "Gyro_Clean_Z",
}
GRAVITY_MPS2 = 9.80665


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
    # Sensor CSV stores acceleration in m/s^2; the gait model was trained on
    # acceleration features in g-scale.
    v = _bandpass(window["Acc_Z"].to_numpy(dtype=float) / GRAVITY_MPS2, fs, low=0.6, high=3.0)
    ml = _bandpass(window["Acc_X"].to_numpy(dtype=float) / GRAVITY_MPS2, fs, low=0.6, high=3.0)
    # Sensor CSV stores gyroscope values in rad/s; the trained gait pipeline
    # expects the roll amplitude feature on the deg/s scale used in validation.
    roll_raw = np.rad2deg(window["Gyro_Clean_Z"].to_numpy(dtype=float))
    roll = _bandpass(roll_raw - np.nanmedian(roll_raw), fs, low=0.5, high=5.0)
    return {
        "v_amp_pool_median": float(np.nanmedian(np.abs(v))),
        "ml_amp_pool_iqr": _iqr(np.abs(ml)),
        "base_v_stride_regularity": _stride_regularity(v, fs),
        "roll_amp_pool_iqr": _iqr(roll),
    }


def extract_gait_features_from_csv(source, selected_window_sec: float = 10.0) -> dict:
    df = load_gait_csv(source)
    fs = _sampling_rate(df["Timestamp_ns"])
    start_ns = float(df["Timestamp_ns"].iloc[0])
    df = df.copy()
    df["_elapsed_sec"] = (df["Timestamp_ns"].astype(float) - start_ns) / 1e9
    duration_sec = float(df["_elapsed_sec"].iloc[-1])

    if duration_sec < selected_window_sec:
        raise ValueError(f"CSV is shorter than {selected_window_sec:.0f} seconds.")

    step_sec = 1.0
    best = None
    max_start = max(0.0, duration_sec - selected_window_sec)
    for start in np.arange(0.0, max_start + 0.001, step_sec):
        end = start + selected_window_sec
        window = df[(df["_elapsed_sec"] >= start) & (df["_elapsed_sec"] < end)]
        if len(window) < int(fs * selected_window_sec * 0.75):
            continue
        features = _extract_window_features(window, fs)
        regularity = features.get("base_v_stride_regularity")
        score = regularity if regularity is not None else -1.0
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "start_offset_sec": float(start),
                "end_offset_sec": float(end),
                "sample_count": int(len(window)),
                "features": features,
            }

    if best is None:
        raise ValueError("No valid 10-second gait window found.")

    return {
        "features": best["features"],
        "window": {
            "protocol": "csv_20s_best10",
            "start_offset_sec": round(best["start_offset_sec"], 3),
            "end_offset_sec": round(best["end_offset_sec"], 3),
            "sample_count": best["sample_count"],
            "quality_score": None if best["score"] < 0 else round(float(best["score"]), 4),
            "sampling_rate_hz": round(float(fs), 3),
            "collected_sec": round(duration_sec, 3),
            "selected_sec": selected_window_sec,
        },
    }
