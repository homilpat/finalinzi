from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
from typing import BinaryIO

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt, spectrogram


TARGET_FS_HZ = 100.0
WINDOW_SEC = 10.0
FEATURES = [
    "v_harmonic_ratio",
    "ap_harmonic_ratio",
    "v_stride_freq_hz",
    "ap_spec_entropy",
]
G_IN_M_S2 = 9.80665


def _read_text(source: str | BinaryIO) -> str:
    if hasattr(source, "read"):
        raw = source.read()
        return raw.decode("utf-8-sig") if isinstance(raw, bytes) else str(raw)
    return Path(source).read_text(encoding="utf-8-sig")


def parse_sensor_metadata(text: str) -> dict:
    metadata: dict[str, float | str] = {}
    patterns = {
        "accel_range_m_s2": r"(?:Accel|Accelerometer|Acc).*?(?:Maximum_?Range|Range).*?=?\s*([0-9.]+)\s*(?:m/s\^?2|m_s2)",
        "accel_range_g": r"(?:Accel|Accelerometer|Acc).*?(?:Maximum_?Range|Range).*?=?\s*([0-9.]+)\s*g\b",
        "gyro_range_rad_s": r"(?:Gyro|Gyroscope).*?(?:Maximum_?Range|Range).*?=?\s*([0-9.]+)\s*(?:rad/s|rad_s)",
        "gyro_range_deg_s": r"(?:Gyro|Gyroscope).*?(?:Maximum_?Range|Range).*?=?\s*([0-9.]+)\s*(?:deg/s|dps|deg_s)",
        "accel_resolution": r"(?:Accel|Accelerometer|Acc).*?Resolution.*?=?\s*([0-9.eE+-]+)",
        "gyro_resolution": r"(?:Gyro|Gyroscope).*?Resolution.*?=?\s*([0-9.eE+-]+)",
    }
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        cleaned = line[1:].strip()
        if ":" in cleaned:
            key, value = cleaned.split(":", 1)
            raw_key = key.strip()
            raw_value = value.strip()
            metadata[raw_key] = raw_value
            numeric = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", raw_value)
            if numeric:
                canonical = {
                    "Accel_Maximum_Range_m_s2": "accel_range_m_s2",
                    "Accel_Maximum_Range_g": "accel_range_g",
                    "Gyro_Maximum_Range_rad_s": "gyro_range_rad_s",
                    "Gyro_Maximum_Range_deg_s": "gyro_range_deg_s",
                    "Accel_Resolution_m_s2": "accel_resolution",
                    "Gyro_Resolution_rad_s": "gyro_resolution",
                }.get(raw_key)
                if canonical:
                    metadata[canonical] = float(numeric.group(0))
        for key, pattern in patterns.items():
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match and key not in metadata:
                metadata[key] = float(match.group(1))
    if "accel_range_g" in metadata and "accel_range_m_s2" not in metadata:
        metadata["accel_range_m_s2"] = float(metadata["accel_range_g"]) * G_IN_M_S2
    if "gyro_range_deg_s" in metadata and "gyro_range_rad_s" not in metadata:
        metadata["gyro_range_rad_s"] = float(np.deg2rad(float(metadata["gyro_range_deg_s"])))
    return metadata


def _find_header_line(text: str) -> int:
    for idx, line in enumerate(text.splitlines()):
        if line.startswith("Timestamp_ns"):
            return idx
    raise ValueError("Timestamp_ns header row was not found.")


def load_sensor_csv(source: str | BinaryIO) -> pd.DataFrame:
    text = _read_text(source)
    header = _find_header_line(text)
    df = pd.read_csv(StringIO("\n".join(text.splitlines()[header:])))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp_ns" not in df.columns:
        raise ValueError("Timestamp_ns column is required.")
    return df.dropna(subset=["Timestamp_ns"]).sort_values("Timestamp_ns").reset_index(drop=True)


def load_sensor_csv_with_metadata(source: str | BinaryIO) -> tuple[pd.DataFrame, dict]:
    text = _read_text(source)
    header = _find_header_line(text)
    df = pd.read_csv(StringIO("\n".join(text.splitlines()[header:])))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp_ns" not in df.columns:
        raise ValueError("Timestamp_ns column is required.")
    df = df.dropna(subset=["Timestamp_ns"]).sort_values("Timestamp_ns").reset_index(drop=True)
    return df, parse_sensor_metadata(text)


def _range_quality(values: np.ndarray, sensor_range: float | None) -> dict:
    if sensor_range is None or not np.isfinite(sensor_range) or sensor_range <= 0:
        return {"range_available": False}
    max_abs = float(np.nanmax(np.abs(values))) if values.size else np.nan
    ratio = max_abs / float(sensor_range) if np.isfinite(max_abs) else np.nan
    return {
        "range_available": True,
        "range": float(sensor_range),
        "max_abs": max_abs,
        "max_to_range_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "saturation_risk": bool(np.isfinite(ratio) and ratio >= 0.98),
    }


def _calibrate_acc_values(acc: np.ndarray, metadata: dict, already_vmlap: bool) -> tuple[np.ndarray, dict]:
    acc = np.asarray(acc, dtype=float)
    acc_range_raw = metadata.get("accel_range_m_s2")
    acc_range = float(acc_range_raw) if isinstance(acc_range_raw, (int, float, np.floating)) else None
    norm_median = float(np.nanmedian(np.linalg.norm(acc, axis=1))) if len(acc) else np.nan
    info = {
        "acc_input_median_norm": norm_median,
        "acc_range_quality": _range_quality(acc, acc_range),
        "acc_calibration": "none",
    }
    if already_vmlap:
        info["acc_calibration"] = "provided_vmlap_g"
        return acc, info
    if np.isfinite(norm_median) and norm_median > 3.0:
        if acc_range and np.nanmax(np.abs(acc)) > acc_range * 1.05:
            acc = np.clip(acc, -acc_range, acc_range)
            info["acc_calibration"] = "m_s2_clipped_to_sensor_range_then_g"
        else:
            info["acc_calibration"] = "m_s2_to_g"
        return acc / G_IN_M_S2, info
    if acc_range and np.isfinite(norm_median) and norm_median <= 1.5 and acc_range > 3.0:
        scaled = acc * acc_range
        scaled_norm = float(np.nanmedian(np.linalg.norm(scaled, axis=1)))
        if 6.0 <= scaled_norm <= 13.5:
            info["acc_calibration"] = "normalized_range_fraction_to_g"
            return scaled / G_IN_M_S2, info
    info["acc_calibration"] = "assumed_g"
    return acc, info


def _sampling_rate_from_timestamp(df: pd.DataFrame) -> float:
    t = np.unique(df["Timestamp_ns"].to_numpy(float))
    if len(t) < 2:
        return TARGET_FS_HZ
    duration = (t[-1] - t[0]) / 1e9
    return float((len(t) - 1) / duration) if duration > 0 else TARGET_FS_HZ


def _acc_columns(df: pd.DataFrame, metadata: dict | None = None) -> tuple[np.ndarray, bool, tuple[str, str, str], dict]:
    metadata = metadata or {}
    anatomical = ["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]
    if set(anatomical).issubset(df.columns):
        acc, calibration = _calibrate_acc_values(df[anatomical].to_numpy(float), metadata, already_vmlap=True)
        return acc, True, ("V", "ML", "AP"), calibration
    raw = ["Acc_X", "Acc_Y", "Acc_Z"]
    if set(raw).issubset(df.columns):
        acc = df[raw].to_numpy(float)
        acc, calibration = _calibrate_acc_values(acc, metadata, already_vmlap=False)
        return acc, False, ("raw_X", "raw_Y", "raw_Z"), calibration
    raise ValueError("Missing accelerometer columns.")


def _resample_by_timestamp(df: pd.DataFrame, values: np.ndarray, target_fs: float = TARGET_FS_HZ) -> tuple[np.ndarray, float]:
    t = df["Timestamp_ns"].to_numpy(float)
    keep = np.isfinite(t) & np.isfinite(values).all(axis=1)
    t = t[keep]
    values = values[keep]
    if len(t) < 4:
        raise ValueError("Not enough valid sensor rows.")
    grouped = pd.DataFrame({"t": t, "x": values[:, 0], "y": values[:, 1], "z": values[:, 2]}).groupby("t", as_index=False).mean()
    elapsed = (grouped["t"].to_numpy(float) - float(grouped["t"].iloc[0])) / 1e9
    duration = float(elapsed[-1])
    if duration <= 0:
        raise ValueError("Invalid sensor duration.")
    grid = np.arange(0.0, duration, 1.0 / target_fs)
    out = np.column_stack([np.interp(grid, elapsed, grouped[col].to_numpy(float)) for col in ["x", "y", "z"]])
    return out, duration


def resample_array_to_100hz(values: np.ndarray, source_fs: float, target_fs: float = TARGET_FS_HZ) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 4:
        raise ValueError("Not enough finite sensor rows.")
    if source_fs <= 0:
        source_fs = target_fs
    t = np.arange(len(values), dtype=float) / float(source_fs)
    grid = np.arange(0.0, t[-1], 1.0 / target_fs)
    return np.column_stack([np.interp(grid, t, values[:, i]) for i in range(values.shape[1])])


def bandpass(x: np.ndarray, fs: float = TARGET_FS_HZ, low: float = 0.6, high: float = 3.0) -> np.ndarray:
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


def align_to_vmlap(acc: np.ndarray, already_vmlap: bool, fs: float = TARGET_FS_HZ) -> tuple[np.ndarray, dict]:
    acc = np.asarray(acc, dtype=float)[:, :3]
    acc = acc[np.isfinite(acc).all(axis=1)]
    if len(acc) < 4:
        raise ValueError("not enough finite accelerometer rows")
    if already_vmlap:
        out = acc.copy()
        if np.nanmedian(out[:, 0]) < 0:
            out[:, 0] *= -1
        return out, {"alignment": "provided_vmlap", "vertical_raw_axis": 0, "vertical_sign": "+"}

    med = np.nanmedian(acc, axis=0)
    vertical_idx = int(np.nanargmax(np.abs(med)))
    v = acc[:, vertical_idx] * (1.0 if med[vertical_idx] >= 0 else -1.0)
    remaining = [idx for idx in range(3) if idx != vertical_idx]
    powers = [float(np.nanvar(bandpass(acc[:, idx], fs))) for idx in remaining]
    if not np.isfinite(powers).any():
        powers = [float(np.nanvar(acc[:, idx])) for idx in remaining]
    ap_pos = int(np.nanargmax(powers))
    ap_idx = remaining[ap_pos]
    ml_idx = remaining[1 - ap_pos]
    ap = acc[:, ap_idx]
    ml = acc[:, ml_idx]
    if np.nanmedian(ap) < 0:
        ap *= -1
    if np.nanmedian(ml) < 0:
        ml *= -1
    return np.column_stack([v, ml, ap]), {
        "alignment": "gravity_plus_horizontal_power",
        "vertical_raw_axis": vertical_idx,
        "vertical_sign": "+" if med[vertical_idx] >= 0 else "-",
        "ml_raw_axis": ml_idx,
        "ap_raw_axis": ap_idx,
        "horizontal_power_axis0": powers[0],
        "horizontal_power_axis1": powers[1],
    }


def acf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    c = correlate(x, x, mode="full")[len(x) - 1 :]
    c = c / np.arange(len(x), 0, -1)
    if c[0] > 1e-12:
        c = c / c[0]
    return c


def peak_in_range(c: np.ndarray, fs: float, low_sec: float, high_sec: float) -> tuple[float, float, float]:
    lo = max(1, int(round(low_sec * fs)))
    hi = min(len(c) - 1, int(round(high_sec * fs)))
    if hi <= lo:
        return np.nan, np.nan, np.nan
    seg = c[lo : hi + 1]
    peaks, props = find_peaks(seg, prominence=0.03)
    if len(peaks):
        idx = lo + int(peaks[np.argmax(props["prominences"])])
    else:
        idx = lo + int(np.nanargmax(seg))
    height = float(c[idx])
    half = max(0.0, height * 0.5)
    left = idx
    while left > 1 and c[left] >= half:
        left -= 1
    right = idx
    while right < len(c) - 1 and c[right] >= half:
        right += 1
    return idx / fs, height, (right - left) / fs


def spec_entropy(x: np.ndarray, fs: float) -> tuple[float, float]:
    freqs, _, pxx = spectrogram(
        x - np.nanmean(x),
        fs=fs,
        window="hann",
        nperseg=max(32, min(len(x), int(round(4 * fs)))),
        detrend=False,
        scaling="density",
        mode="psd",
    )
    mask = (freqs >= 0.6) & (freqs <= 3.0)
    if not np.any(mask):
        return np.nan, np.nan
    band = pxx[mask, :]
    mean_spec = np.nanmean(band, axis=1)
    total = float(np.nansum(mean_spec))
    if total <= 1e-12:
        return np.nan, np.nan
    peak = int(np.nanargmax(mean_spec))
    prob = band.reshape(-1)
    prob = prob / (np.nansum(prob) + 1e-12)
    entropy = float(-np.nansum(prob * np.log2(prob + 1e-12)) / np.log2(len(prob))) if len(prob) > 1 else 0.0
    return float(mean_spec[peak] / total), entropy


def window_features(vmlap: np.ndarray) -> dict:
    fs = TARGET_FS_HZ
    v  = bandpass(vmlap[:, 0], fs)
    ap = bandpass(vmlap[:, 2], fs)

    c_v  = acf(v)
    c_ap = acf(ap)

    v_stride_lag,  v_stride_peak,  v_stride_width  = peak_in_range(c_v,  fs, 0.80, 1.70)
    ap_stride_lag, ap_stride_peak, ap_stride_width = peak_in_range(c_ap, fs, 0.80, 1.70)

    # Step peak (AD1) at half stride lag — Moe-Nilssen & Helbostad 2004
    if np.isfinite(v_stride_lag) and v_stride_lag > 0:
        half_v = v_stride_lag / 2.0
        _, v_step_peak, _ = peak_in_range(c_v, fs, half_v * 0.6, half_v * 1.4)
        v_hr = float(v_step_peak / v_stride_peak) if (np.isfinite(v_stride_peak) and v_stride_peak > 1e-6) else np.nan
        v_stride_freq = 1.0 / v_stride_lag
    else:
        v_step_peak  = np.nan
        v_hr         = np.nan
        v_stride_freq = np.nan

    if np.isfinite(ap_stride_lag) and ap_stride_lag > 0:
        half_ap = ap_stride_lag / 2.0
        _, ap_step_peak, _ = peak_in_range(c_ap, fs, half_ap * 0.6, half_ap * 1.4)
        ap_hr = float(ap_step_peak / ap_stride_peak) if (np.isfinite(ap_stride_peak) and ap_stride_peak > 1e-6) else np.nan
    else:
        ap_step_peak = np.nan
        ap_hr        = np.nan

    _, ap_entropy = spec_entropy(ap, fs)

    # vertical jerk RMS — Kavanagh & Menz 2008; higher = jerkier (worse smoothness)
    v_jerk_rms = float(np.sqrt(np.mean(np.diff(v)**2)) * fs) if len(v) > 1 else np.nan

    return {
        "v_acf_stride_peak":            v_stride_peak,
        "v_acf_stride_peak_width_sec":  v_stride_width,
        "v_acf_step_peak":              v_step_peak,
        "v_harmonic_ratio":             v_hr,
        "v_stride_freq_hz":             v_stride_freq,
        "v_jerk_rms":                   v_jerk_rms,
        "ap_acf_stride_peak":           ap_stride_peak,
        "ap_acf_stride_peak_width_sec": ap_stride_width,
        "ap_acf_step_peak":             ap_step_peak,
        "ap_harmonic_ratio":            ap_hr,
        "ap_spec_entropy":              ap_entropy,
        "quality_score":                v_stride_peak,
    }


def extract_best10_from_vmlap(vmlap: np.ndarray, duration_sec: float | None = None, window_sec: float = WINDOW_SEC) -> dict:
    win = int(round(window_sec * TARGET_FS_HZ))
    if len(vmlap) < int(0.8 * win):
        raise ValueError("At least 8 seconds of gait signal is required.")
    if len(vmlap) < win:
        vmlap = np.pad(vmlap, ((0, win - len(vmlap)), (0, 0)), mode="edge")
    starts = range(0, len(vmlap) - win + 1, max(1, int(round(2 * TARGET_FS_HZ))))
    best = None
    for start in starts:
        feat = window_features(vmlap[start : start + win])
        if best is None or feat.get("quality_score", -np.inf) > best.get("quality_score", -np.inf):
            best = {**feat, "start_sec": start / TARGET_FS_HZ, "end_sec": (start + win) / TARGET_FS_HZ}
    if best is None:
        raise ValueError("Could not extract gait features.")
    return {
        "features": {name: float(best.get(name, np.nan)) for name in FEATURES},
        "all_features": {name: float(value) for name, value in best.items() if isinstance(value, (int, float, np.floating))},
        "window": {
            "start_sec": best["start_sec"],
            "end_sec": best["end_sec"],
            "duration_sec": float(duration_sec) if duration_sec is not None else len(vmlap) / TARGET_FS_HZ,
            "target_fs": TARGET_FS_HZ,
        },
    }


def _axis_scale_array(axis_scale: dict | None) -> np.ndarray | None:
    if not axis_scale:
        return None
    values = [
        axis_scale.get("vertical", 1.0),
        axis_scale.get("mediolateral", 1.0),
        axis_scale.get("anteroposterior", 1.0),
    ]
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        return None
    return arr


def extract_best10_from_acc_array(
    acc: np.ndarray,
    source_fs: float,
    already_vmlap: bool,
    axis_scale: dict | None = None,
) -> dict:
    aligned, alignment = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=source_fs)
    scale = _axis_scale_array(axis_scale)
    if scale is not None:
        aligned = aligned * scale.reshape(1, 3)
        alignment["axis_scale_v_ml_ap"] = {
            "vertical": float(scale[0]),
            "mediolateral": float(scale[1]),
            "anteroposterior": float(scale[2]),
        }
    resampled = resample_array_to_100hz(aligned, source_fs)
    extracted = extract_best10_from_vmlap(resampled, duration_sec=len(acc) / float(source_fs or TARGET_FS_HZ))
    extracted["window"].update(alignment)
    return extracted


def extract_axis_aligned_gait_features(source: str | BinaryIO, axis_scale: dict | None = None) -> dict:
    df, metadata = load_sensor_csv_with_metadata(source)
    acc, already_vmlap, axes, calibration = _acc_columns(df, metadata)
    t = df["Timestamp_ns"].to_numpy(float)
    duration = (np.nanmax(t) - np.nanmin(t)) / 1e9 if len(t) else np.nan
    observed_fs = float(len(df) / duration) if np.isfinite(duration) and duration > 0 else TARGET_FS_HZ
    duration = len(df) / observed_fs if observed_fs > 0 else len(df) / TARGET_FS_HZ
    extracted = extract_best10_from_acc_array(acc, observed_fs, already_vmlap=already_vmlap, axis_scale=axis_scale)
    extracted["window"].update(
        {
            "observed_fs": observed_fs,
            "input_axes": axes,
            "duration_sec": duration,
            "sensor_metadata": metadata,
            "calibration": calibration,
        }
    )
    return extracted


DAILY_FEATURES = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

_DAILY_WIN20   = int(20 * TARGET_FS_HZ)
_DAILY_SUB_WIN = int(10 * TARGET_FS_HZ)
_DAILY_STEP    = int(2  * TARGET_FS_HZ)


def transform_signal(vmlap: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """
    신호 레벨 도메인 보정.
    alpha: 진폭 배율 (jerk_rms ∝ alpha)
    tau:   시간축 배율 (>1 신호 길어짐, <1 짧아짐 → 보행 주파수 변화)
    100Hz 그리드 유지, 배열 길이만 변함.
    """
    scaled = vmlap * float(alpha)
    n = len(scaled)
    n_warped = max(10, int(round(n * float(tau))))
    if n_warped == n:
        return scaled
    t_orig = np.linspace(0.0, 1.0, n)
    t_warp = np.linspace(0.0, 1.0, n_warped)
    return np.column_stack([
        np.interp(t_warp, t_orig, scaled[:, i])
        for i in range(scaled.shape[1])
    ])


def extract_subwindow_daily_features_from_vmlap(vmlap: np.ndarray, duration_sec: float | None = None) -> dict:
    """
    VMLAP 100Hz 배열 → v_jerk_rms_median/iqr, v_harmonic_ratio_iqr
    extract_subwindow_daily_features()와 동일 로직이나 CSV 로드 없이 배열 직접 수신.
    """
    sub_feats: list[dict] = []
    n = len(vmlap)
    seg_starts = list(range(0, max(1, n - _DAILY_WIN20 + 1), _DAILY_WIN20 // 2))
    if not seg_starts:
        seg_starts = [0]
    for w0 in seg_starts:
        seg = vmlap[w0 : w0 + _DAILY_WIN20]
        if len(seg) < int(0.5 * _DAILY_WIN20):
            continue
        for s in range(0, max(1, len(seg) - _DAILY_SUB_WIN + 1), _DAILY_STEP):
            sub = seg[s : s + _DAILY_SUB_WIN]
            if len(sub) < int(0.8 * _DAILY_SUB_WIN):
                continue
            try:
                f = window_features(sub)
                sub_feats.append({
                    "v_harmonic_ratio": f.get("v_harmonic_ratio", np.nan),
                    "v_jerk_rms":       f.get("v_jerk_rms",       np.nan),
                })
            except Exception:
                continue

    if len(sub_feats) < 2:
        dur = duration_sec or n / TARGET_FS_HZ
        raise ValueError(
            f"분석 가능한 구간이 부족합니다 (필요: ≥20초 보행, 현재: {dur:.1f}초)"
        )

    arr      = pd.DataFrame(sub_feats)
    hr_vals  = arr["v_harmonic_ratio"].dropna()
    jrk_vals = arr["v_jerk_rms"].dropna()

    return {
        "features": {
            "v_jerk_rms_median":    float(jrk_vals.median())                                   if len(jrk_vals) >= 2 else np.nan,
            "v_jerk_rms_iqr":       float(jrk_vals.quantile(0.75) - jrk_vals.quantile(0.25))  if len(jrk_vals) >= 2 else np.nan,
            "v_harmonic_ratio_iqr": float(hr_vals.quantile(0.75)  - hr_vals.quantile(0.25))   if len(hr_vals)  >= 2 else np.nan,
        },
        "window": {
            "n_sub_windows": len(sub_feats),
            "duration_sec":  float(duration_sec or n / TARGET_FS_HZ),
        },
    }


def extract_subwindow_daily_features(source: str | BinaryIO, axis_scale: dict | None = None) -> dict:
    """
    CSV → V/ML/AP 정렬 → 20s 윈도우 내 10s 슬라이딩 sub-windows
    → v_jerk_rms_median, v_jerk_rms_iqr, v_harmonic_ratio_iqr 반환
    (PhysioNet 75h 임상 라벨 모델 입력 피처)
    """
    df, metadata = load_sensor_csv_with_metadata(source)
    acc, already_vmlap, axes, calibration = _acc_columns(df, metadata)
    t = df["Timestamp_ns"].to_numpy(float)
    duration    = (np.nanmax(t) - np.nanmin(t)) / 1e9 if len(t) else np.nan
    observed_fs = float(len(df) / duration) if np.isfinite(duration) and duration > 0 else TARGET_FS_HZ
    duration    = len(df) / observed_fs if observed_fs > 0 else len(df) / TARGET_FS_HZ

    aligned, alignment = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=observed_fs)
    scale = _axis_scale_array(axis_scale)
    if scale is not None:
        aligned = aligned * scale.reshape(1, 3)
    vmlap = resample_array_to_100hz(aligned, observed_fs)

    sub_feats: list[dict] = []
    n = len(vmlap)
    # 20s 단위로 분할; 데이터가 짧으면 전체를 하나의 세그먼트로 처리
    seg_starts = list(range(0, max(1, n - _DAILY_WIN20 + 1), _DAILY_WIN20 // 2))
    if not seg_starts:
        seg_starts = [0]
    for w0 in seg_starts:
        seg = vmlap[w0 : w0 + _DAILY_WIN20]
        if len(seg) < int(0.5 * _DAILY_WIN20):  # 10s 이상이면 허용
            continue
        win_len = len(seg)
        for s in range(0, max(1, win_len - _DAILY_SUB_WIN + 1), _DAILY_STEP):
            sub = seg[s : s + _DAILY_SUB_WIN]
            if len(sub) < int(0.8 * _DAILY_SUB_WIN):
                continue
            try:
                f = window_features(sub)
                sub_feats.append({
                    "v_harmonic_ratio": f.get("v_harmonic_ratio", np.nan),
                    "v_jerk_rms":       f.get("v_jerk_rms",       np.nan),
                })
            except Exception:
                continue

    if len(sub_feats) < 2:
        raise ValueError(
            f"분석 가능한 구간이 부족합니다 (필요: ≥20초 보행, 현재: {duration:.1f}초)"
        )

    arr      = pd.DataFrame(sub_feats)
    hr_vals  = arr["v_harmonic_ratio"].dropna()
    jrk_vals = arr["v_jerk_rms"].dropna()

    features = {
        "v_jerk_rms_median":    float(jrk_vals.median())                              if len(jrk_vals) >= 2 else np.nan,
        "v_jerk_rms_iqr":       float(jrk_vals.quantile(0.75) - jrk_vals.quantile(0.25)) if len(jrk_vals) >= 2 else np.nan,
        "v_harmonic_ratio_iqr": float(hr_vals.quantile(0.75)  - hr_vals.quantile(0.25))  if len(hr_vals)  >= 2 else np.nan,
    }
    window = {
        "n_sub_windows": len(sub_feats),
        "duration_sec":  float(duration),
        "observed_fs":   float(observed_fs),
        "input_axes":    axes,
        "calibration":   calibration,
        **alignment,
    }
    return {"features": features, "window": window}
