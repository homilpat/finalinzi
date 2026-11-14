from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt


BASE_REQUIRED_COLUMNS = {"Timestamp_ns"}
ANATOMICAL_COLUMNS = {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g", "Gyro_Roll_deg_s"}
PORTRAIT_RAW_COLUMNS = {"Acc_X", "Acc_Y", "Acc_Z", "Gyro_Clean_X"}
GRAVITY_MPS2 = 9.80665
TARGET_FS_HZ = 100.0
TRIM_START_SEC = 0.0
TRIM_END_SEC = 3.0
LOWPASS_CUTOFF_HZ = 20.0
STEP_MIN_SEC = 0.35
STEP_MAX_SEC = 0.80
STRIDE_MIN_SEC = 0.80
STRIDE_MAX_SEC = 1.60
STRIDE_RATIO_MIN = 1.55
STRIDE_RATIO_MAX = 2.45
MIN_AP_STEP_REGULARITY = 0.30
MIN_AP_STRIDE_REGULARITY = 0.30
MIN_ACF_PROMINENCE = 0.05


def _find_header_line(text: str) -> int:
    for idx, line in enumerate(text.splitlines()):
        if line.startswith("Timestamp_ns"):
            return idx
    raise ValueError("CSV header row starting with Timestamp_ns was not found.")


def _axis_columns(df: pd.DataFrame) -> tuple[str, set[str]]:
    if ANATOMICAL_COLUMNS <= set(df.columns):
        return "anatomical_14col", ANATOMICAL_COLUMNS
    if PORTRAIT_RAW_COLUMNS <= set(df.columns):
        return "portrait_raw", PORTRAIT_RAW_COLUMNS
    expected = sorted(ANATOMICAL_COLUMNS | PORTRAIT_RAW_COLUMNS)
    raise ValueError(f"Missing gait sensor columns. Expected anatomical or portrait raw columns: {', '.join(expected)}")


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
    missing = BASE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    _, axis_required = _axis_columns(df)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(BASE_REQUIRED_COLUMNS | axis_required)).sort_values("Timestamp_ns").reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Not enough gait samples.")
    return df


def _sampling_rate(timestamp_ns: pd.Series) -> float:
    timestamps = np.unique(timestamp_ns.to_numpy(dtype=float))
    if len(timestamps) < 2:
        raise ValueError("Timestamp interval could not be calculated.")
    duration_sec = (timestamps[-1] - timestamps[0]) / 1e9
    elapsed_rate = (len(timestamps) - 1) / duration_sec if duration_sec > 0 else np.nan

    dt = np.diff(timestamps) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Timestamp interval could not be calculated.")
    median_rate = 1.0 / float(np.median(dt))
    fs = elapsed_rate
    if np.isfinite(median_rate) and np.isfinite(elapsed_rate):
        if elapsed_rate / 3.0 <= median_rate <= elapsed_rate * 3.0:
            fs = median_rate
    if not np.isfinite(fs) or fs < 10:
        raise ValueError(f"Sampling rate is too low or invalid: {fs:.2f} Hz")
    return fs


def _resample_to_uniform_hz(df: pd.DataFrame, target_fs: float = TARGET_FS_HZ) -> tuple[pd.DataFrame, float, float]:
    numeric_cols = [
        col
        for col in df.columns
        if col != "Timestamp_ns" and pd.api.types.is_numeric_dtype(df[col])
    ]
    grouped = (
        df[["Timestamp_ns", *numeric_cols]]
        .groupby("Timestamp_ns", as_index=False)
        .mean(numeric_only=True)
        .sort_values("Timestamp_ns")
        .reset_index(drop=True)
    )

    start_ns = float(grouped["Timestamp_ns"].iloc[0])
    elapsed = (grouped["Timestamp_ns"].to_numpy(dtype=float) - start_ns) / 1e9
    duration_sec = float(elapsed[-1])
    if duration_sec <= 0:
        raise ValueError("CSV duration is invalid.")

    step = 1.0 / target_fs
    uniform_t = np.arange(0.0, duration_sec + (step * 0.5), step)
    resampled = pd.DataFrame({"_elapsed_sec": uniform_t})
    resampled["Timestamp_ns"] = start_ns + (uniform_t * 1e9)

    for col in numeric_cols:
        values = grouped[col].to_numpy(dtype=float)
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


def _lowpass(values: np.ndarray, fs: float, cutoff: float = LOWPASS_CUTOFF_HZ, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(cutoff, nyq * 0.95)
    if cutoff <= 0:
        return values
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    if len(values) < 30:
        return values
    return sosfiltfilt(sos, values, axis=0)


def _iqr(values: np.ndarray) -> float:
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def _unbiased_acf(signal: np.ndarray) -> np.ndarray | None:
    centered = signal - np.nanmean(signal)
    denom = float(np.sum(centered * centered))
    if not np.isfinite(denom) or denom <= 0:
        return None

    acf = correlate(centered, centered, mode="full")[len(centered) - 1 :]
    acf = acf / np.arange(len(centered), 0, -1)
    if acf[0] > 1e-10:
        acf = acf / acf[0]
    return acf


def _pick_acf_peak(
    acf: np.ndarray,
    fs: float,
    low_sec: float,
    high_sec: float,
    min_height: float,
    min_prominence: float,
) -> int | None:
    low = max(1, int(round(low_sec * fs)))
    high = min(len(acf) - 1, int(round(high_sec * fs)))
    if high <= low:
        return None
    segment = acf[low : high + 1]
    peaks, props = find_peaks(segment, height=min_height, prominence=min_prominence)
    if len(peaks) == 0:
        return None
    best_local = int(peaks[np.argmax(props["prominences"])])
    peak_idx = low + best_local
    if peak_idx <= low + 1 or peak_idx >= high - 1:
        return None
    return peak_idx


def _ap_referenced_v_stride_regularity(v_signal: np.ndarray, ap_signal: np.ndarray, fs: float) -> float | None:
    ap_acf = _unbiased_acf(ap_signal)
    if ap_acf is None:
        return None
    step_idx = _pick_acf_peak(
        ap_acf,
        fs,
        STEP_MIN_SEC,
        STEP_MAX_SEC,
        MIN_AP_STEP_REGULARITY,
        MIN_ACF_PROMINENCE,
    )
    stride_idx = _pick_acf_peak(
        ap_acf,
        fs,
        STRIDE_MIN_SEC,
        STRIDE_MAX_SEC,
        MIN_AP_STRIDE_REGULARITY,
        MIN_ACF_PROMINENCE,
    )
    if step_idx is None or stride_idx is None:
        return None
    step_duration = step_idx / fs
    stride_duration = stride_idx / fs
    ratio = stride_duration / step_duration if step_duration > 0 else np.nan
    if not np.isfinite(ratio) or not (STRIDE_RATIO_MIN <= ratio <= STRIDE_RATIO_MAX):
        return None

    v_acf = _unbiased_acf(v_signal)
    if v_acf is None or stride_idx >= len(v_acf):
        return None
    value = float(v_acf[stride_idx])
    if not np.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def _gravity_corrected_portrait_axes(
    window: pd.DataFrame, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Portrait-raw 파일에서 중력 벡터를 추정해 anatomical 축을 계산한다.

    [1] 0.5 Hz lowpass로 중력 방향 추정 (보행 DC 편향 제거, 구버전 mean 대비 개선)
    [2] AP 방향: portrait Z축(스크린 법선)을 수평면에 투영 — Z→Y→X 순 폴백
        (PCA는 허리 장착 폰에서 보행 수직 진동이 수평면으로 누설되어 ap_hat 틀어짐 확인됨)
    [3] ML = AP × V (오른손 법칙)
    중력 추정 실패(g 크기 8~12 m/s² 범위 이탈) 시 portrait 고정 폴백(Y=V, X=ML, Z=AP).

    Returns: (v_g, ml_g, ap_g, roll_deg_s)
    """
    ax = window["Acc_X"].to_numpy(dtype=float)
    ay = window["Acc_Y"].to_numpy(dtype=float)
    az = window["Acc_Z"].to_numpy(dtype=float)
    raw = np.column_stack([ax, ay, az])

    # [1] 중력 추정: 0.5 Hz lowpass → 평균 (보행 비대칭 DC 편향 감소)
    if len(ax) >= 30:
        grav_lp = _lowpass(raw, fs, cutoff=0.5, order=2)
        grav = np.nanmean(grav_lp, axis=0)
    else:
        grav = np.array([np.nanmean(ax), np.nanmean(ay), np.nanmean(az)])

    g_mag = float(np.linalg.norm(grav))
    if not np.isfinite(g_mag) or not (8.0 <= g_mag <= 12.0):
        # 폴백: portrait 고정 가정 (Y=수직, X=좌우, Z=전후)
        roll_fb = np.rad2deg(window["Gyro_Clean_X"].to_numpy(dtype=float))
        return ay / GRAVITY_MPS2, ax / GRAVITY_MPS2, az / GRAVITY_MPS2, roll_fb

    # 수직 단위벡터 (Android 가속도계: specific force → 정지 시 위 방향)
    v_hat = grav / g_mag

    # [2] AP 방향: Z(스크린 법선)를 수평면 투영, 수직에 가까우면 Y→X 순 폴백
    _ap_candidates = [
        np.array([0.0, 0.0, 1.0]),   # Z: portrait 원래 AP (스크린 전방)
        np.array([0.0, 1.0, 0.0]),   # Y: 장축
        np.array([1.0, 0.0, 0.0]),   # X: 단축
    ]
    ap_hat = None
    for ap_ref in _ap_candidates:
        ap_proj = ap_ref - np.dot(ap_ref, v_hat) * v_hat
        ap_norm = float(np.linalg.norm(ap_proj))
        if ap_norm > 1e-4:
            ap_hat = ap_proj / ap_norm
            break
    if ap_hat is None:
        ap_hat = np.array([0.0, 0.0, 1.0])

    # [3] ML = AP × V (오른손 법칙)
    ml_hat = np.cross(ap_hat, v_hat)
    ml_hat /= float(np.linalg.norm(ml_hat))

    # anatomical 좌표로 투영 (g 단위)
    raw_g = raw / GRAVITY_MPS2
    v_g  = raw_g @ v_hat
    ml_g = raw_g @ ml_hat
    ap_g = raw_g @ ap_hat

    # Roll: AP 축 주변 각속도 (3축 자이로 있으면 정확 계산)
    gyro_cols = [c for c in ("Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z") if c in window.columns]
    if len(gyro_cols) == 3:
        gyro = np.column_stack([window[c].to_numpy(dtype=float) for c in gyro_cols])
        roll_deg_s = np.rad2deg(gyro @ ap_hat)
    else:
        roll_deg_s = np.rad2deg(window["Gyro_Clean_X"].to_numpy(dtype=float))

    return v_g, ml_g, ap_g, roll_deg_s


def _extract_window_features(window: pd.DataFrame, fs: float) -> dict[str, float | None]:
    if ANATOMICAL_COLUMNS <= set(window.columns):
        v_source = window["Acc_Vertical_g"].to_numpy(dtype=float)
        ml_source = window["Acc_ML_g"].to_numpy(dtype=float)
        ap_source = window["Acc_AP_g"].to_numpy(dtype=float)
        roll_raw = window["Gyro_Roll_deg_s"].to_numpy(dtype=float)
    else:
        # Portrait 모드: 중력 벡터로 anatomical 축 계산 (phone 기울기 자동 보정)
        v_source, ml_source, ap_source, roll_raw = _gravity_corrected_portrait_axes(window, fs)

    base_acc = _lowpass(np.column_stack([v_source, ml_source, ap_source]), fs)
    v = _bandpass(v_source, fs, low=0.6, high=3.0)
    ml = _bandpass(ml_source, fs, low=0.6, high=3.0)
    roll = _bandpass(roll_raw - np.nanmedian(roll_raw), fs, low=0.5, high=5.0)
    return {
        "v_amp_pool_median": float(np.nanmedian(np.abs(v))),
        "ml_amp_pool_iqr": _iqr(np.abs(ml)),
        "base_v_stride_regularity": _ap_referenced_v_stride_regularity(base_acc[:, 0], base_acc[:, 2], fs),
        "roll_amp_pool_iqr": _iqr(np.abs(roll)),
    }


def extract_gait_features_from_csv(
    source,
    selected_window_sec: float = 10.0,
    trim_start_sec: float = TRIM_START_SEC,
    trim_end_sec: float = TRIM_END_SEC,
) -> dict:
    df = load_gait_csv(source)
    axis_mode, _ = _axis_columns(df)
    observed_fs = _sampling_rate(df["Timestamp_ns"])
    df, duration_sec, fs = _resample_to_uniform_hz(df)

    if duration_sec < selected_window_sec:
        raise ValueError(f"CSV is shorter than {selected_window_sec:.0f} seconds.")

    step_sec = 1.0
    analysis_start_sec = trim_start_sec
    analysis_end_sec = duration_sec - trim_end_sec
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
            "axis_mode": axis_mode,
            "trim_start_sec": trim_start_sec,
            "trim_end_sec": trim_end_sec,
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
