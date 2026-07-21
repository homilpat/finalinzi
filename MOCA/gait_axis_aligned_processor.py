from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import json

import joblib
import pandas as pd

from gait_axis_aligned_core import (
    FEATURES, DAILY_FEATURES, BEST10_FEATURES,
    extract_axis_aligned_gait_features,
    extract_subwindow_daily_features,
    load_sensor_csv_with_metadata,
    _acc_columns,
    align_to_vmlap,
    resample_array_to_100hz,
    transform_signal,
    extract_subwindow_daily_features_from_vmlap,
    extract_best10_daily_features_from_vmlap,
    TARGET_FS_HZ,
)


MODEL_CANDIDATES = [
    "gait_axis_aligned_physionet_youden.joblib",
]

DAILY_MODEL_NAME       = "gait_daily_clinical_3feat.joblib"
DAILY_BEST10_MODEL_NAME = "gait_daily_best10_3feat.joblib"


def _load_axis_aligned_artifact(model_dir: str | Path) -> tuple[dict, str]:
    model_dir = Path(model_dir)
    for name in MODEL_CANDIDATES:
        path = model_dir / name
        if path.exists():
            return joblib.load(path), name
    raise FileNotFoundError(f"No axis-aligned gait model found in {model_dir}")


def _load_waist_sensor_calibration(model_dir: str | Path) -> dict | None:
    path = Path(model_dir) / "waist_sensor_range_loss_calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_to_vmlap(source: str | BinaryIO) -> tuple:
    """CSV → (vmlap 100Hz 배열, duration_sec, metadata dict, df, observed_fs, alignment)"""
    import numpy as np
    df, metadata = load_sensor_csv_with_metadata(source)
    acc, already_vmlap, axes, calibration = _acc_columns(df, metadata)
    t = df["Timestamp_ns"].to_numpy(float)
    duration = (float(t.max()) - float(t.min())) / 1e9
    observed_fs = float(len(df) / duration) if duration > 0 else TARGET_FS_HZ
    aligned, alignment = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=observed_fs)
    vmlap = resample_array_to_100hz(aligned, observed_fs)
    window_meta = {
        "observed_fs": observed_fs,
        "input_axes": axes,
        "duration_sec": duration,
        "sensor_metadata": metadata,
        "calibration": calibration,
        **alignment,
    }
    return vmlap, duration, window_meta, df, observed_fs, alignment


def _extract_gyro_pitch(df: "pd.DataFrame", alignment: dict, observed_fs: float) -> "np.ndarray | None":
    """
    Gyro_Clean_X/Y/Z에서 pitch(ML 방향 각속도) 추출 → 100Hz 배열 반환.
    ml_raw_axis: align_to_vmlap이 반환한 ML 축 인덱스(0=X,1=Y,2=Z).
    already_vmlap 케이스(앱 CSV)에서는 raw Acc_X/Y/Z로 별도 align 수행.
    """
    import numpy as np
    gyro_cols = ["Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z"]
    if not all(c in df.columns for c in gyro_cols):
        return None

    gyro_raw = df[gyro_cols].to_numpy(float)

    ml_raw_axis = alignment.get("ml_raw_axis")
    if ml_raw_axis is None:
        # already_vmlap 케이스: raw Acc_X/Y/Z로 axis mapping 재계산
        raw_acc_cols = ["Acc_X", "Acc_Y", "Acc_Z"]
        if not all(c in df.columns for c in raw_acc_cols):
            return None
        raw_acc = df[raw_acc_cols].to_numpy(float)
        try:
            _, raw_align = align_to_vmlap(raw_acc, already_vmlap=False, fs=observed_fs)
            ml_raw_axis = raw_align.get("ml_raw_axis")
        except Exception:
            return None
    if ml_raw_axis is None:
        return None

    pitch_series = gyro_raw[:, int(ml_raw_axis)]
    pitch_100hz = resample_array_to_100hz(pitch_series.reshape(-1, 1), observed_fs)[:, 0]
    return pitch_100hz


def predict_daily_gait_csv(source: str | BinaryIO, model_dir: str | Path) -> dict:
    """
    3-feature acc-only 일상보행 모델
    (v_jerk_rms_median, v_jerk_rms_iqr, v_harmonic_ratio_iqr)
    훈련: PhysioNet 75h 임상 OR 라벨 (motor_impairment_score ≥ 0.5)

    도메인 보정:
      signal_correction (alpha + tau): 가속도 신호 레벨 변환 후 피처 추출
    """
    import numpy as np

    model_dir     = Path(model_dir)
    artifact_path = model_dir / DAILY_MODEL_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Daily gait model not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    sig_corr = artifact.get("signal_correction")
    additive = artifact.get("domain_correction", {})

    vmlap, duration, window_meta, df, observed_fs, alignment = _csv_to_vmlap(source)

    if sig_corr:
        alpha = float(sig_corr.get("alpha", 1.0))
        tau   = float(sig_corr.get("tau",   1.0))
        corrected_vmlap = transform_signal(vmlap, alpha, tau)
        extracted = extract_subwindow_daily_features_from_vmlap(corrected_vmlap, duration)
        features  = extracted["features"]
        correction_mode = "signal"
        correction_applied = {"alpha": alpha, "tau": tau}
    else:
        extracted  = extract_subwindow_daily_features_from_vmlap(vmlap, duration)
        raw_feats  = extracted["features"]
        features   = {
            f: (raw_feats.get(f, float("nan")) + additive.get(f, 0.0))
            for f in DAILY_FEATURES
        }
        correction_mode    = "additive"
        correction_applied = additive

    window_meta.update(extracted["window"])

    X           = np.array([[features.get(f, float("nan")) for f in DAILY_FEATURES]])
    probability = float(artifact["pipeline"].predict_proba(X)[:, 1][0])
    threshold   = float(artifact.get("threshold", 0.5))
    prediction  = int(probability >= threshold)

    return {
        "probability":          probability,
        "threshold":            threshold,
        "prediction":           prediction,
        "label":                "이동기능 저하 가능성" if prediction else "이동기능 정상 범위 가능성",
        "threshold_strategy":   artifact.get("threshold_strategy"),
        "model_mode":           artifact.get("model_mode"),
        "model_artifact":       DAILY_MODEL_NAME,
        "correction_mode":      correction_mode,
        "correction_applied":   correction_applied,
        "features":             features,
        "window":               window_meta,
    }


def predict_daily_best10_gait_csv(source: str | BinaryIO, model_dir: str | Path) -> dict:
    """
    Best-10s 방식 일상보행 모델 (훈련·추론 파이프라인 완전 일치)
      - 20s 녹화 → best quality 10s 창 → v_jerk_rms, v_harmonic_ratio, pitch_band_rms
      - 모델: gait_daily_best10_3feat.joblib
    """
    import numpy as np

    model_dir     = Path(model_dir)
    artifact_path = model_dir / DAILY_BEST10_MODEL_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Best10 daily gait model not found: {artifact_path}")

    artifact   = joblib.load(artifact_path)
    sig_corr   = artifact.get("signal_correction")
    gyro_alpha = float(artifact.get("gyro_alpha", 1.0))
    feat_names = artifact.get("features", BEST10_FEATURES)

    vmlap, duration, window_meta, df, observed_fs, alignment = _csv_to_vmlap(source)
    gyro_pitch = _extract_gyro_pitch(df, alignment, observed_fs)

    if sig_corr:
        alpha = float(sig_corr.get("alpha", 1.0))
        tau   = float(sig_corr.get("tau",   1.0))
        corrected_vmlap = transform_signal(vmlap, alpha, tau)
        extracted = extract_best10_daily_features_from_vmlap(
            corrected_vmlap, duration, gyro_pitch=gyro_pitch, gyro_alpha=gyro_alpha
        )
        correction_mode    = "signal+gyro"
        correction_applied = {"alpha": alpha, "tau": tau, "gyro_alpha": gyro_alpha}
    else:
        extracted = extract_best10_daily_features_from_vmlap(
            vmlap, duration, gyro_pitch=gyro_pitch, gyro_alpha=gyro_alpha
        )
        correction_mode    = "gyro_only"
        correction_applied = {"gyro_alpha": gyro_alpha}

    features = extracted["features"]
    window_meta.update(extracted["window"])

    X           = np.array([[features.get(f, float("nan")) for f in feat_names]])
    probability = float(artifact["pipeline"].predict_proba(X)[:, 1][0])
    threshold   = float(artifact.get("threshold", 0.5))
    prediction  = int(probability >= threshold)

    return {
        "probability":        probability,
        "threshold":          threshold,
        "prediction":         prediction,
        "label":              "이동기능 저하 가능성" if prediction else "이동기능 정상 범위 가능성",
        "threshold_strategy": artifact.get("threshold_strategy"),
        "model_mode":         artifact.get("model_mode"),
        "model_artifact":     DAILY_BEST10_MODEL_NAME,
        "correction_mode":    correction_mode,
        "correction_applied": correction_applied,
        "features":           features,
        "window":             window_meta,
    }


def predict_axis_aligned_gait_csv(source: str | BinaryIO, model_dir: str | Path) -> dict:
    artifact, artifact_name = _load_axis_aligned_artifact(model_dir)
    waist_calibration = _load_waist_sensor_calibration(model_dir)
    axis_scale = None
    if waist_calibration:
        axis_scale = waist_calibration.get("axis_scale_v_ml_ap")
    extracted = extract_axis_aligned_gait_features(source, axis_scale=axis_scale)
    features = artifact.get("features", FEATURES)
    frame = pd.DataFrame([[extracted["features"][name] for name in features]], columns=features)
    probability = float(artifact["pipeline"].predict_proba(frame)[:, 1][0])
    threshold = float(artifact["threshold"])
    prediction = int(probability >= threshold)
    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "label": "이동기능 저하 가능성" if prediction else "이동기능 정상 범위 가능성",
        "threshold_strategy": artifact.get("threshold_strategy", "physionet_normal_domain_corrected_final_train_youden"),
        "model_mode": artifact.get("model_mode", "axis_aligned_physionet_normal_domain_corrected"),
        "model_artifact": artifact_name,
        "waist_sensor_calibration": waist_calibration,
        "features": extracted["features"],
        "window": extracted["window"],
    }
