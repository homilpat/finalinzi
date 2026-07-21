from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import json

import joblib
import pandas as pd

from gait_axis_aligned_core import (
    FEATURES, DAILY_FEATURES,
    extract_axis_aligned_gait_features,
    extract_subwindow_daily_features,
    load_sensor_csv_with_metadata,
    _acc_columns,
    align_to_vmlap,
    resample_array_to_100hz,
    transform_signal,
    extract_subwindow_daily_features_from_vmlap,
    TARGET_FS_HZ,
)


MODEL_CANDIDATES = [
    "gait_axis_aligned_physionet_youden.joblib",
]

DAILY_MODEL_NAME = "gait_daily_clinical_3feat.joblib"


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
    """CSV → (vmlap 100Hz 배열, duration_sec, metadata dict)"""
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
    return vmlap, duration, window_meta


def predict_daily_gait_csv(source: str | BinaryIO, model_dir: str | Path) -> dict:
    """
    3-feature 일상보행 모델 (v_jerk_rms_median/iqr, v_harmonic_ratio_iqr)
    훈련: PhysioNet 75h 임상 OR 라벨 (motor_impairment_score ≥ 0.5)
    subject AUC=0.881, sens=0.971, spec=0.722

    도메인 보정 우선순위:
      1) signal_correction (alpha + tau): 신호 레벨 변환 후 피처 추출 — 권장
      2) domain_correction (additive delta): 피처 레벨 덧셈 — 폴백
    """
    import numpy as np

    model_dir     = Path(model_dir)
    artifact_path = model_dir / DAILY_MODEL_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Daily gait model not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    sig_corr = artifact.get("signal_correction")  # alpha, tau
    additive = artifact.get("domain_correction", {})

    vmlap, duration, window_meta = _csv_to_vmlap(source)

    if sig_corr:
        # ── 신호 레벨 보정 (권장) ──────────────────────────────────────────
        alpha = float(sig_corr.get("alpha", 1.0))
        tau   = float(sig_corr.get("tau",   1.0))
        corrected_vmlap = transform_signal(vmlap, alpha, tau)
        extracted = extract_subwindow_daily_features_from_vmlap(corrected_vmlap, duration)
        features  = extracted["features"]
        correction_mode = "signal"
        correction_applied = {"alpha": alpha, "tau": tau}
    else:
        # ── 피처 레벨 보정 (폴백) ─────────────────────────────────────────
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
