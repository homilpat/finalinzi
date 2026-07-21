from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import json

import joblib
import pandas as pd

from gait_axis_aligned_core import FEATURES, extract_axis_aligned_gait_features


MODEL_CANDIDATES = [
    "gait_axis_aligned_physionet_youden.joblib",
]


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
