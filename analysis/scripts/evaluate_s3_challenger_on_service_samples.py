"""Smoke-test the s3 challenger LR on the real 20-second service samples.

The deployed artifact is read-only. A separate challenger is trained on the
same 71 subject rows, then both models are evaluated after applying the
deployed signal-level domain correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from modeling.gait_axis_aligned_core import (
    TARGET_FS_HZ,
    extract_subwindow_daily_features_from_vmlap,
    transform_signal,
    window_features,
)
from modeling.gait_axis_aligned_processor import _csv_to_vmlap, predict_daily_gait_csv


SAMPLE_DIR = ROOT / "보행SAMPLE"
MODEL_DIR = ROOT / "backend" / "models"
DEPLOYED_MODEL = MODEL_DIR / "gait_daily_clinical_3feat.joblib"
FEATURE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "nested_feature_selection_final10_100rep"
    / "subject_candidate_table.csv"
)
S3_TARGET_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
OUT_DIR = ROOT / "analysis_outputs" / "s3_challenger_service_sample_smoke"

CHALLENGER_FEATURES = [
    "v_acf_step_stride_symmetry_iqr",
    "ap_spec_entropy_iqr",
    "v_jerk_rms_iqr",
]
THRESHOLD = 0.5
WIN20 = int(20 * TARGET_FS_HZ)
SUBWIN10 = int(10 * TARGET_FS_HZ)
STEP2 = int(2 * TARGET_FS_HZ)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )


def train_challenger() -> tuple[Pipeline, pd.DataFrame]:
    features = pd.read_csv(FEATURE_TABLE).drop(columns=["target"])
    targets = pd.read_csv(
        S3_TARGET_TABLE, usecols=["subject_id", "target"]
    )
    table = features.merge(
        targets, on="subject_id", how="inner", validate="one_to_one"
    ).dropna(subset=CHALLENGER_FEATURES + ["target"])
    pipeline = make_pipeline()
    pipeline.fit(
        table[CHALLENGER_FEATURES].to_numpy(float),
        table["target"].to_numpy(int),
    )
    return pipeline, table


def pad_near_20_seconds(vmlap: np.ndarray) -> tuple[np.ndarray, int]:
    """Edge-pad small timestamp shortfalls while rejecting truly short data."""
    if len(vmlap) >= WIN20:
        return vmlap, 0
    if len(vmlap) < int(19 * TARGET_FS_HZ):
        return vmlap, 0
    pad_count = WIN20 - len(vmlap)
    return np.pad(vmlap, ((0, pad_count), (0, 0)), mode="edge"), pad_count


def predict_current_from_vmlap(
    vmlap: np.ndarray, artifact: dict
) -> tuple[float, dict, int]:
    extracted = extract_subwindow_daily_features_from_vmlap(
        vmlap, len(vmlap) / TARGET_FS_HZ
    )
    feature_names = list(artifact["features"])
    vector = np.asarray(
        [extracted["features"].get(name, np.nan) for name in feature_names],
        dtype=float,
    )
    probability = float(
        artifact["pipeline"].predict_proba(vector.reshape(1, -1))[0, 1]
    )
    return (
        probability,
        extracted["features"],
        int(extracted["window"]["n_sub_windows"]),
    )


def extract_challenger_features(vmlap: np.ndarray) -> tuple[dict, int]:
    rows: list[dict] = []
    starts20 = list(range(0, max(1, len(vmlap) - WIN20 + 1), WIN20 // 2))
    if not starts20:
        starts20 = [0]
    for start20 in starts20:
        segment = vmlap[start20 : start20 + WIN20]
        if len(segment) < WIN20 // 2:
            continue
        for start10 in range(
            0, max(1, len(segment) - SUBWIN10 + 1), STEP2
        ):
            window = segment[start10 : start10 + SUBWIN10]
            if len(window) < int(0.8 * SUBWIN10):
                continue
            feature = window_features(window)
            rows.append(
                {
                    "v_acf_step_stride_symmetry": feature.get(
                        "v_acf_step_stride_symmetry",
                        feature.get("v_harmonic_ratio", np.nan),
                    ),
                    "ap_spec_entropy": feature.get(
                        "ap_spec_entropy", np.nan
                    ),
                    "v_jerk_rms": feature.get("v_jerk_rms", np.nan),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) < 2:
        raise ValueError("Fewer than two valid 10-second subwindows")

    def iqr(column: str) -> float:
        values = frame[column].dropna()
        if len(values) < 2:
            return np.nan
        return float(values.quantile(0.75) - values.quantile(0.25))

    return (
        {
            "v_acf_step_stride_symmetry_iqr": iqr(
                "v_acf_step_stride_symmetry"
            ),
            "ap_spec_entropy_iqr": iqr("ap_spec_entropy"),
            "v_jerk_rms_iqr": iqr("v_jerk_rms"),
        },
        len(frame),
    )


def classification_metrics(table: pd.DataFrame, probability: str) -> dict:
    y = table["target"].to_numpy(int)
    p = table[probability].to_numpy(float)
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "accuracy": float((tp + tn) / len(y)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    challenger, training = train_challenger()
    deployed_artifact = joblib.load(DEPLOYED_MODEL)
    correction = deployed_artifact.get("signal_correction", {})
    alpha = float(correction.get("alpha", 1.0))
    tau = float(correction.get("tau", 1.0))

    rows = []
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        try:
            deployed = predict_daily_gait_csv(path, MODEL_DIR)
            vmlap, duration, *_ = _csv_to_vmlap(path)
            corrected = transform_signal(vmlap, alpha, tau)
            padded, pad_samples = pad_near_20_seconds(corrected)
            original_challenger_features, original_n_subwindows = (
                extract_challenger_features(corrected)
            )
            challenger_features, n_subwindows = extract_challenger_features(
                padded
            )
            original_vector = np.asarray(
                [
                    original_challenger_features[name]
                    for name in CHALLENGER_FEATURES
                ],
                dtype=float,
            )
            vector = np.asarray(
                [
                    challenger_features[name]
                    for name in CHALLENGER_FEATURES
                ],
                dtype=float,
            )
            original_challenger_probability = float(
                challenger.predict_proba(
                    original_vector.reshape(1, -1)
                )[0, 1]
            )
            challenger_probability = float(
                challenger.predict_proba(vector.reshape(1, -1))[0, 1]
            )
            (
                current_padded_probability,
                current_padded_features,
                current_padded_n_subwindows,
            ) = predict_current_from_vmlap(padded, deployed_artifact)
            row = {
                "file": path.name,
                "target": int("발다침" in path.name),
                "duration_sec": float(duration),
                "resampled_samples_before_padding": int(len(corrected)),
                "pad_samples": int(pad_samples),
                "original_n_subwindows": int(original_n_subwindows),
                "n_subwindows": int(n_subwindows),
                "deployed_probability": float(deployed["probability"]),
                "deployed_prediction": int(deployed["prediction"]),
                "current_padded_probability": current_padded_probability,
                "current_padded_prediction": int(
                    current_padded_probability >= THRESHOLD
                ),
                "current_padded_n_subwindows": current_padded_n_subwindows,
                "challenger_original_probability": (
                    original_challenger_probability
                ),
                "challenger_original_prediction": int(
                    original_challenger_probability >= THRESHOLD
                ),
                "challenger_probability": challenger_probability,
                "challenger_prediction": int(
                    challenger_probability >= THRESHOLD
                ),
                **{
                    f"current_padded_{name}": float(value)
                    for name, value in current_padded_features.items()
                    if name
                    in {
                        "v_jerk_rms_median",
                        "v_jerk_rms_iqr",
                        "v_acf_step_stride_symmetry_iqr",
                    }
                },
                **challenger_features,
            }
            rows.append(row)
            print(
                f"{path.name}: target={row['target']} "
                f"deployed={row['deployed_probability']:.3f}->"
                f"{current_padded_probability:.3f} "
                f"challenger={original_challenger_probability:.3f}->"
                f"{challenger_probability:.3f} "
                f"windows={original_n_subwindows}->{n_subwindows}"
            )
        except Exception as exc:
            rows.append({"file": path.name, "error": repr(exc)})
            print(f"{path.name}: ERROR {exc}")

    result = pd.DataFrame(rows)
    valid = result.dropna(
        subset=["target", "deployed_probability", "challenger_probability"]
    ).copy()
    result.to_csv(
        OUT_DIR / "sample_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "status": "pilot_smoke_test_not_model_selection_evidence",
        "label_rule": "filename contains '발다침' = impaired; all others = normal",
        "n_samples": int(len(valid)),
        "n_impaired": int(valid["target"].sum()),
        "n_normal": int((valid["target"] == 0).sum()),
        "signal_correction": {"alpha": alpha, "tau": tau},
        "challenger_features": CHALLENGER_FEATURES,
        "training_subjects": int(len(training)),
        "threshold": THRESHOLD,
        "deployed_original": classification_metrics(
            valid, "deployed_probability"
        ),
        "deployed_padded": classification_metrics(
            valid, "current_padded_probability"
        ),
        "challenger_original": classification_metrics(
            valid, "challenger_original_probability"
        ),
        "challenger_padded": classification_metrics(
            valid, "challenger_probability"
        ),
        "warning": (
            "Only two impaired files are available and repeated files may "
            "come from the same person. Metrics are descriptive only."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "pipeline": challenger,
            "features": CHALLENGER_FEATURES,
            "threshold": THRESHOLD,
            "signal_correction": {"alpha": alpha, "tau": tau},
            "status": "experimental_challenger_not_deployed",
        },
        OUT_DIR / "challenger_experimental.joblib",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Written: {OUT_DIR}")


if __name__ == "__main__":
    main()
