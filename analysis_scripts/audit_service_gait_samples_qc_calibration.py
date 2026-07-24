"""Audit real app gait samples for QC, turn burden, and prediction stability."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "giukhaji"))

from modeling.gait_axis_aligned_core import (
    TARGET_FS_HZ,
    transform_signal,
    window_features,
)
from modeling.gait_axis_aligned_processor import (
    _csv_to_vmlap,
    predict_daily_gait_csv,
)


SAMPLE_DIR = ROOT / "보행SAMPLE"
MODEL_DIR = ROOT / "giukhaji" / "models"
MODEL_PATH = MODEL_DIR / "gait_daily_clinical_3feat.joblib"
TRAIN_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
OUT_DIR = ROOT / "analysis_outputs" / "service_sample_qc_audit"
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
TURN_RATE_DPS = 56.0
TURN_ANGLE_DEG = 23.0


def read_sensor_csv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = re.match(r"#\s*([^:]+):\s*(.*)", line.strip())
            if match:
                metadata[match.group(1).strip()] = match.group(2).strip()
    frame = pd.read_csv(path, comment="#")
    return frame, metadata


def parse_vector(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    values = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", text)
    if len(values) < 3:
        return None
    vector = np.asarray([float(value) for value in values[:3]], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def turn_metrics(
    frame: pd.DataFrame, metadata: dict[str, str], fs: float
) -> dict:
    gyro_columns = ["Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z"]
    vertical = parse_vector(metadata.get("Basis_Vertical_Unit"))
    if vertical is None or not set(gyro_columns).issubset(frame.columns):
        return {
            "yaw_abs_p95_dps": np.nan,
            "yaw_abs_max_dps": np.nan,
            "turn_fraction": np.nan,
            "turn_count": np.nan,
            "turn_total_angle_deg": np.nan,
        }
    gyro = frame[gyro_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    yaw_dps = gyro @ vertical * 57.295779513
    finite = np.isfinite(yaw_dps)
    if not finite.any():
        return {
            "yaw_abs_p95_dps": np.nan,
            "yaw_abs_max_dps": np.nan,
            "turn_fraction": np.nan,
            "turn_count": np.nan,
            "turn_total_angle_deg": np.nan,
        }
    yaw_dps = np.where(finite, yaw_dps, 0.0)
    candidate = np.abs(yaw_dps) >= TURN_RATE_DPS
    accepted = np.zeros(len(yaw_dps), dtype=bool)
    angles = []
    for start, end in contiguous_true_runs(candidate):
        angle = float(np.sum(np.abs(yaw_dps[start:end])) / fs)
        if angle >= TURN_ANGLE_DEG:
            accepted[start:end] = True
            angles.append(angle)
    return {
        "yaw_abs_p95_dps": float(np.percentile(np.abs(yaw_dps), 95)),
        "yaw_abs_max_dps": float(np.max(np.abs(yaw_dps))),
        "turn_fraction": float(np.mean(accepted)),
        "turn_count": int(len(angles)),
        "turn_total_angle_deg": float(np.sum(angles)),
    }


def segment_feature_rows(corrected_vmlap: np.ndarray) -> pd.DataFrame:
    window = int(10 * TARGET_FS_HZ)
    step = int(2 * TARGET_FS_HZ)
    rows = []
    for start in range(0, len(corrected_vmlap) - window + 1, step):
        try:
            feature = window_features(corrected_vmlap[start : start + window])
        except Exception:
            continue
        rows.append(
            {
                "start_sec": start / TARGET_FS_HZ,
                "v_jerk_rms": float(feature.get("v_jerk_rms", np.nan)),
                "v_harmonic_ratio": float(
                    feature.get("v_harmonic_ratio", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_final3(rows: pd.DataFrame) -> np.ndarray:
    jerk = rows["v_jerk_rms"].dropna().to_numpy(float)
    harmonic_ratio = rows["v_harmonic_ratio"].dropna().to_numpy(float)
    return np.asarray(
        [
            np.median(jerk) if len(jerk) else np.nan,
            np.percentile(jerk, 75) - np.percentile(jerk, 25)
            if len(jerk) >= 2
            else np.nan,
            np.percentile(harmonic_ratio, 75)
            - np.percentile(harmonic_ratio, 25)
            if len(harmonic_ratio) >= 2
            else np.nan,
        ],
        dtype=float,
    )


def jackknife_probabilities(
    rows: pd.DataFrame, artifact: dict
) -> np.ndarray:
    probabilities = []
    if len(rows) < 3:
        return np.asarray(probabilities, dtype=float)
    for omitted in range(len(rows)):
        values = aggregate_final3(rows.drop(rows.index[omitted]))
        if not np.isfinite(values).all():
            continue
        probabilities.append(
            float(artifact["pipeline"].predict_proba(values.reshape(1, -1))[0, 1])
        )
    return np.asarray(probabilities, dtype=float)


def training_reference() -> tuple[pd.Series, pd.Series]:
    training = pd.read_csv(TRAIN_TABLE)
    median = training[FEATURES].median()
    robust_sigma = (training[FEATURES].quantile(0.75) - training[FEATURES].quantile(0.25)) / 1.349
    robust_sigma = robust_sigma.replace(0, np.nan)
    return median, robust_sigma


def exact_binomial_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [np.nan, np.nan]
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(0.975, successes + 1, total - successes))
    )
    return [lower, upper]


def labeled_metrics(
    table: pd.DataFrame, probability_column: str, threshold: float
) -> dict:
    y = table["target"].to_numpy(int)
    prediction = (
        pd.to_numeric(table[probability_column], errors="coerce")
        .to_numpy(float)
        >= threshold
    ).astype(int)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    accuracy = (tp + tn) / len(y) if len(y) else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else np.nan
    )
    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "sensitivity_exact_95ci": exact_binomial_interval(tp, tp + fn),
        "specificity": specificity,
        "specificity_exact_95ci": exact_binomial_interval(tn, tn + fp),
        "accuracy": accuracy,
        "accuracy_exact_95ci": exact_binomial_interval(tp + tn, len(y)),
        "precision": precision,
        "f1": f1,
    }


def audit_sample(
    path: Path,
    artifact: dict,
    train_median: pd.Series,
    train_sigma: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    frame, metadata = read_sensor_csv(path)
    timestamps = pd.to_numeric(frame.get("Timestamp_ns"), errors="coerce")
    valid_timestamps = timestamps.dropna().to_numpy(float)
    duration = (
        float((valid_timestamps[-1] - valid_timestamps[0]) / 1e9)
        if len(valid_timestamps) >= 2
        else np.nan
    )
    dt = np.diff(valid_timestamps) / 1e9 if len(valid_timestamps) >= 2 else np.array([])
    positive_dt = dt[dt > 0]
    median_fs = (
        float(1.0 / np.median(positive_dt)) if len(positive_dt) else np.nan
    )
    elapsed_fs = (
        float((len(valid_timestamps) - 1) / duration)
        if np.isfinite(duration) and duration > 0
        else np.nan
    )
    duplicate_or_reverse_fraction = (
        float(np.mean(dt <= 0)) if len(dt) else np.nan
    )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    missing_fraction = float(numeric.isna().mean().mean())

    prediction = predict_daily_gait_csv(str(path), MODEL_DIR)
    vmlap, _, _, _, _, _ = _csv_to_vmlap(str(path))
    correction = artifact.get("signal_correction", {})
    corrected = transform_signal(
        vmlap,
        float(correction.get("alpha", 1.0)),
        float(correction.get("tau", 1.0)),
    )
    uncorrected_rows = segment_feature_rows(vmlap)
    uncorrected_vector = aggregate_final3(uncorrected_rows)
    uncorrected_probability = (
        float(
            artifact["pipeline"].predict_proba(
                uncorrected_vector.reshape(1, -1)
            )[0, 1]
        )
        if np.isfinite(uncorrected_vector).all()
        else np.nan
    )
    uncorrected_series = pd.Series(
        uncorrected_vector, index=FEATURES, dtype=float
    )
    uncorrected_robust_z = (
        (uncorrected_series - train_median) / train_sigma
    ).abs()
    window_rows = segment_feature_rows(corrected)
    jackknife = jackknife_probabilities(window_rows, artifact)
    threshold = float(artifact.get("threshold", 0.5))
    feature_vector = pd.Series(prediction["features"])[FEATURES].astype(float)
    robust_z = ((feature_vector - train_median) / train_sigma).abs()
    turns = turn_metrics(
        frame,
        metadata,
        median_fs if np.isfinite(median_fs) else elapsed_fs,
    )
    audit = {
        "file": path.name,
        "rows": int(len(frame)),
        "duration_sec": duration,
        "median_fs_hz": median_fs,
        "elapsed_fs_hz": elapsed_fs,
        "missing_fraction": missing_fraction,
        "duplicate_or_reverse_timestamp_fraction": duplicate_or_reverse_fraction,
        "n_subwindows": int(prediction["window"].get("n_sub_windows", 0)),
        "probability": float(prediction["probability"]),
        "threshold": threshold,
        "prediction": int(prediction["prediction"]),
        **{feature: float(feature_vector[feature]) for feature in FEATURES},
        "uncorrected_probability": uncorrected_probability,
        **{
            f"uncorrected_{feature}": float(uncorrected_series[feature])
            for feature in FEATURES
        },
        "max_abs_training_robust_z": float(robust_z.max()),
        "ood_feature": str(robust_z.idxmax()),
        "uncorrected_max_abs_training_robust_z": float(
            uncorrected_robust_z.max()
        ),
        "uncorrected_ood_feature": str(uncorrected_robust_z.idxmax()),
        "jackknife_n": int(len(jackknife)),
        "jackknife_probability_min": float(np.min(jackknife))
        if len(jackknife)
        else np.nan,
        "jackknife_probability_max": float(np.max(jackknife))
        if len(jackknife)
        else np.nan,
        "jackknife_probability_sd": float(np.std(jackknife))
        if len(jackknife)
        else np.nan,
        "jackknife_crosses_threshold": bool(
            len(jackknife)
            and np.min(jackknife) < threshold <= np.max(jackknife)
        ),
        **turns,
    }
    window_rows = window_rows.assign(file=path.name)
    return audit, window_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(MODEL_PATH)
    train_median, train_sigma = training_reference()
    audits = []
    windows = []
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        try:
            audit, window_rows = audit_sample(
                path, artifact, train_median, train_sigma
            )
            audits.append(audit)
            windows.append(window_rows)
            print(
                f"{path.name}: p={audit['probability']:.3f} "
                f"windows={audit['n_subwindows']} "
                f"turn={audit['turn_fraction']:.1%} "
                f"jackknife={audit['jackknife_probability_min']:.3f}-"
                f"{audit['jackknife_probability_max']:.3f}"
            )
        except Exception as exc:
            audits.append({"file": path.name, "error": repr(exc)})
            print(f"{path.name}: ERROR {exc}")

    audit_table = pd.DataFrame(audits)
    audit_table["target"] = (
        audit_table["file"].astype(str).str.contains("발다침").astype(int)
    )
    audit_table.to_csv(
        OUT_DIR / "sample_qc_audit.csv", index=False, encoding="utf-8-sig"
    )
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(
            OUT_DIR / "sample_subwindow_features.csv",
            index=False,
            encoding="utf-8-sig",
        )
    threshold = float(artifact.get("threshold", 0.5))
    corrected_metrics = labeled_metrics(
        audit_table, "probability", threshold
    )
    uncorrected_metrics = labeled_metrics(
        audit_table, "uncorrected_probability", threshold
    )
    stable = audit_table[
        ~audit_table["jackknife_crosses_threshold"].fillna(False).astype(bool)
    ]
    stable_metrics = labeled_metrics(stable, "probability", threshold)
    labeled_pilot = {
        "label_rule": "filename contains '발다침' = impaired; all others = normal",
        "n_files": int(len(audit_table)),
        "n_impaired": int(audit_table["target"].sum()),
        "n_normal": int((audit_table["target"] == 0).sum()),
        "corrected": corrected_metrics,
        "uncorrected": uncorrected_metrics,
        "jackknife_stable_only": {
            "coverage": float(len(stable) / len(audit_table)),
            "n_files": int(len(stable)),
            **stable_metrics,
        },
        "warning": (
            "Pilot only: two impaired files and eight normal files; "
            "independence between repeated files is not established."
        ),
    }
    (OUT_DIR / "labeled_pilot_metrics.json").write_text(
        json.dumps(labeled_pilot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "n_files": int(len(audit_table)),
        "n_errors": int(audit_table.get("error", pd.Series(dtype=object)).notna().sum()),
        "n_with_6_windows": int(audit_table.get("n_subwindows", pd.Series(dtype=float)).eq(6).sum()),
        "n_jackknife_cross_threshold": int(
            audit_table.get(
                "jackknife_crosses_threshold", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "n_robust_z_gt_3p5": int(
            audit_table.get(
                "max_abs_training_robust_z", pd.Series(dtype=float)
            ).gt(3.5).sum()
        ),
        "n_uncorrected_robust_z_gt_3p5": int(
            audit_table.get(
                "uncorrected_max_abs_training_robust_z",
                pd.Series(dtype=float),
            ).gt(3.5).sum()
        ),
        "turn_detection": {
            "yaw_rate_threshold_dps": TURN_RATE_DPS,
            "minimum_integrated_angle_deg": TURN_ANGLE_DEG,
            "note": "Literature starting point; device-specific validation still required.",
        },
        "clinical_threshold_validation_possible": "pilot_only",
        "reason": (
            "User confirmed only filenames containing '발다침' are impaired; "
            "the pilot has only 2 impaired and 8 normal files."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Written: {OUT_DIR}")


if __name__ == "__main__":
    main()
