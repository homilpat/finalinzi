from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

from gait_axis_aligned_core import (  # noqa: E402
    TARGET_FS_HZ,
    align_to_vmlap,
    bandpass,
    extract_axis_aligned_gait_features,
    load_sensor_csv_with_metadata,
    resample_array_to_100hz,
)


OUT_DIR = ROOT / "analysis_outputs" / "waist_sensor_range_loss_calibration"
PHYSIONET = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir()) / "physionet_AWS" / "LabWalks"
SAMPLE_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and "SAMPLE" in p.name)


RAW_STATS = [
    "v_bp_rms",
    "ml_bp_rms",
    "ap_bp_rms",
    "v_bp_p95",
    "ml_bp_p95",
    "ap_bp_p95",
]
FEATURE_STATS = [
    "v_acf_stride_peak",
    "v_acf_stride_peak_width_sec",
    "ap_acf_stride_peak_width_sec",
    "ap_spec_entropy",
]


def _read_physionet(stem: str) -> tuple[np.ndarray, float]:
    header = (PHYSIONET / f"{stem}.hea").read_text(encoding="utf-8").splitlines()
    first = header[0].split()
    n_sig = int(first[1])
    fs = float(first[2])
    n = int(first[3])
    gains, baselines = [], []
    for line in header[1 : 1 + n_sig]:
        match = re.match(r"([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2)))
    raw = np.memmap(PHYSIONET / f"{stem}.dat", dtype="<i2", mode="r", shape=(n, n_sig))
    data = (raw.astype(float) - np.array(baselines)) / np.array(gains)
    return data[:, :3], fs


def _best_vmlap(acc: np.ndarray, fs: float, already_vmlap: bool) -> np.ndarray:
    aligned, _ = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=fs)
    resampled = resample_array_to_100hz(aligned, fs)
    win = int(round(10.0 * TARGET_FS_HZ))
    if len(resampled) < win:
        resampled = np.pad(resampled, ((0, win - len(resampled)), (0, 0)), mode="edge")
    best, best_score = None, -np.inf
    for start in range(0, len(resampled) - win + 1, max(1, int(round(2 * TARGET_FS_HZ)))):
        seg = resampled[start : start + win]
        v = bandpass(seg[:, 0], TARGET_FS_HZ)
        score = float(np.nanstd(v))
        if score > best_score:
            best, best_score = seg, score
    if best is None:
        raise ValueError("no valid window")
    return best


def _raw_stats(vmlap: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, axis in enumerate(("v", "ml", "ap")):
        sig = bandpass(vmlap[:, idx], TARGET_FS_HZ)
        out[f"{axis}_bp_rms"] = float(np.sqrt(np.nanmean(sig * sig)))
        out[f"{axis}_bp_p95"] = float(np.nanpercentile(np.abs(sig), 95))
    return out


def _sample_vmlap(path: Path) -> tuple[np.ndarray, dict]:
    df, metadata = load_sensor_csv_with_metadata(str(path))
    if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(df.columns):
        acc = df[["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]].to_numpy(float)
        already = True
    else:
        acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float)
        if np.nanmedian(np.linalg.norm(acc, axis=1)) > 3.0:
            acc = acc / 9.80665
        already = False
    t = df["Timestamp_ns"].to_numpy(float)
    duration = (np.nanmax(t) - np.nanmin(t)) / 1e9
    fs = float(len(df) / duration) if np.isfinite(duration) and duration > 0 else TARGET_FS_HZ
    return _best_vmlap(acc, fs, already), metadata


def _robust_reference(rows: list[dict], keys: list[str]) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame(rows)
    med = frame[keys].median(numeric_only=True)
    iqr = frame[keys].quantile(0.75, numeric_only=True) - frame[keys].quantile(0.25, numeric_only=True)
    return med, iqr.replace(0, np.nan).fillna(frame[keys].std(numeric_only=True).replace(0, np.nan)).fillna(1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    physio_rows = []
    for hea in sorted(PHYSIONET.glob("co*_base.hea")):
        stem = hea.stem.removesuffix("_base")
        full_stem = f"{stem}_base"
        try:
            acc, fs = _read_physionet(full_stem)
            vmlap = _best_vmlap(acc, fs, already_vmlap=True)
            physio_rows.append({"subject_id": full_stem, **_raw_stats(vmlap)})
        except Exception:
            continue
    ref_med, ref_scale = _robust_reference(physio_rows, RAW_STATS)

    sample_rows, sample_windows = [], []
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        if "발다침" in path.stem:
            continue
        vmlap, metadata = _sample_vmlap(path)
        features = extract_axis_aligned_gait_features(str(path))["features"]
        row = {"sample_id": path.stem, **_raw_stats(vmlap), **features}
        sample_rows.append(row)
        sample_windows.append((path.stem, vmlap, metadata))

    def loss(log_scale: np.ndarray) -> float:
        scale = np.exp(log_scale)
        losses = []
        for _, vmlap, metadata in sample_windows:
            scaled = vmlap * scale.reshape(1, 3)
            stats = _raw_stats(scaled)
            for key in RAW_STATS:
                z = (stats[key] - ref_med[key]) / ref_scale[key]
                losses.append(np.sqrt(z * z + 1.0) - 1.0)
            accel_range = metadata.get("accel_range_m_s2")
            if isinstance(accel_range, (int, float, np.floating)):
                max_g = float(accel_range) / 9.80665
                ratio = float(np.nanmax(np.abs(scaled)) / max_g)
                losses.append(max(0.0, ratio - 0.95) ** 2 * 10.0)
        # Keep the calibration conservative; this is not paired IMU-phone calibration.
        losses.extend((log_scale / 0.25) ** 2)
        return float(np.nanmean(losses))

    result = minimize(loss, np.zeros(3), method="Nelder-Mead", options={"maxiter": 500})
    axis_scale = np.exp(result.x)

    calibrated_rows = []
    for sample_id, vmlap, _ in sample_windows:
        calibrated_rows.append({"sample_id": sample_id, **_raw_stats(vmlap * axis_scale.reshape(1, 3))})

    summary = {
        "method": "waist_normal_raw_range_loss_calibration",
        "reference": "PhysioNet LabWalks normal lower-back/L5 IMU raw best-10-second windows",
        "calibration_samples": [row["sample_id"] for row in sample_rows],
        "axis_scale_v_ml_ap": {
            "vertical": float(axis_scale[0]),
            "mediolateral": float(axis_scale[1]),
            "anteroposterior": float(axis_scale[2]),
        },
        "loss_before": loss(np.zeros(3)),
        "loss_after": float(result.fun),
        "warning": "Estimated from two OUR_SAMPLE normal calibration files only. Do not fit on impaired/new test measurements.",
        "feature_note": "Final deployed ACF/entropy features are mostly scale-invariant, so this raw scale calibration is for sensor-range harmonization and QC, not a replacement for paired IMU-phone calibration.",
    }

    pd.DataFrame(physio_rows).to_csv(OUT_DIR / "physionet_waist_normal_raw_reference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows).to_csv(OUT_DIR / "sample_normal_raw_before.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calibrated_rows).to_csv(OUT_DIR / "sample_normal_raw_after.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "waist_sensor_range_loss_calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written", OUT_DIR)


if __name__ == "__main__":
    main()
