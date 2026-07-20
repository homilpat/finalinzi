from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend


ROOT = Path(__file__).resolve().parents[1]
GAIT_CODE = Path.home() / "Desktop" / "파이널 보행 프로젝트" / "75h_processing_butterworth"
if str(GAIT_CODE) not in sys.path:
    sys.path.insert(0, str(GAIT_CODE))

from extract_single_window_spectrogram_features import spec_features  # noqa: E402
from run_strict_preprocessing_from_physionet import (  # noqa: E402
    DEFAULT_DATA_DIR,
    butterworth_bandpass,
    butterworth_lowpass,
    extract_segment_features,
    load_physical_record,
    unbiased_acf,
)


OUT_DIR = ROOT / "analysis_outputs" / "physionet_labwalks_smartphone_shape_extractor_all_or"
WINDOW_SEC = 10.0
STRIDE_SEC = 2.5

PARAMS = {
    "butter_cutoff_hz": 20.0,
    "butter_order": 4,
    "gyro_mode": "v2",
    "gyro_bandpass_low_hz": 0.5,
    "gyro_bandpass_high_hz": 5.0,
    "gyro_bandpass_order": 4,
    "gyro_freq_min_hz": 0.6,
    "gyro_freq_max_hz": 3.0,
    "turn_yaw_rate_threshold_dps": 30.0,
    "filter_turn_segments": False,
    "step_min_sec": 0.35,
    "step_max_sec": 0.80,
    "stride_min_sec": 0.80,
    "stride_max_sec": 1.60,
    "stride_ratio_min": 1.55,
    "stride_ratio_max": 2.45,
    "min_ap_step_reg": 0.30,
    "min_ap_stride_reg": 0.30,
    "min_v_step_reg": 0.25,
    "min_acf_prominence": 0.05,
    "freq_min_hz": 0.6,
    "freq_max_hz": 3.0,
    "min_ap_peak_power_ratio": 0.08,
}


def lab_records(data_dir: Path) -> list[str]:
    records_file = data_dir / "RECORDS"
    return [
        line.strip()
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("LabWalks/")
    ]


def subject_for_record(record: str) -> str:
    return Path(record).name.split("_")[0].upper()


def load_clinical_labels() -> pd.DataFrame:
    clinical_paths = [
        p
        for p in (Path.home() / "Desktop").rglob("ClinicalDemogData_COFL.xlsx")
        if "physionet_AWS" in str(p)
    ]
    if not clinical_paths:
        raise FileNotFoundError("ClinicalDemogData_COFL.xlsx not found")

    frames = []
    for sheet in ("Controls", "Fallers"):
        df = pd.read_excel(clinical_paths[0], sheet_name=sheet).rename(columns={"#": "raw_subject_id"})
        df["subject_id"] = (
            df["raw_subject_id"].astype(str).str.upper().str.replace("-", "", regex=False).str.strip()
        )
        frames.append(df)

    clinical = pd.concat(frames, ignore_index=True)
    motor_cols = ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)"]
    for col in motor_cols:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")

    labels = clinical[["subject_id", *motor_cols]].copy()
    specs = {
        "TUG_ge_12": ("TUG", labels["TUG"] >= 12),
        "FSST_ge_15": ("FSST", labels["FSST"] >= 15),
        "BERG_lt_52": ("BERG", labels["BERG"] < 52),
        "DGI_le_19": ("DGI", labels["DGI"] <= 19),
        "velocity_lt_1p0": ("base(velocity)", labels["base(velocity)"] < 1.0),
    }
    for name, (raw_col, condition) in specs.items():
        labels[name] = np.where(labels[raw_col].notna(), condition.astype(int), np.nan)

    label_cols = list(specs)
    labels["motor_any_or_available"] = np.where(
        labels[label_cols].notna().any(axis=1),
        labels[label_cols].fillna(0).astype(int).any(axis=1).astype(int),
        np.nan,
    )
    labels["DGI_le19_or_TUG_ge12"] = np.where(
        labels[["DGI", "TUG"]].notna().all(axis=1),
        ((labels["DGI"] <= 19) | (labels["TUG"] >= 12)).astype(int),
        np.nan,
    )
    return labels


def acf_peak_features(signal: np.ndarray, fs: float, stride_duration: float | None, prefix: str) -> dict:
    acf = unbiased_acf(signal)
    out = {
        f"{prefix}_acf_decay_below_0p5_sec": np.nan,
        f"{prefix}_acf_stride_peak_width_sec": np.nan,
    }
    below = np.flatnonzero(acf <= 0.5)
    if len(below):
        out[f"{prefix}_acf_decay_below_0p5_sec"] = float(below[0] / fs)
    if stride_duration and np.isfinite(stride_duration):
        center = int(round(stride_duration * fs))
        lo = max(1, center - int(round(0.25 * fs)))
        hi = min(len(acf) - 1, center + int(round(0.25 * fs)))
        if hi > lo:
            seg = acf[lo : hi + 1]
            peak = lo + int(np.nanargmax(seg))
            half = max(0.0, float(acf[peak]) * 0.5)
            left = peak
            while left > 1 and acf[left] >= half:
                left -= 1
            right = peak
            while right < len(acf) - 1 and acf[right] >= half:
                right += 1
            out[f"{prefix}_acf_stride_peak_width_sec"] = float((right - left) / fs)
    return out


def stride_shape_features(signal: np.ndarray, fs: float, stride_duration: float | None, prefix: str) -> dict:
    out = {
        f"{prefix}_stride_shape_cv_mean": np.nan,
        f"{prefix}_stride_shape_sd_mean": np.nan,
        f"{prefix}_stride_shape_corr_mean": np.nan,
        f"{prefix}_stride_shape_corr_sd": np.nan,
        f"{prefix}_peak_timing_sd_pct": np.nan,
    }
    if not stride_duration or not np.isfinite(stride_duration):
        return out

    stride_n = int(round(stride_duration * fs))
    if stride_n < int(0.8 * fs) or stride_n > int(1.6 * fs):
        return out

    waves = []
    peak_times = []
    grid = np.linspace(0, stride_n - 1, 100)
    starts = range(0, len(signal) - stride_n + 1, max(1, stride_n))
    for start in starts:
        seg = np.asarray(signal[start : start + stride_n], dtype=float)
        if len(seg) != stride_n or not np.isfinite(seg).all():
            continue
        sd = float(np.std(seg))
        if sd <= 1e-10:
            continue
        z = (seg - np.mean(seg)) / sd
        waves.append(np.interp(grid, np.arange(stride_n), z))
        peak_times.append(float(np.argmax(z) / max(1, stride_n - 1) * 100.0))

    if len(waves) < 3:
        return out

    arr = np.vstack(waves)
    mean_wave = np.nanmean(arr, axis=0)
    sd_wave = np.nanstd(arr, axis=0)
    out[f"{prefix}_stride_shape_sd_mean"] = float(np.nanmean(sd_wave))
    out[f"{prefix}_stride_shape_cv_mean"] = float(
        np.nanmean(sd_wave / (np.abs(mean_wave) + 1e-3))
    )

    corrs = []
    for wave in arr:
        if np.std(wave) > 1e-10 and np.std(mean_wave) > 1e-10:
            corrs.append(float(np.corrcoef(wave, mean_wave)[0, 1]))
    if corrs:
        out[f"{prefix}_stride_shape_corr_mean"] = float(np.nanmean(corrs))
        out[f"{prefix}_stride_shape_corr_sd"] = float(np.nanstd(corrs))
    out[f"{prefix}_peak_timing_sd_pct"] = float(np.nanstd(peak_times))
    return out


def extract_shape_features(acc: np.ndarray, gyro: np.ndarray, fs: float) -> tuple[dict, float]:
    base, reason = extract_segment_features(acc, fs, PARAMS, gyro)
    stride_duration = None
    if base is not None and np.isfinite(base.get("stride_duration", np.nan)):
        stride_duration = float(base["stride_duration"])

    v = butterworth_bandpass(acc[:, 0], fs, low=0.6, high=3.0, order=4)
    ml = butterworth_bandpass(acc[:, 1], fs, low=0.6, high=3.0, order=4)
    ap = butterworth_bandpass(acc[:, 2], fs, low=0.6, high=3.0, order=4)

    out: dict[str, float | str] = {}
    if base is None:
        for key in [
            "step_duration",
            "stride_duration",
            "v_stride_regularity",
            "ml_stride_regularity",
            "ap_stride_regularity",
        ]:
            out[key if key.endswith("duration") else key] = np.nan
    else:
        out["step_time_median"] = base.get("step_duration", np.nan)
        out["stride_time_median"] = base.get("stride_duration", np.nan)
        out["cadence"] = 60.0 / base["step_duration"] if base.get("step_duration", 0) else np.nan
        out["v_stride_regularity"] = base.get("v_stride_regularity", np.nan)
        out["ml_stride_regularity"] = base.get("ml_stride_regularity", np.nan)
        out["ap_stride_regularity"] = base.get("ap_stride_regularity", np.nan)
        out["v_step_peak"] = base.get("v_step_regularity", np.nan)
        out["v_stride_peak"] = base.get("v_stride_regularity", np.nan)
        out["ml_acf_stride_peak"] = base.get("ml_stride_regularity", np.nan)
        out["ap_acf_stride_peak"] = base.get("ap_stride_regularity", np.nan)

    for prefix, sig in (("v", v), ("ml", ml), ("ap", ap)):
        out.update(acf_peak_features(sig, fs, stride_duration, prefix))
        out.update(stride_shape_features(sig, fs, stride_duration, prefix))

    out.update(spec_features(v, fs, "v", band_low=0.6, band_high=3.0))
    out.update(spec_features(ml, fs, "ml", band_low=0.6, band_high=3.0))
    out["base_feature_status"] = "ok" if reason == "ok" else f"base_failed_{reason}"

    quality_parts = [
        out.get("v_stride_regularity", np.nan),
        out.get("ap_stride_regularity", np.nan),
        out.get("v_stride_shape_corr_mean", np.nan),
    ]
    quality = float(np.nanmean(quality_parts)) if np.isfinite(quality_parts).any() else -np.inf
    return out, quality


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_clinical_labels()

    rows = []
    rejects = []
    for record in lab_records(DEFAULT_DATA_DIR):
        data, fs, _ = load_physical_record(DEFAULT_DATA_DIR / record, channels=(0, 1, 2, 3, 4, 5))
        data = butterworth_lowpass(data, fs, cutoff=20.0, order=4)
        acc = data[:, :3]
        gyro = data[:, 3:6]
        win = int(round(WINDOW_SEC * fs))
        step = int(round(STRIDE_SEC * fs))
        if len(acc) < win:
            continue
        subject_id = subject_for_record(record)
        for idx, start in enumerate(range(0, len(acc) - win + 1, step)):
            end = start + win
            try:
                feats, quality = extract_shape_features(acc[start:end], gyro[start:end], fs)
            except Exception as exc:  # keep batch extraction robust
                rejects.append({"record": record, "segment_idx": idx, "reason": repr(exc)})
                continue
            rows.append(
                {
                    "dataset": "PhysioNet_LabWalks",
                    "source_id": record,
                    "subject_id": subject_id,
                    "segment_idx": idx,
                    "chunk_start_sec": start / fs,
                    "start_sec": start / fs,
                    "end_sec": end / fs,
                    "window_sec": WINDOW_SEC,
                    "quality_score": quality,
                    **feats,
                }
            )

    windows = pd.DataFrame(rows)
    windows = windows.merge(labels, on="subject_id", how="left")
    windows["target"] = windows["motor_any_or_available"]
    windows["label_group"] = np.where(windows["target"].astype(float) == 1, "impaired", "normal")
    windows.to_csv(OUT_DIR / "physionet_labwalks_shape_windows_all_or.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejects).to_csv(OUT_DIR / "physionet_labwalks_shape_rejects.csv", index=False, encoding="utf-8-sig")

    best = (
        windows.sort_values(["subject_id", "quality_score"], ascending=[True, False])
        .groupby("subject_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(OUT_DIR / "physionet_labwalks_shape_best10_all_or.csv", index=False, encoding="utf-8-sig")

    counts = best["target"].dropna().astype(int).value_counts().sort_index()
    print("windows", windows.shape, "best", best.shape, "rejects", len(rejects))
    print("all OR best10 label counts", counts.to_dict())
    print(best[["subject_id", "target", "quality_score", "start_sec", "v_stride_shape_cv_mean", "ml_stride_shape_cv_mean", "ap_stride_regularity", "v_stride_regularity"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
