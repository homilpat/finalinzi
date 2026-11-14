from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from extract_single_window_amplitude_pooling_features import amplitude_pooling_features
from extract_single_window_spectrogram_features import spec_features
from run_strict_preprocessing_from_physionet import (
    DEFAULT_DATA_DIR,
    butterworth_bandpass,
    butterworth_lowpass,
    extract_segment_features,
    load_physical_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "physionet_AWS"
    / "strict_preprocessing_runs"
    / "labwalks_service20_features"
)


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
    "gyro_freq_min_hz": 0.6,
    "gyro_freq_max_hz": 3.0,
}


def lab_records(data_dir: Path) -> list[str]:
    return [
        line.strip()
        for line in (data_dir / "RECORDS").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("LabWalks/")
    ]


def subject_for_record(record: str) -> str:
    return Path(record).name.split("_")[0].upper()


def group_for_record(record: str) -> str:
    return "Control" if Path(record).name.lower().startswith("co") else "Faller"


def find_turn_exclusion(gyro: np.ndarray, fs: float, margin_sec: float) -> tuple[int, int, int]:
    yaw = detrend(gyro[:, 0], type="linear")
    win = max(1, int(round(fs)))
    rolling_abs_yaw = np.convolve(np.abs(yaw), np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(rolling_abs_yaw))
    margin = int(round(margin_sec * fs))
    return max(0, peak_idx - margin), min(len(yaw), peak_idx + margin), peak_idx


def overlaps(start: int, end: int, excl_start: int, excl_end: int) -> bool:
    return start < excl_end and end > excl_start


def extract_window_features(acc_seg: np.ndarray, gyro_seg: np.ndarray, fs: float) -> tuple[dict | None, str]:
    base_features, reason = extract_segment_features(acc_seg, fs, PARAMS, gyro_seg)
    if base_features is None:
        base_features = {}

    stride_duration = base_features.get("stride_duration", np.nan)
    if not np.isfinite(stride_duration):
        stride_duration = None
    out: dict[str, float] = {}

    v = butterworth_bandpass(acc_seg[:, 0], fs, low=0.6, high=3.0, order=4)
    ml = butterworth_bandpass(acc_seg[:, 1], fs, low=0.6, high=3.0, order=4)
    roll = gyro_seg[:, 2] - np.nanmedian(gyro_seg[:, 2])
    roll = butterworth_bandpass(roll, fs, low=0.5, high=5.0, order=4)

    out.update(amplitude_pooling_features(v, fs, "v", stride_duration))
    out.update(amplitude_pooling_features(ml, fs, "ml", stride_duration))
    out.update(amplitude_pooling_features(roll, fs, "roll", stride_duration))

    out.update(spec_features(v, fs, "v", band_low=0.6, band_high=3.0))
    out.update(spec_features(ml, fs, "ml", band_low=0.6, band_high=3.0))
    out.update(spec_features(roll, fs, "roll", band_low=0.6, band_high=5.0))

    # Keep a few established gait features for combined-feature ablation.
    for key in [
        "step_duration",
        "stride_duration",
        "v_stride_regularity",
        "ml_stride_regularity",
        "ap_stride_regularity",
        "v_amplitude",
        "ml_amplitude",
        "ap_amplitude",
        "v_harmonic_ratio",
        "ml_harmonic_ratio",
        "ap_harmonic_ratio",
    ]:
        out[f"base_{key}"] = base_features.get(key, np.nan)
    out["base_feature_status"] = "ok" if reason == "ok" else f"base_failed_{reason}"
    return out, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--turn-exclude-margin-sec", type=float, default=5.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "labwalks_service20_amp_spec_features.csv"
    reject_csv = args.out_dir / "labwalks_service20_rejects.csv"
    turn_csv = args.out_dir / "labwalks_service20_turn_windows.csv"

    rows: list[dict] = []
    rejects: list[dict] = []
    turns: list[dict] = []

    for record in lab_records(args.data_dir):
        data, fs, _ = load_physical_record(args.data_dir / record, channels=(0, 1, 2, 3, 4, 5))
        data = butterworth_lowpass(data, fs, cutoff=20.0, order=4)
        acc = data[:, :3]
        gyro = data[:, 3:6]
        excl_start, excl_end, peak_idx = find_turn_exclusion(gyro, fs, args.turn_exclude_margin_sec)
        subject_id = subject_for_record(record)
        turns.append(
            {
                "record": record,
                "subject_id": subject_id,
                "group": group_for_record(record),
                "duration_sec": len(acc) / fs,
                "turn_peak_sec": peak_idx / fs,
                "exclude_start_sec": excl_start / fs,
                "exclude_end_sec": excl_end / fs,
            }
        )

        win = int(round(args.window_sec * fs))
        stride = int(round(args.stride_sec * fs))
        if len(acc) < win:
            continue
        n_seg = (len(acc) - win) // stride + 1
        for segment_idx in range(n_seg):
            start = segment_idx * stride
            end = start + win
            meta = {
                "record": record,
                "subject_id": subject_id,
                "group": group_for_record(record),
                "segment_idx": segment_idx,
                "start_sec": start / fs,
                "end_sec": end / fs,
                "window_sec": args.window_sec,
            }
            if overlaps(start, end, excl_start, excl_end):
                rejects.append({**meta, "reject_reason": "turn_overlap"})
                continue
            features, reason = extract_window_features(acc[start:end], gyro[start:end], fs)
            if features is None:
                rejects.append({**meta, "reject_reason": reason})
                continue
            rows.append({**meta, **features})
        print(f"{subject_id} accepted_so_far={len(rows)} rejects_so_far={len(rejects)}")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(rejects).to_csv(reject_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(turns).to_csv(turn_csv, index=False, encoding="utf-8-sig")
    (args.out_dir / "labwalks_service20_metadata.json").write_text(
        json.dumps(
            {
                "window_sec": args.window_sec,
                "stride_sec": args.stride_sec,
                "turn_exclude_margin_sec": args.turn_exclude_margin_sec,
                "note": "LabWalks service-style 20s windows. 180-degree turn windows are excluded. Features include amplitude pooling, compact spectrogram, and selected base gait features.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved features: {out_csv}")
    print(f"Accepted windows: {len(rows)} subjects={pd.DataFrame(rows)['subject_id'].nunique() if rows else 0}")
    print(f"Rejected windows: {len(rejects)}")


if __name__ == "__main__":
    main()
