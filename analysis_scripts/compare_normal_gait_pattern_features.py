from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt, spectrogram, welch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "normal_gait_pattern_comparison"


def bandpass(x: np.ndarray, fs: float, low: float = 0.6, high: float = 3.0) -> np.ndarray:
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
    if len(peaks) == 0:
        idx = lo + int(np.nanargmax(seg))
    else:
        idx = lo + int(peaks[np.argmax(props["prominences"])])
    height = float(c[idx])
    half = max(0.0, height * 0.5)
    left = idx
    while left > 1 and c[left] >= half:
        left -= 1
    right = idx
    while right < len(c) - 1 and c[right] >= half:
        right += 1
    return idx / fs, height, (right - left) / fs


def spec_entropy(x: np.ndarray, fs: float) -> tuple[float, float, float]:
    freqs, _, pxx = spectrogram(
        x - np.nanmean(x),
        fs=fs,
        window="hann",
        nperseg=max(16, min(len(x), int(round(4 * fs)))),
        noverlap=None,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    mask = (freqs >= 0.6) & (freqs <= 3.0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    band = pxx[mask, :]
    mean_spec = np.nanmean(band, axis=1)
    band_freqs = freqs[mask]
    total = float(np.nansum(mean_spec))
    if total <= 1e-12:
        return np.nan, np.nan, np.nan
    peak = int(np.nanargmax(mean_spec))
    prob = band.reshape(-1)
    prob = prob / (np.nansum(prob) + 1e-12)
    ent = float(-np.nansum(prob * np.log2(prob + 1e-12)) / np.log2(len(prob))) if len(prob) > 1 else 0.0
    return float(band_freqs[peak]), float(mean_spec[peak] / total), ent


def band_power_features(x: np.ndarray, fs: float) -> dict:
    freqs, pxx = welch(
        x - np.nanmean(x),
        fs=fs,
        window="hann",
        nperseg=max(32, min(len(x), int(round(8 * fs)))),
        noverlap=None,
        detrend=False,
        scaling="density",
    )

    def power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapz(pxx[mask], freqs[mask])) if np.any(mask) else np.nan

    p06_12 = power(0.6, 1.2)
    p12_20 = power(1.2, 2.0)
    p20_30 = power(2.0, 3.0)
    total = np.nansum([p06_12, p12_20, p20_30])
    return {
        "bandpower_0p6_1p2": p06_12,
        "bandpower_1p2_2p0": p12_20,
        "bandpower_2p0_3p0": p20_30,
        "bandpower_mid_total_ratio": p12_20 / (total + 1e-12) if total > 1e-12 else np.nan,
    }


def spectral_shape_features(x: np.ndarray, fs: float) -> dict:
    freqs, pxx = welch(
        x - np.nanmean(x),
        fs=fs,
        window="hann",
        nperseg=max(32, min(len(x), int(round(8 * fs)))),
        noverlap=None,
        detrend=False,
        scaling="density",
    )
    mask = (freqs >= 0.6) & (freqs <= 3.0)
    if not np.any(mask):
        return {
            "harmonic_ratio": np.nan,
            "spectral_flatness": np.nan,
            "dominant_peak_prominence": np.nan,
            "low_high_band_power_ratio": np.nan,
        }
    bf = freqs[mask]
    bp = pxx[mask]
    total = float(np.nansum(bp))
    if total <= 1e-12:
        return {
            "harmonic_ratio": np.nan,
            "spectral_flatness": np.nan,
            "dominant_peak_prominence": np.nan,
            "low_high_band_power_ratio": np.nan,
        }
    peak_idx = int(np.nanargmax(bp))
    peak_freq = float(bf[peak_idx])
    peak_power = float(bp[peak_idx])
    noise_floor = float(np.nanmedian(bp))
    flatness = float(np.exp(np.nanmean(np.log(bp + 1e-12))) / (np.nanmean(bp) + 1e-12))
    low = float(np.nansum(bp[(bf >= 0.6) & (bf < 1.5)]))
    high = float(np.nansum(bp[(bf >= 1.5) & (bf <= 3.0)]))

    harmonic_power = 0.0
    for mult in (1, 2, 3):
        center = peak_freq * mult
        band = (bf >= center - 0.15) & (bf <= center + 0.15)
        if np.any(band):
            harmonic_power += float(np.nansum(bp[band]))
    return {
        "harmonic_ratio": harmonic_power / total,
        "spectral_flatness": flatness,
        "dominant_peak_prominence": (peak_power - noise_floor) / (total + 1e-12),
        "low_high_band_power_ratio": low / (high + 1e-12),
    }


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) > 500:
        idx = np.linspace(0, len(x) - 1, 500).round().astype(int)
        x = x[idx]
    if len(x) < 40:
        return np.nan
    sd = float(np.nanstd(x))
    if sd <= 1e-12:
        return np.nan
    r = r_ratio * sd

    def _count(mm: int) -> int:
        emb = np.array([x[i : i + mm] for i in range(len(x) - mm + 1)])
        count = 0
        for i in range(len(emb) - 1):
            dist = np.max(np.abs(emb[i + 1 :] - emb[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    a = _count(m + 1)
    b = _count(m)
    if a <= 0 or b <= 0:
        return np.nan
    return float(-np.log(a / b))


def jerk_features(sig: np.ndarray, fs: float) -> dict:
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 4:
        return {"jerk_rms": np.nan, "jerk_entropy": np.nan}
    jerk = np.diff(sig) * fs
    rms = float(np.sqrt(np.nanmean(jerk**2)))
    _, _, ent = spec_entropy(jerk, fs)
    return {"jerk_rms": rms, "jerk_entropy": ent}


def shape_cv(x: np.ndarray, fs: float, stride_sec: float) -> tuple[float, float]:
    stride_n = int(round(stride_sec * fs))
    if stride_n < int(0.8 * fs) or stride_n > int(1.8 * fs):
        return np.nan, np.nan
    waves = []
    peak_times = []
    grid = np.linspace(0, stride_n - 1, 100)
    for start in range(0, len(x) - stride_n + 1, stride_n):
        seg = np.asarray(x[start : start + stride_n], dtype=float)
        sd = float(np.nanstd(seg))
        if len(seg) != stride_n or sd <= 1e-12:
            continue
        z = (seg - np.nanmean(seg)) / sd
        waves.append(np.interp(grid, np.arange(stride_n), z))
        peak_times.append(float(np.nanargmax(z) / max(1, stride_n - 1) * 100))
    if len(waves) < 3:
        return np.nan, np.nan
    arr = np.vstack(waves)
    mean_wave = np.nanmean(arr, axis=0)
    sd_wave = np.nanstd(arr, axis=0)
    return float(np.nanmean(sd_wave / (np.abs(mean_wave) + 1e-3))), float(np.nanstd(peak_times))


def pattern_features(acc: np.ndarray, fs: float, dataset: str, subject_id: str, source_id: str) -> dict:
    acc = np.asarray(acc, dtype=float)
    # Use magnitude-normalized pattern features to avoid over-reading axis conventions.
    mag = np.linalg.norm(acc, axis=1)
    sig = bandpass(mag, fs)
    c = acf(sig)
    step_sec, step_peak, _ = peak_in_range(c, fs, 0.30, 0.80)
    stride_sec, stride_peak, stride_width = peak_in_range(c, fs, 0.80, 1.70)
    dom_freq, peak_ratio, entropy = spec_entropy(sig, fs)
    band_extra = band_power_features(sig, fs)
    spectral_extra = spectral_shape_features(sig, fs)
    cv, peak_sd = shape_cv(sig, fs, stride_sec)
    step_stride_asym = np.nan
    if np.isfinite(step_peak) and np.isfinite(stride_peak) and abs(stride_peak) > 1e-12:
        step_stride_asym = float(abs(stride_peak - step_peak) / (abs(stride_peak) + 1e-12))
    stride_step_ratio = np.nan
    if np.isfinite(step_peak) and abs(step_peak) > 1e-12 and np.isfinite(stride_peak):
        stride_step_ratio = float(stride_peak / step_peak)
    stride_sharpness = np.nan
    if np.isfinite(stride_peak) and np.isfinite(stride_width) and stride_width > 1e-12:
        stride_sharpness = float(stride_peak / stride_width)
    decay_1s = np.nan
    idx_1s = int(round(fs))
    if 0 <= idx_1s < len(c):
        decay_1s = float(c[idx_1s])
    jerk_extra = jerk_features(sig, fs)
    return {
        "dataset": dataset,
        "subject_id": str(subject_id),
        "source_id": str(source_id),
        "fs": fs,
        "duration_sec": len(acc) / fs,
        "step_sec": step_sec,
        "stride_sec": stride_sec,
        "cadence": 60 / step_sec if np.isfinite(step_sec) and step_sec > 0 else np.nan,
        "acf_step_peak": step_peak,
        "acf_stride_peak": stride_peak,
        "acf_stride_peak_width_sec": stride_width,
        "spec_peak_freq": dom_freq,
        "spec_peak_ratio": peak_ratio,
        "spec_entropy": entropy,
        "stride_shape_cv_mean": cv,
        "peak_timing_sd_pct": peak_sd,
        "step_stride_regularity_asymmetry": step_stride_asym,
        "sample_entropy": sample_entropy(sig),
        "sig_rms": float(np.sqrt(np.nanmean(sig**2))),
        "sig_iqr": float(np.nanpercentile(sig, 75) - np.nanpercentile(sig, 25)),
        "sig_range": float(np.nanpercentile(sig, 95) - np.nanpercentile(sig, 5)),
        "acf_stride_step_ratio": stride_step_ratio,
        "acf_stride_peak_sharpness": stride_sharpness,
        "acf_decay_1s": decay_1s,
        **spectral_extra,
        **band_extra,
        **jerk_extra,
    }


def iter_uci_har() -> list[dict]:
    base = ROOT / "external_data" / "uci_har" / "dataset" / "UCI HAR Dataset"
    rows = []
    for split in ("train", "test"):
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=int)
        subjects = np.loadtxt(base / split / f"subject_{split}.txt", dtype=int)
        sig_dir = base / split / "Inertial Signals"
        ax = np.loadtxt(sig_dir / f"total_acc_x_{split}.txt")
        ay = np.loadtxt(sig_dir / f"total_acc_y_{split}.txt")
        az = np.loadtxt(sig_dir / f"total_acc_z_{split}.txt")
        for i in np.flatnonzero(y == 1):
            acc = np.column_stack([ax[i], ay[i], az[i]])
            rows.append(pattern_features(acc, 50.0, "UCI_HAR", subjects[i], f"{split}_{i}"))
    return rows


def _parse_wisdm_arff(path: Path, max_rows: int = 5000) -> pd.DataFrame:
    cols = []
    data = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    cols.append(parts[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data and not low.startswith("@"):
                data.append(next(csv.reader([line])))
                if len(data) >= max_rows:
                    break
    return pd.DataFrame(data, columns=cols[: len(data[0])] if data else cols)


def iter_wisdm(limit_subjects: int = 40) -> list[dict]:
    base = ROOT / "external_data" / "wisdm_2019" / "dataset" / "wisdm-dataset" / "arff_files" / "phone" / "accel"
    rows = []
    for path in sorted(base.glob("data_*_accel_phone.arff"))[:limit_subjects]:
        df = _parse_wisdm_arff(path, max_rows=12000)
        if df.empty or "ACTIVITY" not in df.columns:
            continue
        walk = df[df["ACTIVITY"].astype(str).str.strip().eq("A")].copy()
        cols = [c for c in walk.columns if c.lower() in {"x0", "y0", "z0", "x", "y", "z"}]
        if len(cols) < 3:
            numeric = [c for c in walk.columns if c not in {"ACTIVITY", "class"}]
            cols = numeric[-3:]
        vals = walk[cols[:3]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(vals) < 200:
            continue
        acc = vals.to_numpy(float)
        rows.append(pattern_features(acc, 20.0, "WISDM_PHONE", path.stem.split("_")[1], path.name))
    return rows


def iter_geotec() -> list[dict]:
    base = ROOT / "external_data" / "geotec_tug_smartphone" / "extracted"
    rows = []
    for path in sorted(base.rglob("*_sp.csv")):
        df = pd.read_csv(path)
        walk = df[df["label"].astype(str).eq("WALKING")].copy()
        if len(walk) < 200:
            continue
        t = pd.to_numeric(walk["timestamp"], errors="coerce").to_numpy(float)
        duration = (np.nanmax(t) - np.nanmin(t)) / 1000 if len(t) else np.nan
        fs = len(walk) / duration if np.isfinite(duration) and duration > 0 else 25.0
        fs = float(np.clip(fs, 10, 100))
        acc = walk[["x_acc", "y_acc", "z_acc"]].to_numpy(float) / 9.80665
        subject = re.search(r"s\d+", path.name)
        rows.append(pattern_features(acc, fs, "GEOTEC_SP", subject.group(0) if subject else path.stem, path.name))
    return rows


def iter_existing_final_table() -> list[dict]:
    table = pd.read_csv(ROOT / "analysis_outputs" / "all_extractors_domain_stability_screen" / "fixed_best10_quality_subject_table.csv")
    normals = table[table["target"].eq(0)].copy()
    keep = [
        "dataset",
        "subject_id",
        "group_id",
        "step_time_median",
        "stride_time_median",
        "cadence",
        "v_stride_regularity",
        "v_acf_stride_peak_width_sec",
        "v_spec_entropy",
        "v_stride_shape_cv_mean",
        "v_peak_timing_sd_pct",
    ]
    rows = []
    for _, row in normals[keep].iterrows():
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "subject_id": str(row["subject_id"]),
                "source_id": str(row["group_id"]),
                "fs": np.nan,
                "duration_sec": np.nan,
                "step_sec": row["step_time_median"],
                "stride_sec": row["stride_time_median"],
                "cadence": row["cadence"],
                "acf_step_peak": np.nan,
                "acf_stride_peak": row["v_stride_regularity"],
                "acf_stride_peak_width_sec": row["v_acf_stride_peak_width_sec"],
                "spec_peak_freq": np.nan,
                "spec_peak_ratio": np.nan,
                "spec_entropy": row["v_spec_entropy"],
                "stride_shape_cv_mean": row["v_stride_shape_cv_mean"],
                "peak_timing_sd_pct": row["v_peak_timing_sd_pct"],
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "step_sec",
        "stride_sec",
        "cadence",
        "acf_stride_peak",
        "acf_stride_peak_width_sec",
        "spec_peak_freq",
        "spec_peak_ratio",
        "spec_entropy",
        "stride_shape_cv_mean",
        "peak_timing_sd_pct",
        "step_stride_regularity_asymmetry",
        "sample_entropy",
        "sig_rms",
        "sig_iqr",
        "sig_range",
        "acf_stride_step_ratio",
        "acf_stride_peak_sharpness",
        "acf_decay_1s",
        "harmonic_ratio",
        "spectral_flatness",
        "dominant_peak_prominence",
        "low_high_band_power_ratio",
        "bandpower_0p6_1p2",
        "bandpower_1p2_2p0",
        "bandpower_2p0_3p0",
        "bandpower_mid_total_ratio",
        "jerk_rms",
        "jerk_entropy",
    ]
    rows = []
    for dataset, part in df.groupby("dataset"):
        for feature in features:
            x = pd.to_numeric(part[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "n": len(x),
                    "median": x.median(),
                    "iqr": x.quantile(0.75) - x.quantile(0.25),
                    "p10": x.quantile(0.10),
                    "p90": x.quantile(0.90),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(iter_existing_final_table())
    rows.extend(iter_uci_har())
    rows.extend(iter_wisdm())
    rows.extend(iter_geotec())
    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(OUT_DIR / "normal_pattern_features_by_subject_or_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "normal_pattern_feature_summary.csv", index=False, encoding="utf-8-sig")
    print("counts")
    print(df.groupby("dataset").size().to_string())
    print("\nsummary focus")
    focus = summary[summary["feature"].isin(["step_sec", "stride_sec", "cadence", "acf_stride_peak", "spec_entropy", "stride_shape_cv_mean"])]
    print(focus.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
