from __future__ import annotations

import csv
import re
import sys
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt, spectrogram
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

from gait_axis_aligned_core import FEATURES as CORE_FEATURES
from gait_axis_aligned_core import TARGET_FS_HZ, WINDOW_SEC, extract_axis_aligned_gait_features, extract_best10_from_acc_array

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
OUT_DIR = ROOT / "analysis_outputs" / "axis_aligned_gait_model"
FS_OUT = TARGET_FS_HZ


FEATURES = list(CORE_FEATURES)


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


def resample_to_100hz(acc: np.ndarray, fs: float) -> np.ndarray:
    acc = np.asarray(acc, dtype=float)
    if len(acc) < 4:
        return acc
    t = np.arange(len(acc), dtype=float) / fs
    grid = np.arange(0.0, t[-1], 1.0 / FS_OUT)
    return np.column_stack([np.interp(grid, t, acc[:, i]) for i in range(3)])


def align_to_vmlap(acc: np.ndarray, fs: float, already_vmlap: bool = False) -> tuple[np.ndarray, dict]:
    acc = np.asarray(acc, dtype=float)[:, :3]
    keep = np.isfinite(acc).all(axis=1)
    acc = acc[keep]
    if len(acc) < 4:
        raise ValueError("not enough finite accelerometer rows")
    if already_vmlap:
        aligned = acc.copy()
        if np.nanmedian(aligned[:, 0]) < 0:
            aligned[:, 0] *= -1
        return aligned, {"alignment": "provided_vmlap", "vertical_raw_axis": 0, "vertical_sign": "+"}

    med = np.nanmedian(acc, axis=0)
    vertical_idx = int(np.nanargmax(np.abs(med)))
    vertical_sign = 1.0 if med[vertical_idx] >= 0 else -1.0
    v = acc[:, vertical_idx] * vertical_sign

    remaining = [i for i in range(3) if i != vertical_idx]
    powers = []
    for idx in remaining:
        sig = bandpass(acc[:, idx], fs, 0.6, 3.0)
        powers.append(float(np.nanvar(sig)))
    # Deterministic anatomical approximation for unlabeled phone axes:
    # stronger rhythmic horizontal axis -> AP candidate, weaker -> ML candidate.
    if not np.isfinite(powers).any():
        powers = [float(np.nanvar(acc[:, idx])) for idx in remaining]
    if not np.isfinite(powers).any():
        raise ValueError("cannot infer horizontal gait axis")
    ap_pos = int(np.nanargmax(powers))
    ap_idx = remaining[ap_pos]
    ml_idx = remaining[1 - ap_pos]
    ap = acc[:, ap_idx]
    ml = acc[:, ml_idx]
    if np.nanmedian(ap) < 0:
        ap *= -1
    if np.nanmedian(ml) < 0:
        ml *= -1
    return np.column_stack([v, ml, ap]), {
        "alignment": "gravity_plus_horizontal_power",
        "vertical_raw_axis": vertical_idx,
        "vertical_sign": "+" if vertical_sign > 0 else "-",
        "ml_raw_axis": ml_idx,
        "ap_raw_axis": ap_idx,
        "horizontal_power_axis0": powers[0],
        "horizontal_power_axis1": powers[1],
    }


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
    if len(peaks):
        idx = lo + int(peaks[np.argmax(props["prominences"])])
    else:
        idx = lo + int(np.nanargmax(seg))
    height = float(c[idx])
    half = max(0.0, height * 0.5)
    left = idx
    while left > 1 and c[left] >= half:
        left -= 1
    right = idx
    while right < len(c) - 1 and c[right] >= half:
        right += 1
    return idx / fs, height, (right - left) / fs


def spec_features(x: np.ndarray, fs: float) -> tuple[float, float, float]:
    freqs, _, pxx = spectrogram(
        x - np.nanmean(x),
        fs=fs,
        window="hann",
        nperseg=max(32, min(len(x), int(round(4 * fs)))),
        detrend=False,
        scaling="density",
        mode="psd",
    )
    mask = (freqs >= 0.6) & (freqs <= 3.0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    band = pxx[mask, :]
    mean_spec = np.nanmean(band, axis=1)
    total = float(np.nansum(mean_spec))
    if total <= 1e-12:
        return np.nan, np.nan, np.nan
    peak = int(np.nanargmax(mean_spec))
    prob = band.reshape(-1)
    prob = prob / (np.nansum(prob) + 1e-12)
    ent = float(-np.nansum(prob * np.log2(prob + 1e-12)) / np.log2(len(prob))) if len(prob) > 1 else 0.0
    return float(freqs[mask][peak]), float(mean_spec[peak] / total), ent


def shape_cv(x: np.ndarray, fs: float, stride_sec: float) -> float:
    if not np.isfinite(stride_sec):
        return np.nan
    stride_n = int(round(stride_sec * fs))
    if stride_n < int(0.8 * fs) or stride_n > int(1.8 * fs):
        return np.nan
    waves = []
    grid = np.linspace(0, stride_n - 1, 100)
    for start in range(0, len(x) - stride_n + 1, stride_n):
        seg = np.asarray(x[start : start + stride_n], dtype=float)
        sd = float(np.nanstd(seg))
        if sd <= 1e-12:
            continue
        z = (seg - np.nanmean(seg)) / sd
        waves.append(np.interp(grid, np.arange(stride_n), z))
    if len(waves) < 3:
        return np.nan
    return float(np.nanmean(np.nanstd(np.vstack(waves), axis=0)))


def extract_features(vmlap: np.ndarray, fs: float = FS_OUT) -> dict:
    sigs = {"v": bandpass(vmlap[:, 0], fs), "ml": bandpass(vmlap[:, 1], fs), "ap": bandpass(vmlap[:, 2], fs)}
    c_v = acf(sigs["v"])
    step_sec, step_peak, _ = peak_in_range(c_v, fs, 0.30, 0.80)
    stride_sec, stride_peak, stride_width = peak_in_range(c_v, fs, 0.80, 1.70)
    out = {
        "step_sec": step_sec,
        "stride_sec": stride_sec,
        "v_acf_step_peak": step_peak,
        "v_acf_stride_peak": stride_peak,
        "v_acf_stride_peak_width_sec": stride_width,
    }
    for axis, sig in sigs.items():
        if axis != "v":
            c = acf(sig)
            _, out[f"{axis}_acf_stride_peak"], out[f"{axis}_acf_stride_peak_width_sec"] = peak_in_range(c, fs, 0.80, 1.70)
        _, out[f"{axis}_spec_peak_ratio"], out[f"{axis}_spec_entropy"] = spec_features(sig, fs)
        out[f"{axis}_stride_shape_cv_mean"] = shape_cv(sig, fs, stride_sec)
    out["quality_score"] = out.get("v_acf_stride_peak", np.nan)
    return out


def best10_features(acc: np.ndarray, fs: float, already_vmlap: bool, meta: dict) -> dict | None:
    try:
        extracted = extract_best10_from_acc_array(acc, fs, already_vmlap=already_vmlap)
    except Exception:
        return None
    window = extracted["window"]
    all_features = extracted.get("all_features", {})
    return {
        **meta,
        **{k: v for k, v in window.items() if k not in {"start_sec", "end_sec"}},
        **all_features,
        **extracted["features"],
        "quality_score": all_features.get("quality_score", extracted["features"].get("v_acf_stride_peak", np.nan)),
        "best10_start_sec": window["start_sec"],
        "best10_end_sec": window["end_sec"],
    }


def add_label(row: dict, group: str) -> dict:
    out = dict(row)
    out["group"] = group
    out["target"] = 1 if group == "impaired" else 0
    out["group_id"] = f"{out['dataset']}::{out['subject_id']}"
    return out


def iter_samples() -> list[dict]:
    out_rows = []
    sample_dir = next(p for p in ROOT.iterdir() if p.is_dir() and "SAMPLE" in p.name)
    for path in sorted(sample_dir.glob("*.csv")):
        group = "impaired" if "발다침" in path.stem else "normal"
        try:
            extracted = extract_axis_aligned_gait_features(str(path))
        except Exception:
            continue
        window = extracted["window"]
        row = {
            "dataset": "OUR_SAMPLE",
            "subject_id": path.stem,
            "source_id": path.name,
            **{k: v for k, v in window.items() if k not in {"start_sec", "end_sec"}},
            **extracted["features"],
            "quality_score": extracted["features"].get("v_acf_stride_peak", np.nan),
            "best10_start_sec": window["start_sec"],
            "best10_end_sec": window["end_sec"],
        }
        out_rows.append(add_label(row, group))
    return out_rows

    rows = []
    sample_dir = next(p for p in ROOT.iterdir() if p.is_dir() and "SAMPLE" in p.name)
    for path in sorted(sample_dir.glob("*.csv")):
        df = pd.read_csv(path, comment="#")
        if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(df.columns):
            acc = df[["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]].to_numpy(float)
            already = True
        else:
            acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float)
            if np.nanmedian(np.linalg.norm(acc, axis=1)) > 3.0:
                acc = acc / 9.80665
            already = False
        t = pd.to_numeric(df.get("Timestamp_ns", pd.Series(np.arange(len(df)))), errors="coerce").to_numpy(float)
        dur = (np.nanmax(t) - np.nanmin(t)) / 1e9 if "Timestamp_ns" in df.columns else len(df) / FS_OUT
        fs = len(df) / dur if dur and dur > 0 else FS_OUT
        group = "impaired" if "발다침" in path.stem else "normal"
        feat = best10_features(acc, fs, already, {"dataset": "OUR_SAMPLE", "subject_id": path.stem, "source_id": path.name})
        if feat:
            rows.append(add_label(feat, group))
    return rows


def iter_samples() -> list[dict]:
    rows = []
    sample_dir = next(p for p in ROOT.iterdir() if p.is_dir() and "SAMPLE" in p.name)
    for path in sorted(sample_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, comment="#")
            if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(df.columns):
                acc = df[["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]].to_numpy(float)
                already = True
            else:
                acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float)
                if np.nanmedian(np.linalg.norm(acc, axis=1)) > 3.0:
                    acc = acc / 9.80665
                already = False
            t = pd.to_numeric(df.get("Timestamp_ns", pd.Series(np.arange(len(df)))), errors="coerce").to_numpy(float)
            dur = (np.nanmax(t) - np.nanmin(t)) / 1e9 if "Timestamp_ns" in df.columns else len(df) / FS_OUT
            fs = len(df) / dur if dur and dur > 0 else FS_OUT
            group = "impaired" if "발다침" in path.stem else "normal"
            feat = best10_features(acc, fs, already, {"dataset": "OUR_SAMPLE", "subject_id": path.stem, "source_id": path.name})
        except Exception:
            feat = None
        if feat:
            rows.append(add_label(feat, group))
    return rows


def iter_physionet_labwalks(limit_each: int | None = None) -> list[dict]:
    base = GAIT_PROJECT / "physionet_AWS" / "LabWalks"
    stems = [p.stem for p in sorted(base.glob("*_base.hea"))]
    if limit_each:
        cos = [s for s in stems if s.startswith("co")][:limit_each]
        fls = [s for s in stems if s.startswith("fl")][:limit_each]
        stems = cos + fls
    rows = []
    for stem in stems:
        lines = (base / f"{stem}.hea").read_text(encoding="utf-8").splitlines()
        parts = lines[0].split()
        fs = float(parts[2])
        n = int(parts[3])
        ch = int(parts[1])
        gains, baselines = [], []
        for line in lines[1 : 1 + ch]:
            m = re.match(r"([0-9.]+)\((-?\d+)\)/", line.split()[2])
            gains.append(float(m.group(1)))
            baselines.append(float(m.group(2)))
        raw = np.memmap(base / f"{stem}.dat", dtype="<i2", mode="r", shape=(n, ch))
        data = (raw.astype(float) - np.array(baselines)) / np.array(gains)
        group = "normal" if stem.startswith("co") else "impaired"
        feat = best10_features(data[:, :3], fs, True, {"dataset": "PhysioNet_LabWalks", "subject_id": stem, "source_id": stem})
        if feat:
            rows.append(add_label(feat, group))
    return rows


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
        for subject in sorted(set(subjects)):
            idx = np.flatnonzero((subjects == subject) & (y == 1))
            if len(idx) < 4:
                continue
            acc = np.vstack([np.column_stack([ax[i], ay[i], az[i]]) for i in idx[:8]])
            feat = best10_features(acc, 50.0, False, {"dataset": "UCI_HAR", "subject_id": str(subject), "source_id": f"{split}_{subject}"})
            if feat:
                rows.append(add_label(feat, "normal"))
    return rows


def _parse_wisdm_arff(path: Path, max_rows: int = 12000) -> pd.DataFrame:
    cols, data, in_data = [], [], False
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
        df = _parse_wisdm_arff(path)
        if df.empty or "ACTIVITY" not in df.columns:
            continue
        walk = df[df["ACTIVITY"].astype(str).str.strip().eq("A")].copy()
        cols = [c for c in walk.columns if c.lower() in {"x0", "y0", "z0", "x", "y", "z"}]
        vals = walk[cols[:3]].apply(pd.to_numeric, errors="coerce").dropna() if len(cols) >= 3 else pd.DataFrame()
        if len(vals) < 200:
            continue
        feat = best10_features(vals.to_numpy(float), 20.0, False, {"dataset": "WISDM_PHONE", "subject_id": path.stem.split("_")[1], "source_id": path.name})
        if feat:
            rows.append(add_label(feat, "normal"))
    return rows


def iter_geotec() -> list[dict]:
    base = ROOT / "external_data" / "geotec_tug_smartphone" / "extracted"
    rows = []
    for path in sorted(base.rglob("*_sp.csv")):
        df = pd.read_csv(path)
        if "label" not in df.columns:
            continue
        walk = df[df["label"].astype(str).eq("WALKING")].copy()
        if len(walk) < 120 or not {"x_acc", "y_acc", "z_acc"}.issubset(walk.columns):
            continue
        t = pd.to_numeric(walk["timestamp"], errors="coerce").to_numpy(float)
        duration = (np.nanmax(t) - np.nanmin(t)) / 1000 if len(t) else np.nan
        fs = len(walk) / duration if np.isfinite(duration) and duration > 0 else 25.0
        fs = float(np.clip(fs, 10, 100))
        acc = walk[["x_acc", "y_acc", "z_acc"]].to_numpy(float) / 9.80665
        subject = re.search(r"s\d+", path.name)
        subject_id = subject.group(0) if subject else path.stem
        feat = best10_features(acc, fs, False, {"dataset": "GEOTEC_SP", "subject_id": subject_id, "source_id": path.name})
        if feat:
            rows.append(add_label(feat, "normal"))
    return rows


def iter_chapman_off(max_windows_per_subject: int = 1) -> list[dict]:
    zpath = ROOT / "external_data" / "chapman_pd" / "RawWalkingDatabase.zip"
    rows = []
    with zipfile.ZipFile(zpath) as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            df = pd.read_csv(zf.open(name), usecols=["accelerometer_x", "accelerometer_y", "accelerometer_z", "class"])
            off = df[df["class"].astype(str).eq("C")]
            if len(off) < int(80 * WINDOW_SEC):
                continue
            acc = off[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].to_numpy(float) / 9.80665
            subject = Path(name).stem
            feat = best10_features(acc, 80.0, False, {"dataset": "Chapman_PD_OFF_RAW", "subject_id": subject, "source_id": name})
            if feat:
                rows.append(add_label(feat, "impaired"))
    return rows


def iter_fog_star(max_subjects: int | None = None) -> list[dict]:
    path = ROOT / "external_data" / "fog_star" / "sensor_data.csv"
    df = pd.read_csv(path, usecols=["timestamp", "back_acc_x", "back_acc_y", "back_acc_z", "activity", "subjectID", "sessionID", "taskID"])
    df = df[df["activity"].eq(1)].copy()
    rows = []
    seen = 0
    for keys, part in df.groupby(["subjectID", "sessionID", "taskID"], sort=True):
        acc = part.sort_values("timestamp")[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(float)
        if len(acc) < int(60 * WINDOW_SEC):
            continue
        subject = str(keys[0])
        feat = best10_features(acc, 60.0, False, {"dataset": "FoG_STAR_BACK_WALK", "subject_id": subject, "source_id": f"{keys}"})
        if feat:
            rows.append(add_label(feat, "impaired"))
            seen += 1
            if max_subjects and seen >= max_subjects:
                break
    return rows


def model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])


def pick_sens80_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_spec = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= 0.80 and spec > best_spec:
            best_thr, best_spec = float(thr), float(spec)
    return best_thr


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, pred)),
        "sens": float(tp / (tp + fn)) if tp + fn else np.nan,
        "spec": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred)),
        "threshold": float(thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def group_oof(table: pd.DataFrame, features: list[str]) -> dict | None:
    data = table.dropna(subset=["target"]).copy()
    data = data.dropna(subset=features, how="all")
    if len(data) < 20 or data["target"].nunique() < 2:
        return None
    y = data["target"].astype(int).to_numpy()
    groups = data["group_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 5:
        return None
    oof = np.zeros(len(data))
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(data[features], y, groups):
        clf = model()
        clf.fit(data.iloc[tr][features], y[tr])
        oof[te] = clf.predict_proba(data.iloc[te][features])[:, 1]
    return metrics(y, oof, pick_sens80_threshold(y, oof))


def screen_models(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_features = [f for f in FEATURES if f in table.columns and pd.to_numeric(table[f], errors="coerce").notna().sum() >= 10]
    for k in (2, 3, 4):
        for combo in combinations(candidate_features, k):
            features = list(combo)
            corr = table[features].corr(method="spearman").abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            if upper.max().max() > 0.85:
                continue
            res = group_oof(table, features)
            if res is None:
                continue
            rows.append({"features": " + ".join(features), "k": k, "max_abs_spearman": float(upper.max().max()), **res})
    return pd.DataFrame(rows).sort_values(["sens", "auc", "spec"], ascending=[False, False, False])


def predict_samples(public_table: pd.DataFrame, sample_table: pd.DataFrame, model_screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if model_screen.empty or sample_table.empty:
        return pd.DataFrame()
    for _, model_row in model_screen.head(30).iterrows():
        features = [f.strip() for f in str(model_row["features"]).split("+")]
        train = public_table.dropna(subset=["target"]).copy()
        train = train.dropna(subset=features, how="all")
        sample = sample_table.dropna(subset=features, how="all").copy()
        if train.empty or sample.empty:
            continue
        y = train["target"].astype(int).to_numpy()
        clf = model()
        clf.fit(train[features], y)
        train_p = clf.predict_proba(train[features])[:, 1]
        thr = pick_sens80_threshold(y, train_p)
        sample_p = clf.predict_proba(sample[features])[:, 1]
        for i, (_, sample_row) in enumerate(sample.iterrows()):
            rows.append(
                {
                    "features": model_row["features"],
                    "model_auc_oof": model_row["auc"],
                    "model_sens_oof": model_row["sens"],
                    "model_spec_oof": model_row["spec"],
                    "sample_id": sample_row["subject_id"],
                    "sample_target": int(sample_row["target"]),
                    "probability_impaired": float(sample_p[i]),
                    "threshold": float(thr),
                    "prediction": int(sample_p[i] >= thr),
                    "correct": bool(int(sample_p[i] >= thr) == int(sample_row["target"])),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(iter_physionet_labwalks())
    rows.extend(iter_samples())
    rows.extend(iter_uci_har())
    rows.extend(iter_wisdm())
    rows.extend(iter_geotec())
    rows.extend(iter_chapman_off())
    rows.extend(iter_fog_star())
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "axis_aligned_best10_subject_table.csv", index=False, encoding="utf-8-sig")
    public_table = table[table["dataset"].ne("OUR_SAMPLE")].copy()
    sample_table = table[table["dataset"].eq("OUR_SAMPLE")].copy()
    model_screen = screen_models(public_table)
    sample_predictions = predict_samples(public_table, sample_table, model_screen)
    model_screen.to_csv(OUT_DIR / "axis_aligned_public_model_screen.csv", index=False, encoding="utf-8-sig")
    sample_predictions.to_csv(OUT_DIR / "axis_aligned_public_model_sample_predictions.csv", index=False, encoding="utf-8-sig")
    print("counts")
    print(table.groupby(["group", "dataset"]).size().to_string())
    print("\nalignment")
    print(table.groupby(["dataset", "alignment"]).size().to_string())
    print("\nbest models")
    print(model_screen.head(20).to_string(index=False))
    if not sample_predictions.empty:
        print("\nsample predictions for top model")
        top = sample_predictions[sample_predictions["features"].eq(model_screen.iloc[0]["features"])]
        print(top.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
