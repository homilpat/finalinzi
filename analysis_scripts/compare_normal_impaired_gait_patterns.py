from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_normal_gait_pattern_features import iter_geotec, iter_uci_har, iter_wisdm, pattern_features


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison"
WINDOW_SEC = 10.0


FEATURES = [
    "step_sec",
    "stride_sec",
    "cadence",
    "acf_step_peak",
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


def safe_pattern_features(acc: np.ndarray, fs: float, dataset: str, subject_id: str, source_id: str) -> dict | None:
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[0] < int(fs * WINDOW_SEC * 0.8) or acc.shape[1] < 3:
        return None
    if not np.isfinite(acc).all():
        return None
    if float(np.nanstd(acc)) <= 1e-8:
        return None
    try:
        feat = pattern_features(acc, fs, dataset, subject_id, source_id)
    except Exception:
        return None
    finite_core = sum(np.isfinite(pd.to_numeric(pd.Series([feat.get(k) for k in FEATURES]), errors="coerce")))
    return feat if finite_core >= 4 else None


def _add_label(row: dict, group: str, source_note: str) -> dict:
    out = dict(row)
    out["group"] = group
    out["target"] = 1 if group == "impaired" else 0
    out["source_note"] = source_note
    return out


def iter_existing_fixed_best10_table() -> list[dict]:
    path = ROOT / "analysis_outputs" / "all_extractors_domain_stability_screen" / "fixed_best10_quality_subject_table.csv"
    table = pd.read_csv(path)
    rows: list[dict] = []
    for _, row in table.iterrows():
        if not np.isfinite(row.get("target", np.nan)):
            continue
        group = "impaired" if float(row["target"]) == 1.0 else "normal"
        rows.append(
            _add_label(
                {
                    "dataset": str(row["dataset"]),
                    "subject_id": str(row["subject_id"]),
                    "source_id": str(row["group_id"]),
                    "fs": np.nan,
                    "duration_sec": np.nan,
                    "step_sec": row.get("step_time_median", np.nan),
                    "stride_sec": row.get("stride_time_median", np.nan),
                    "cadence": row.get("cadence", np.nan),
                    "acf_step_peak": np.nan,
                    "acf_stride_peak": row.get("v_stride_regularity", np.nan),
                    "acf_stride_peak_width_sec": row.get("v_acf_stride_peak_width_sec", np.nan),
                    "spec_peak_freq": np.nan,
                    "spec_peak_ratio": np.nan,
                    "spec_entropy": row.get("v_spec_entropy", np.nan),
                    "stride_shape_cv_mean": row.get("v_stride_shape_cv_mean", np.nan),
                    "peak_timing_sd_pct": row.get("v_peak_timing_sd_pct", np.nan),
                },
                group,
                "existing fixed_best10_quality subject table",
            )
        )
    return rows


def iter_added_normal_datasets() -> list[dict]:
    rows: list[dict] = []
    for source_rows, note in [
        (iter_uci_har(), "UCI HAR smartphone walking windows"),
        (iter_wisdm(), "WISDM phone walking windows"),
        (iter_geotec(), "GeoTec smartphone TUG walking segments"),
    ]:
        for row in source_rows:
            rows.append(_add_label(row, "normal", note))
    return rows


def _window_indices(n: int, fs: float, max_windows: int | None = None) -> list[tuple[int, int]]:
    win = int(round(WINDOW_SEC * fs))
    if win <= 0 or n < win:
        return []
    starts = list(range(0, n - win + 1, win))
    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return [(s, s + win) for s in starts]


def iter_fog_star(max_windows_per_subject: int = 12) -> list[dict]:
    path = ROOT / "external_data" / "fog_star" / "sensor_data.csv"
    if not path.exists():
        return []
    usecols = [
        "timestamp",
        "back_acc_x",
        "back_acc_y",
        "back_acc_z",
        "activity",
        "fog",
        "subjectID",
        "sessionID",
        "taskID",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["activity"].eq(1)].copy()  # README: 1 = Walk
    rows: list[dict] = []
    for keys, part in df.groupby(["subjectID", "sessionID", "taskID"], sort=True):
        part = part.sort_values("timestamp")
        acc = part[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(float)
        for j, (s, e) in enumerate(_window_indices(len(acc), 60.0, max_windows_per_subject)):
            feat = safe_pattern_features(acc[s:e], 60.0, "FoG_STAR_BACK_WALK", str(keys[0]), f"s{keys[0]}_sess{keys[1]}_task{keys[2]}_w{j}")
            if feat is None:
                continue
            fog_rate = float(pd.to_numeric(part.iloc[s:e]["fog"], errors="coerce").mean())
            feat["fog_rate"] = fog_rate
            rows.append(_add_label(feat, "impaired", "FoG-STAR PD lower-back walking activity"))
    return rows


def iter_pd_turning(max_windows_per_file: int = 4) -> list[dict]:
    zpath = ROOT / "external_data" / "pd_turning_figshare" / "IMU.zip"
    if not zpath.exists():
        return []
    rows: list[dict] = []
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt") and "standing" not in n.lower()]
        for name in names:
            try:
                df = pd.read_csv(zf.open(name), sep="\t")
            except Exception:
                continue
            cols = ["ACC ML [g]", "ACC AP [g]", "ACC SI [g]"]
            if not set(cols).issubset(df.columns):
                continue
            acc = df[cols].to_numpy(float)
            subject = re.search(r"SUB\d+", name)
            fs = 128.0
            for j, (s, e) in enumerate(_window_indices(len(acc), fs, max_windows_per_file)):
                feat = safe_pattern_features(acc[s:e], fs, "PD_TURNING_IMU", subject.group(0) if subject else name, f"{name}_w{j}")
                if feat is None:
                    continue
                if "Freezing event [flag]" in df.columns:
                    feat["fog_rate"] = float(pd.to_numeric(df.iloc[s:e]["Freezing event [flag]"], errors="coerce").mean())
                rows.append(_add_label(feat, "impaired", "PD turning task, not straight walking"))
    return rows


def iter_chapman_off_raw(max_windows_per_subject: int = 5) -> list[dict]:
    zpath = ROOT / "external_data" / "chapman_pd" / "RawWalkingDatabase.zip"
    if not zpath.exists():
        return []
    rows: list[dict] = []
    with zipfile.ZipFile(zpath) as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            try:
                df = pd.read_csv(
                    zf.open(name),
                    usecols=["accelerometer_x", "accelerometer_y", "accelerometer_z", "class"],
                )
            except Exception:
                continue
            off = df[df["class"].astype(str).eq("C")]
            if len(off) < int(80 * WINDOW_SEC):
                continue
            acc = off[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].to_numpy(float) / 9.80665
            subject = Path(name).stem
            for j, (s, e) in enumerate(_window_indices(len(acc), 80.0, max_windows_per_subject)):
                feat = safe_pattern_features(acc[s:e], 80.0, "Chapman_PD_OFF_RAW", subject, f"{subject}_off_w{j}")
                if feat is None:
                    continue
                rows.append(_add_label(feat, "impaired", "Chapman PD OFF raw windows"))
    return rows


def summarize_by_group_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, dataset), part in df.groupby(["group", "dataset"], sort=True):
        for feature in FEATURES:
            x = pd.to_numeric(part[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) == 0:
                continue
            rows.append(
                {
                    "group": group,
                    "dataset": dataset,
                    "feature": feature,
                    "n": int(len(x)),
                    "median": float(x.median()),
                    "iqr": float(x.quantile(0.75) - x.quantile(0.25)),
                    "p10": float(x.quantile(0.10)),
                    "p90": float(x.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def feature_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        part = df[["target", "dataset", feature]].copy()
        part[feature] = pd.to_numeric(part[feature], errors="coerce")
        part = part.replace([np.inf, -np.inf], np.nan).dropna()
        if part["target"].nunique() < 2 or len(part) < 20:
            continue
        normal = part[part["target"].eq(0)][feature]
        impaired = part[part["target"].eq(1)][feature]
        if len(normal) < 3 or len(impaired) < 3:
            continue
        raw_auc = roc_auc_score(part["target"], part[feature])
        auc_directional = max(raw_auc, 1.0 - raw_auc)
        direction = "higher_in_impaired" if raw_auc >= 0.5 else "lower_in_impaired"
        domain_dirs = []
        for dataset, dpart in part.groupby("dataset"):
            if dpart["target"].nunique() == 2:
                dn = dpart[dpart["target"].eq(0)][feature].median()
                di = dpart[dpart["target"].eq(1)][feature].median()
                domain_dirs.append(math.copysign(1, di - dn) if di != dn else 0)
        rows.append(
            {
                "feature": feature,
                "n": int(len(part)),
                "normal_n": int(len(normal)),
                "impaired_n": int(len(impaired)),
                "normal_median": float(normal.median()),
                "impaired_median": float(impaired.median()),
                "delta_impaired_minus_normal": float(impaired.median() - normal.median()),
                "auc_directional": float(auc_directional),
                "direction": direction,
                "within_dataset_direction_checks": len(domain_dirs),
                "within_dataset_direction_agree": int(sum(1 for d in domain_dirs if (d > 0 and direction == "higher_in_impaired") or (d < 0 and direction == "lower_in_impaired"))),
            }
        )
    return pd.DataFrame(rows).sort_values(["auc_directional", "feature"], ascending=[False, True])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(iter_existing_fixed_best10_table())
    rows.extend(iter_added_normal_datasets())
    rows.extend(iter_fog_star())
    rows.extend(iter_chapman_off_raw())
    rows.extend(iter_pd_turning())
    df = pd.DataFrame(rows)
    summary = summarize_by_group_dataset(df)
    effects = feature_effects(df)
    df.to_csv(OUT_DIR / "normal_impaired_pattern_features.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "normal_impaired_pattern_summary_by_group_dataset.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT_DIR / "normal_impaired_pattern_feature_effects.csv", index=False, encoding="utf-8-sig")
    print("counts")
    print(df.groupby(["group", "dataset"]).size().to_string())
    print("\nfeature effects")
    print(effects.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
