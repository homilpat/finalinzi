from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
PATTERN_PATH = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison" / "normal_impaired_pattern_features.csv"
SAMPLE_PATH = ROOT / "analysis_outputs" / "pattern4_domain_corrected_model" / "final_pattern4_no_turning_our_sample_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "service_reference_pattern_feature_screen"

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

LITERATURE_BASIS = {
    "step_sec": "spatiotemporal gait timing; step time",
    "stride_sec": "spatiotemporal gait timing; stride time",
    "cadence": "spatiotemporal gait timing; cadence",
    "acf_step_peak": "autocorrelation gait regularity at step lag",
    "acf_stride_peak": "autocorrelation gait regularity at stride lag",
    "acf_stride_peak_width_sec": "autocorrelation peak width / rhythm consistency",
    "spec_peak_freq": "dominant gait frequency from spectrum",
    "spec_peak_ratio": "dominant spectral peak concentration / harmonic structure",
    "spec_entropy": "spectral entropy; gait signal complexity",
    "stride_shape_cv_mean": "stride-normalized waveform variability",
    "peak_timing_sd_pct": "stride-normalized peak timing variability",
    "step_stride_regularity_asymmetry": "step-to-stride autocorrelation regularity asymmetry",
    "sample_entropy": "sample entropy; signal regularity/complexity",
    "sig_rms": "bandpass-filtered acceleration magnitude RMS",
    "sig_iqr": "bandpass-filtered acceleration magnitude interquartile range",
    "sig_range": "bandpass-filtered acceleration magnitude 5-95 percentile range",
    "acf_stride_step_ratio": "stride-to-step autocorrelation peak ratio",
    "acf_stride_peak_sharpness": "stride autocorrelation peak height divided by peak width",
    "acf_decay_1s": "autocorrelation value at 1 second lag",
    "harmonic_ratio": "harmonic structure of gait acceleration spectrum",
    "spectral_flatness": "spectral flatness; tone-like versus noise-like gait rhythm",
    "dominant_peak_prominence": "dominant gait frequency peak prominence",
    "low_high_band_power_ratio": "low/high gait-band spectral power ratio",
    "bandpower_0p6_1p2": "low gait-band spectral power",
    "bandpower_1p2_2p0": "mid gait-band spectral power",
    "bandpower_2p0_3p0": "high gait-band spectral power",
    "bandpower_mid_total_ratio": "mid gait-band power divided by total gait-band power",
    "jerk_rms": "jerk smoothness from acceleration derivative",
    "jerk_entropy": "spectral entropy of jerk signal",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PATTERN_PATH)
    df = df[df["dataset"].ne("PD_TURNING_IMU")].copy()
    df = df[~df["source_note"].astype(str).str.contains("existing fixed_best10_quality", na=False)].copy()
    sample = pd.read_csv(SAMPLE_PATH)
    sample["dataset"] = "OUR_SAMPLE_RAW"
    return df, sample


def robust_distance(sample: pd.Series, normal: pd.Series) -> float:
    iqr = normal.quantile(0.75) - normal.quantile(0.25)
    scale = float(iqr) if np.isfinite(iqr) and iqr > 1e-9 else float(normal.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return float(abs(sample.median() - normal.median()) / scale)


def feature_screen(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    public_normal = df[df["target"].eq(0)]
    impaired = df[df["target"].eq(1)]
    for feature in FEATURES:
        part = df[["target", "dataset", feature]].copy()
        part[feature] = pd.to_numeric(part[feature], errors="coerce")
        part = part.replace([np.inf, -np.inf], np.nan).dropna()
        sx = pd.to_numeric(sample[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        nx = pd.to_numeric(public_normal[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ix = pd.to_numeric(impaired[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(part) < 30 or len(sx) == 0 or len(nx) < 5 or len(ix) < 5 or part["target"].nunique() < 2:
            continue
        auc = roc_auc_score(part["target"], part[feature])
        direction = "higher_in_impaired" if auc >= 0.5 else "lower_in_impaired"
        sep_auc = max(auc, 1.0 - auc)
        sample_to_normal = robust_distance(sx, nx)
        sample_to_impaired = robust_distance(sx, ix)
        domain_medians = []
        for dataset, dpart in public_normal.groupby("dataset"):
            x = pd.to_numeric(dpart[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) >= 2:
                domain_medians.append(float(x.median()))
        domain_iqr = float(pd.Series(domain_medians).quantile(0.75) - pd.Series(domain_medians).quantile(0.25)) if len(domain_medians) >= 3 else np.nan
        rows.append(
            {
                "feature": feature,
                "literature_basis": LITERATURE_BASIS[feature],
                "n": int(len(part)),
                "sample_n": int(len(sx)),
                "normal_n": int(len(nx)),
                "impaired_n": int(len(ix)),
                "sample_median": float(sx.median()),
                "public_normal_median": float(nx.median()),
                "impaired_median": float(ix.median()),
                "sample_to_normal_robust_dist": sample_to_normal,
                "sample_to_impaired_robust_dist": sample_to_impaired,
                "normal_domain_median_iqr": domain_iqr,
                "separation_auc_directional": float(sep_auc),
                "direction": direction,
                "service_usable_score": float(sep_auc - 0.20 * sample_to_normal - 0.05 * (domain_iqr if np.isfinite(domain_iqr) else 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("service_usable_score", ascending=False)


def subject_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.dropna(subset=["target"]).copy()
    keep["_quality"] = pd.to_numeric(keep["acf_stride_peak"], errors="coerce")
    idx = keep.sort_values("_quality", ascending=False).groupby(["dataset", "subject_id"], sort=False).head(1).index
    return keep.loc[idx, ["dataset", "subject_id", "target", *FEATURES]].reset_index(drop=True)


def pick_threshold(y: np.ndarray, p: np.ndarray, min_sens: float = 0.80) -> float:
    best = (0.5, -1.0)
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= min_sens and spec > best[1]:
            best = (float(thr), spec)
    return best[0]


def cv_combo(subj: pd.DataFrame, features: list[str]) -> dict | None:
    data = subj.dropna(subset=["target"]).copy()
    finite_rows = data[features].notna().sum(axis=1) >= len(features)
    data = data[finite_rows]
    if len(data) < 30 or data["target"].nunique() < 2:
        return None
    y = data["target"].astype(int).to_numpy()
    if min(np.bincount(y)) < 5:
        return None
    oof = np.zeros(len(data))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260719)
    for tr, te in skf.split(data[features], y):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        )
        pipe.fit(data.iloc[tr][features], y[tr])
        oof[te] = pipe.predict_proba(data.iloc[te][features])[:, 1]
    thr = pick_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "features": " + ".join(features),
        "k": len(features),
        "n": int(len(data)),
        "normal_n": int((y == 0).sum()),
        "impaired_n": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, oof)),
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


def combo_screen(df: pd.DataFrame, screened: pd.DataFrame) -> pd.DataFrame:
    subj = subject_table(df)
    # Keep features that are reasonably close to service samples and have at least weak separation.
    candidates = screened[
        (screened["sample_to_normal_robust_dist"] <= 1.5)
        & (screened["separation_auc_directional"] >= 0.58)
    ]["feature"].tolist()
    rows = []
    for k in range(1, min(4, len(candidates)) + 1):
        for combo in combinations(candidates, k):
            corr = subj[list(combo)].corr(method="spearman").abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            max_corr = float(upper.max().max()) if len(combo) > 1 else 0.0
            if max_corr >= 0.80:
                continue
            res = cv_combo(subj, list(combo))
            if res:
                res["max_spearman_abs_corr"] = max_corr
                rows.append(res)
    return pd.DataFrame(rows).sort_values(["auc", "spec", "sens"], ascending=[False, False, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, sample = load_data()
    screened = feature_screen(df, sample)
    combos = combo_screen(df, screened)
    screened.to_csv(OUT_DIR / "service_reference_feature_screen.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(OUT_DIR / "service_reference_combo_cv.csv", index=False, encoding="utf-8-sig")
    print("feature screen")
    print(screened.to_string(index=False))
    print("\ncombo screen")
    print(combos.head(30).to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
