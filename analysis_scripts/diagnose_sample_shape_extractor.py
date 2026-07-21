from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "sample_shape_extractor_diagnostics"
SAMPLE_PRED = (
    ROOT
    / "analysis_outputs"
    / "combined_fixed_shape6_sample_predictions"
    / "sample_best10_fixed_shape6_features.csv"
)
PHYS = (
    ROOT
    / "analysis_outputs"
    / "physionet_labwalks_smartphone_shape_extractor_all_or"
    / "physionet_labwalks_shape_best10_all_or.csv"
)
SMART = ROOT / "analysis_outputs" / "waveform_shape_feature_analysis" / "waveform_shape_features_same_preprocessing.csv"

FEATURES = [
    "quality_score",
    "step_time_median",
    "stride_time_median",
    "cadence",
    "v_stride_regularity",
    "ap_stride_regularity",
    "v_stride_shape_cv_mean",
    "v_stride_shape_sd_mean",
    "v_stride_shape_corr_mean",
    "v_stride_shape_corr_sd",
    "v_peak_timing_sd_pct",
    "ml_spec_entropy",
]


def summarize_reference(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        if feature not in df.columns:
            continue
        x = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            continue
        rows.append(
            {
                "reference": name,
                "feature": feature,
                "n": len(x),
                "p01": x.quantile(0.01),
                "p05": x.quantile(0.05),
                "p10": x.quantile(0.10),
                "p25": x.quantile(0.25),
                "median": x.median(),
                "p75": x.quantile(0.75),
                "p90": x.quantile(0.90),
                "p95": x.quantile(0.95),
                "p99": x.quantile(0.99),
                "max": x.max(),
            }
        )
    return pd.DataFrame(rows)


def sample_percentiles(samples: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sample in samples.iterrows():
        for feature in FEATURES:
            if feature not in samples.columns:
                continue
            value = pd.to_numeric(pd.Series([sample[feature]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            for ref_name, ref_df in refs.groupby("reference"):
                source = ref_sources[ref_name]
                if feature not in source.columns:
                    continue
                x = pd.to_numeric(source[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(x) == 0:
                    continue
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                robust_z = (value - x.median()) / (iqr / 1.349) if iqr > 1e-12 else np.nan
                rows.append(
                    {
                        "sample_file": sample["sample_file"],
                        "feature": feature,
                        "value": value,
                        "reference": ref_name,
                        "ref_median": x.median(),
                        "ref_p25": q1,
                        "ref_p75": q3,
                        "percentile": float((x <= value).mean() * 100),
                        "robust_z": robust_z,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SAMPLE_PRED)
    phys = pd.read_csv(PHYS)
    phys = phys[phys["target"].notna()].copy()
    smart = pd.read_csv(SMART)
    smart_normal = smart[smart["target"].eq(0)].copy()
    smart_impaired = smart[smart["target"].eq(1)].copy()
    phys_normal = phys[phys["target"].eq(0)].copy()
    phys_impaired = phys[phys["target"].eq(1)].copy()

    global ref_sources
    ref_sources = {
        "PhysioNet_normal": phys_normal,
        "PhysioNet_impaired": phys_impaired,
        "Smartphone_normal": smart_normal,
        "Smartphone_impaired": smart_impaired,
    }
    summaries = pd.concat(
        [summarize_reference(df, name) for name, df in ref_sources.items()],
        ignore_index=True,
    )
    pct = sample_percentiles(samples, summaries)

    summaries.to_csv(OUT_DIR / "reference_feature_distributions.csv", index=False, encoding="utf-8-sig")
    pct.to_csv(OUT_DIR / "sample_feature_percentiles_vs_references.csv", index=False, encoding="utf-8-sig")

    focus = pct[pct["feature"].isin(["v_stride_shape_cv_mean", "v_stride_regularity", "ap_stride_regularity", "step_time_median", "stride_time_median", "quality_score"])]
    print("SAMPLE BEST10")
    print(samples[["sample_file", "start_sec", "quality_score", "step_time_median", "stride_time_median", "v_stride_regularity", "ap_stride_regularity", "v_stride_shape_cv_mean", "v_stride_shape_sd_mean", "v_stride_shape_corr_mean"]].to_string(index=False))
    print("\nFOCUS PERCENTILES")
    print(focus.sort_values(["sample_file", "feature", "reference"]).to_string(index=False))
    print("\nREFERENCE v_stride_shape_cv_mean")
    print(summaries[summaries["feature"].eq("v_stride_shape_cv_mean")].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
