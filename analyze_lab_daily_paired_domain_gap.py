from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
LAB_CSV = ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
DAILY_SUBJECT_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_fixed_model_subject_predictions.csv"
FINAL_OOF_CSV = ROOT / "final__2026" / "02_model" / "domain4_nested_oof_predictions.csv"
OUT_DIR = ROOT / "final__2026" / "06_lab_daily_domain_gap"

FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]


def bootstrap_mean_ci(values: pd.Series, seed: int = 20260713, n_boot: int = 10000) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(clean, size=(n_boot, clean.size), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def subject_aggregate(df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        valid = group.dropna(subset=FEATURES)
        if valid.empty:
            continue
        if aggregation == "best_window":
            idx = valid["base_v_stride_regularity"].idxmax()
            row = valid.loc[idx, FEATURES].to_dict()
        elif aggregation == "top10_regularity_median":
            cutoff = valid["base_v_stride_regularity"].quantile(0.90)
            selected = valid[valid["base_v_stride_regularity"] >= cutoff]
            row = selected[FEATURES].median(numeric_only=True).to_dict()
        elif aggregation == "all_window_median":
            row = valid[FEATURES].median(numeric_only=True).to_dict()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        row["subject_id"] = subject_id
        row["n_windows"] = int(valid.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gap_table(lab: pd.DataFrame, daily: pd.DataFrame, aggregation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab_a = subject_aggregate(lab, aggregation).add_prefix("lab_")
    lab_a = lab_a.rename(columns={"lab_subject_id": "subject_id"})

    daily_a = daily[daily["aggregation"] == aggregation].copy()
    daily_a = daily_a[["subject_id", "target", "clinical_group", "TUG", "DGI", "probability", "prediction", *FEATURES]]
    daily_a = daily_a.rename(columns={f: f"daily_{f}" for f in FEATURES})

    paired = lab_a.merge(daily_a, on="subject_id", how="inner")
    paired["aggregation"] = aggregation

    summaries = []
    for feat in FEATURES:
        lab_col = f"lab_{feat}"
        daily_col = f"daily_{feat}"
        diff = paired[daily_col] - paired[lab_col]
        ratio = paired[daily_col] / paired[lab_col].replace(0, np.nan)
        corr = paired[[lab_col, daily_col]].corr(method="spearman").iloc[0, 1]
        mean_ci_low, mean_ci_high = bootstrap_mean_ci(diff)
        t_p = stats.ttest_rel(paired[daily_col], paired[lab_col], nan_policy="omit").pvalue
        try:
            w_p = stats.wilcoxon(diff.dropna()).pvalue
        except ValueError:
            w_p = np.nan
        summaries.append(
            {
                "aggregation": aggregation,
                "feature": feat,
                "n_paired": int(diff.notna().sum()),
                "lab_mean": float(paired[lab_col].mean()),
                "daily_mean": float(paired[daily_col].mean()),
                "mean_diff_daily_minus_lab": float(diff.mean()),
                "mean_diff_95ci_low": mean_ci_low,
                "mean_diff_95ci_high": mean_ci_high,
                "median_diff_daily_minus_lab": float(diff.median()),
                "sd_diff": float(diff.std(ddof=1)),
                "iqr_diff": float(diff.quantile(0.75) - diff.quantile(0.25)),
                "mean_ratio_daily_over_lab": float(ratio.mean()),
                "median_ratio_daily_over_lab": float(ratio.median()),
                "spearman_lab_daily": float(corr) if pd.notna(corr) else np.nan,
                "paired_ttest_p": float(t_p) if pd.notna(t_p) else np.nan,
                "wilcoxon_p": float(w_p) if pd.notna(w_p) else np.nan,
            }
        )
        paired[f"diff_{feat}"] = diff
        paired[f"ratio_{feat}"] = ratio
    return paired, pd.DataFrame(summaries)


def performance_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (aggregation, cohort), g in daily.groupby(["aggregation", "cohort"], sort=True):
        y = g["target"].astype(int)
        prob = g["probability"].astype(float)
        pred = g["prediction"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "aggregation": aggregation,
                "cohort": cohort,
                "n": int(len(g)),
                "positive": int(y.sum()),
                "negative": int((1 - y).sum()),
                "auc": float(roc_auc_score(y, prob)),
                "accuracy": float(accuracy_score(y, pred)),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "f1": float(f1_score(y, pred)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(LAB_CSV)
    daily = pd.read_csv(DAILY_SUBJECT_CSV)
    final_subjects = set(pd.read_csv(FINAL_OOF_CSV)["subject_id"].dropna().unique())
    lab_final = lab[lab["subject_id"].isin(final_subjects)].copy()

    aggregations = ["best_window", "top10_regularity_median", "all_window_median"]
    paired_all = []
    summary_all = []
    for aggregation in aggregations:
        paired, summary = paired_gap_table(lab_final, daily[daily["cohort"] == "all_matched_valid"], aggregation)
        paired_all.append(paired)
        summary_all.append(summary)

    paired_df = pd.concat(paired_all, ignore_index=True)
    summary_df = pd.concat(summary_all, ignore_index=True)
    perf_df = performance_table(daily)

    lab_subjects = set(lab_final["subject_id"].dropna().unique())
    daily_subjects = set(daily[daily["cohort"] == "all_matched_valid"]["subject_id"].dropna().unique())
    overlap = sorted(lab_subjects & daily_subjects)
    raw_lab_subjects = set(lab["subject_id"].dropna().unique())

    paired_df.to_csv(OUT_DIR / "paired_lab_daily_subject_feature_gaps.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "paired_lab_daily_gap_summary.csv", index=False, encoding="utf-8-sig")
    perf_df.to_csv(OUT_DIR / "daily_fixed_model_performance_by_aggregation.csv", index=False, encoding="utf-8-sig")

    notes = {
        "analysis_type": "paired lab-to-daily domain gap diagnostic",
        "subject_scope": "restricted to the 67 subjects used in the final nested-CV lab model before matching to daily features",
        "leakage_policy": [
            "No label was used to estimate feature gaps.",
            "No model was retrained in this script.",
            "Do not treat subject-specific offsets estimated from the same evaluation subjects as deployable calibration unless a separate calibration protocol exists.",
            "Any performance-improving correction must be estimated inside training folds or on a separate calibration set.",
        ],
        "raw_lab_feature_subjects": len(raw_lab_subjects),
        "final_nested_lab_subjects": len(final_subjects),
        "final_nested_lab_subjects_present_in_lab_feature_file": len(lab_subjects),
        "daily_subjects_all_matched_valid": len(daily_subjects),
        "overlap_subjects": len(overlap),
        "final_lab_subjects_missing_in_daily": sorted(lab_subjects - daily_subjects),
        "daily_subjects_not_in_final_lab_model": sorted(daily_subjects - lab_subjects),
        "features": FEATURES,
        "outputs": {
            "paired_subject_gaps": str(OUT_DIR / "paired_lab_daily_subject_feature_gaps.csv"),
            "gap_summary": str(OUT_DIR / "paired_lab_daily_gap_summary.csv"),
            "daily_performance": str(OUT_DIR / "daily_fixed_model_performance_by_aggregation.csv"),
        },
    }
    (OUT_DIR / "analysis_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(notes, ensure_ascii=False, indent=2))
    print("\nGAP SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nDAILY PERFORMANCE")
    print(perf_df.to_string(index=False))


if __name__ == "__main__":
    main()
