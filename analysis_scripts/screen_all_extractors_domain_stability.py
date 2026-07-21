from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "all_extractors_domain_stability_screen"


INPUTS = {
    "candidate_gait": {
        "tables": [
            ROOT / "analysis_outputs" / "smartphone_model_reanalysis" / "all_domain_candidate_feature_table.csv",
        ],
        "note": "candidate gait feature table, already combined",
    },
    "same_preprocessing_shape": {
        "tables": [
            ROOT / "analysis_outputs" / "physionet_labwalks_smartphone_shape_extractor_all_or" / "physionet_labwalks_shape_best10_all_or.csv",
            ROOT / "analysis_outputs" / "waveform_shape_feature_analysis" / "waveform_shape_features_same_preprocessing.csv",
            ROOT / "analysis_outputs" / "combined_fixed_shape6_sample_predictions" / "sample_best10_fixed_shape6_features.csv",
        ],
        "note": "shape extractor common columns across PhysioNet, public smartphone, and local sample",
    },
    "fixed_best10_quality": {
        "tables": [
            ROOT / "analysis_outputs" / "physionet_labwalks_smartphone_shape_extractor_all_or" / "physionet_labwalks_shape_best10_all_or.csv",
            ROOT / "analysis_outputs" / "fixed_best10_quality_pipeline" / "fixed_best10_public_features.csv",
            ROOT / "analysis_outputs" / "fixed_best10_quality_pipeline" / "fixed_best10_sample_features.csv",
        ],
        "note": "fixed best10 quality pipeline plus overlapping PhysioNet columns",
    },
}

ID_COLS = {
    "dataset",
    "label_group",
    "target",
    "source_id",
    "subject_id",
    "group_id",
    "sample_file",
    "path",
    "file",
    "base_feature_status",
    "observed_hz",
    "chunk_idx",
    "chunk_start_sec",
    "start_sec",
    "end_sec",
    "best10_start_sec",
    "best10_end_sec",
    "window_sec",
    "segment_idx",
    "quality_score",
}


def normalize_table(path: Path, extractor: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    if "sample_file" in df.columns and "dataset" not in df.columns:
        df["dataset"] = "OUR_SAMPLE"
    if "dataset" not in df.columns:
        df["dataset"] = "UNKNOWN"
    if "target" not in df.columns:
        df["target"] = np.nan
    if "label_group" not in df.columns:
        df["label_group"] = np.where(df["target"].eq(1), "impaired", "normal")
    if "subject_id" not in df.columns:
        if "sample_file" in df.columns:
            df["subject_id"] = df["sample_file"].astype(str)
        elif "source_id" in df.columns:
            df["subject_id"] = df["source_id"].astype(str)
        else:
            df["subject_id"] = np.arange(len(df)).astype(str)
    if "source_id" not in df.columns:
        if "sample_file" in df.columns:
            df["source_id"] = df["sample_file"].astype(str)
        else:
            df["source_id"] = df["subject_id"].astype(str)
    if "sample_file" in df.columns:
        df["dataset"] = df["dataset"].fillna("OUR_SAMPLE").replace({"": "OUR_SAMPLE"})
        df.loc[df["dataset"].astype(str).str.contains("OUR_SAMPLE|SAMPLE", case=False, na=False), "target"] = 0
        df.loc[df["dataset"].astype(str).str.contains("OUR_SAMPLE|SAMPLE", case=False, na=False), "label_group"] = "local_normal"
    df["extractor"] = extractor
    df["source_table"] = str(path)
    return df


def load_extractor(extractor: str, paths: list[Path]) -> pd.DataFrame:
    frames = [normalize_table(path, extractor) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError(extractor)
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    keep = [c for c in frames[0].columns if c in common]
    return pd.concat([frame[keep] for frame in frames], ignore_index=True, sort=False)


def numeric_features(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col in ID_COLS or col in {"extractor", "source_table"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 10:
            out.append(col)
    return out


def subject_level(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].fillna(df["source_id"]).astype(str)
    df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)
    rows = []
    for group_id, part in df.groupby("group_id", sort=True):
        target_values = pd.to_numeric(part["target"], errors="coerce").dropna().astype(int).unique()
        target = int(target_values[0]) if len(target_values) == 1 else np.nan
        row = {
            "extractor": part["extractor"].iloc[0],
            "dataset": part["dataset"].iloc[0],
            "subject_id": part["subject_id"].iloc[0],
            "group_id": group_id,
            "target": target,
            "n_rows": len(part),
        }
        for feature in features:
            row[feature] = pd.to_numeric(part[feature], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows)


def iqr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(float(np.sum(x > b)) for x in a)
    lt = sum(float(np.sum(x < b)) for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def screen_features(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    normal = table[table["target"].eq(0)].copy()
    impaired = table[table["target"].eq(1)].copy()
    rows = []
    normal_datasets = sorted(normal["dataset"].dropna().astype(str).unique())
    for feature in features:
        normal_values = pd.to_numeric(normal[feature], errors="coerce").dropna()
        impaired_values = pd.to_numeric(impaired[feature], errors="coerce").dropna()
        if len(normal_values) < 5 or len(impaired_values) < 5:
            continue
        pooled_iqr = iqr(normal_values)
        pooled_iqr = pooled_iqr if np.isfinite(pooled_iqr) and pooled_iqr > 1e-12 else float(normal_values.std(ddof=0) or 1.0)
        medians = {}
        counts = {}
        for dataset in normal_datasets:
            vals = pd.to_numeric(normal.loc[normal["dataset"].astype(str).eq(dataset), feature], errors="coerce").dropna()
            if len(vals):
                medians[dataset] = float(vals.median())
                counts[dataset] = int(len(vals))
        if len(medians) < 2:
            continue
        normal_pooled_median = float(normal_values.median())
        max_spread_iqr = max(abs(m - normal_pooled_median) / pooled_iqr for m in medians.values())
        pairwise = []
        ds_items = list(medians.items())
        for i in range(len(ds_items)):
            for j in range(i + 1, len(ds_items)):
                pairwise.append(abs(ds_items[i][1] - ds_items[j][1]) / pooled_iqr)
        max_pairwise_iqr = max(pairwise) if pairwise else np.nan
        try:
            groups = [
                pd.to_numeric(normal.loc[normal["dataset"].astype(str).eq(dataset), feature], errors="coerce").dropna()
                for dataset in normal_datasets
            ]
            groups = [g for g in groups if len(g) >= 2]
            kruskal_p = float(kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan
        except Exception:
            kruskal_p = np.nan
        try:
            label_p = float(mannwhitneyu(normal_values, impaired_values, alternative="two-sided").pvalue)
        except Exception:
            label_p = np.nan
        try:
            auc = float(roc_auc_score(
                np.r_[np.zeros(len(normal_values)), np.ones(len(impaired_values))],
                np.r_[normal_values.to_numpy(float), impaired_values.to_numpy(float)],
            ))
            label_auc_distance = abs(auc - 0.5) * 2.0
        except Exception:
            auc = np.nan
            label_auc_distance = np.nan
        cliff = cliffs_delta(impaired_values.to_numpy(float), normal_values.to_numpy(float))
        rows.append(
            {
                "feature": feature,
                "normal_n": int(len(normal_values)),
                "impaired_n": int(len(impaired_values)),
                "normal_dataset_count": len(medians),
                "normal_domains": "|".join(medians),
                "normal_domain_counts": "|".join(f"{k}:{v}" for k, v in counts.items()),
                "normal_median": normal_pooled_median,
                "impaired_median": float(impaired_values.median()),
                "impaired_minus_normal": float(impaired_values.median() - normal_pooled_median),
                "normal_iqr": pooled_iqr,
                "normal_domain_max_spread_iqr": float(max_spread_iqr),
                "normal_domain_max_pairwise_iqr": float(max_pairwise_iqr),
                "normal_domain_kruskal_p": kruskal_p,
                "label_mannwhitney_p": label_p,
                "label_auc": auc,
                "label_auc_distance": label_auc_distance,
                "cliffs_impaired_vs_normal": cliff,
                **{f"{dataset}_normal_median": med for dataset, med in medians.items()},
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["domain_stable_strict"] = (
        (out["normal_dataset_count"] >= 3)
        & (out["normal_domain_max_spread_iqr"] <= 1.0)
        & (out["normal_domain_max_pairwise_iqr"] <= 1.5)
    )
    out["domain_stable_loose"] = (
        (out["normal_dataset_count"] >= 3)
        & (out["normal_domain_max_spread_iqr"] <= 1.5)
        & (out["normal_domain_max_pairwise_iqr"] <= 2.0)
    )
    out["useful_label_effect"] = (out["label_auc_distance"] >= 0.25) | (out["cliffs_impaired_vs_normal"].abs() >= 0.30)
    out["selection_score"] = (
        out["label_auc_distance"].fillna(0)
        + out["cliffs_impaired_vs_normal"].abs().fillna(0)
        - 0.25 * out["normal_domain_max_spread_iqr"].fillna(10)
    )
    return out.sort_values(
        ["domain_stable_strict", "domain_stable_loose", "useful_label_effect", "selection_score"],
        ascending=[False, False, False, False],
    )


def greedy_corr_prune(table: pd.DataFrame, ranked: pd.DataFrame, max_features: int = 6, corr_limit: float = 0.75) -> pd.DataFrame:
    candidates = ranked[(ranked["domain_stable_loose"]) & (ranked["useful_label_effect"])].copy()
    selected = []
    rows = []
    for _, row in candidates.iterrows():
        feature = row["feature"]
        ok = True
        for existing in selected:
            corr = table[[feature, existing]].corr(method="spearman").iloc[0, 1]
            if pd.notna(corr) and abs(float(corr)) >= corr_limit:
                ok = False
                break
        rows.append(
            {
                "feature": feature,
                "selected": ok and len(selected) < max_features,
                "selection_order": len(selected) + 1 if ok and len(selected) < max_features else np.nan,
                "selection_score": row["selection_score"],
                "label_auc_distance": row["label_auc_distance"],
                "normal_domain_max_spread_iqr": row["normal_domain_max_spread_iqr"],
            }
        )
        if ok and len(selected) < max_features:
            selected.append(feature)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ranked = []
    all_selected = []
    metadata = []
    for extractor, spec in INPUTS.items():
        df = load_extractor(extractor, spec["tables"])
        features = numeric_features(df)
        table = subject_level(df, features)
        table.to_csv(OUT_DIR / f"{extractor}_subject_table.csv", index=False, encoding="utf-8-sig")
        ranked = screen_features(table, features)
        if ranked.empty:
            continue
        ranked.insert(0, "extractor", extractor)
        ranked.to_csv(OUT_DIR / f"{extractor}_feature_domain_screen.csv", index=False, encoding="utf-8-sig")
        selected = greedy_corr_prune(table, ranked.drop(columns=["extractor"]), max_features=6)
        selected.insert(0, "extractor", extractor)
        selected.to_csv(OUT_DIR / f"{extractor}_greedy_corr_pruned.csv", index=False, encoding="utf-8-sig")
        all_ranked.append(ranked)
        all_selected.append(selected)
        metadata.append(
            {
                "extractor": extractor,
                "note": spec["note"],
                "n_subject_rows": len(table),
                "n_features_screened": len(features),
                "target_counts": table["target"].value_counts(dropna=False).to_dict(),
                "dataset_counts": table["dataset"].value_counts(dropna=False).to_dict(),
            }
        )
    combined_ranked = pd.concat(all_ranked, ignore_index=True)
    combined_selected = pd.concat(all_selected, ignore_index=True)
    combined_ranked.to_csv(OUT_DIR / "all_extractors_feature_domain_screen.csv", index=False, encoding="utf-8-sig")
    combined_selected.to_csv(OUT_DIR / "all_extractors_greedy_corr_pruned.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata).to_csv(OUT_DIR / "all_extractors_screen_metadata.csv", index=False, encoding="utf-8-sig")
    print("TOP DOMAIN-STABLE + LABEL FEATURES")
    cols = [
        "extractor",
        "feature",
        "domain_stable_strict",
        "domain_stable_loose",
        "useful_label_effect",
        "selection_score",
        "label_auc_distance",
        "cliffs_impaired_vs_normal",
        "normal_domain_max_spread_iqr",
        "normal_domain_max_pairwise_iqr",
        "normal_domains",
    ]
    print(combined_ranked[cols].head(40).to_string(index=False))
    print("\nGREEDY SELECTED")
    print(combined_selected[combined_selected["selected"]].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
