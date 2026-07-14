from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
DAILY_AMP = ROOT / "final__2026" / "01_preprocessing" / "daily75h_service10_amp_features.csv"
DAILY_BASE = Path(
    r"C:\Users\whdgu\Desktop\파이널 보행 프로젝트\physionet_AWS\strict_preprocessing_runs\strict_preprocessed\gait_features_strict_10s.csv"
)
CLINICAL = Path(r"C:\Users\whdgu\Desktop\파이널 보행 프로젝트\physionet_AWS\ClinicalDemogData_COFL.xlsx")
MODEL = ROOT / "final__2026" / "02_model" / "final_motor_domain4_labwalks10_logistic_C0p5_nested_youden.joblib"
OUT_DIR = ROOT / "final__2026" / "05_daily75h_validation"
FEATURES = ["v_amp_pool_median", "ml_amp_pool_iqr", "base_v_stride_regularity", "roll_amp_pool_iqr"]
EXCLUDED_TRAIN_NAN = {"CO024", "FL020"}


def normalize_subject_id(value: object) -> str:
    text = str(value).strip().upper().replace("-", "")
    return text


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def metric_values(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return {
        "auc": auc_score(y, prob),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, threshold: float, seed: int = 20260713, n_boot: int = 10000) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in ["auc", "accuracy", "sensitivity", "specificity", "f1"]}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals = metric_values(yy, prob[idx], threshold)
        for name in boot:
            boot[name].append(vals[name])
    return {
        name: (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
        for name, vals in boot.items()
    }


def load_clinical_labels() -> pd.DataFrame:
    rows = []
    for sheet, group in [("Controls", "Control"), ("Fallers", "Faller")]:
        df = pd.read_excel(CLINICAL, sheet_name=sheet)
        df["subject_id"] = df["#"].map(normalize_subject_id)
        df["clinical_group"] = group
        df["TUG"] = pd.to_numeric(df["TUG"], errors="coerce")
        df["DGI"] = pd.to_numeric(df["DGI"], errors="coerce")
        df["target"] = ((df["DGI"] <= 19) | (df["TUG"] >= 12)).astype(int)
        rows.append(df[["subject_id", "clinical_group", "TUG", "DGI", "target"]])
    return pd.concat(rows, ignore_index=True)


def aggregate_subject_features(windows: pd.DataFrame, mode: str) -> pd.DataFrame:
    valid = windows.dropna(subset=FEATURES).copy()
    if mode == "best_window":
        idx = valid.groupby("subject_id")["base_v_stride_regularity"].idxmax()
        out = valid.loc[idx, ["subject_id", *FEATURES, "n_windows"]].copy()
        out["aggregation"] = mode
        return out
    if mode == "all_window_median":
        out = valid.groupby("subject_id")[FEATURES].median().reset_index()
        out["aggregation"] = mode
        out = out.merge(valid.groupby("subject_id").size().rename("n_windows").reset_index(), on="subject_id", how="left")
        return out
    if mode == "top10_regularity_median":
        parts = []
        for subject_id, g in valid.groupby("subject_id"):
            cutoff = g["base_v_stride_regularity"].quantile(0.90)
            top = g[g["base_v_stride_regularity"] >= cutoff]
            row = top[FEATURES].median().to_dict()
            row["subject_id"] = subject_id
            row["aggregation"] = mode
            row["n_windows"] = len(g)
            row["n_top_windows"] = len(top)
            parts.append(row)
        return pd.DataFrame(parts)
    raise ValueError(mode)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    amp = pd.read_csv(DAILY_AMP)
    base_cols = ["subject_id", "start_sec", "end_sec", "v_stride_regularity", "stride_duration"]
    base = pd.read_csv(DAILY_BASE, usecols=base_cols)
    amp["subject_id"] = amp["subject_id"].map(normalize_subject_id)
    base["subject_id"] = base["subject_id"].map(normalize_subject_id)

    merged = amp.merge(base, on=["subject_id", "start_sec", "end_sec"], how="left")
    merged["base_v_stride_regularity"] = merged["v_stride_regularity"]
    n_by_subject = merged.groupby("subject_id").size().rename("n_windows").reset_index()
    merged = merged.merge(n_by_subject, on="subject_id", how="left")

    clinical = load_clinical_labels()
    model_data = joblib.load(MODEL)
    pipeline = model_data["pipeline"]
    threshold = float(model_data["threshold"])

    rows = []
    predictions = []
    for mode in ["best_window", "top10_regularity_median", "all_window_median"]:
        subj = aggregate_subject_features(merged, mode)
        subj = subj.merge(clinical, on="subject_id", how="inner")
        subj["in_labwalks_train_excluded_nan"] = subj["subject_id"].isin(EXCLUDED_TRAIN_NAN)
        for cohort_name, cohort in [
            ("all_matched_valid", subj),
            ("exclude_CO024_FL020", subj[~subj["in_labwalks_train_excluded_nan"]].copy()),
        ]:
            X = cohort[FEATURES]
            prob = pipeline.predict_proba(X)[:, 1]
            y = cohort["target"].to_numpy(dtype=int)
            vals = metric_values(y, prob, threshold)
            ci = bootstrap_ci(y, prob, threshold)
            row = {
                "aggregation": mode,
                "cohort": cohort_name,
                "n_subjects": len(cohort),
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                "threshold": threshold,
                **vals,
            }
            for metric, (lo, hi) in ci.items():
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
            rows.append(row)
            pred_df = cohort[["subject_id", "clinical_group", "TUG", "DGI", "target", "n_windows", *FEATURES]].copy()
            pred_df["aggregation"] = mode
            pred_df["cohort"] = cohort_name
            pred_df["probability"] = prob
            pred_df["prediction"] = (prob >= threshold).astype(int)
            predictions.append(pred_df)

    summary = pd.DataFrame(rows)
    pred_all = pd.concat(predictions, ignore_index=True)
    summary.to_csv(OUT_DIR / "daily75h_fixed_model_validation_summary.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(OUT_DIR / "daily75h_fixed_model_subject_predictions.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT_DIR / "daily75h_service10_model_windows_merged.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "daily75h_fixed_model_validation_notes.json").write_text(
        json.dumps(
            {
                "interpretation": "Same CO/FL cohort daily-life 72h data. Treat as free-living/domain-shift validation, not fully independent external validation.",
                "model": str(MODEL),
                "threshold": threshold,
                "features": FEATURES,
                "label_rule": "DGI <= 19 OR TUG >= 12",
                "bootstrap": "Subject-level bootstrap, 10000 resamples, seed=20260713.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(OUT_DIR)


if __name__ == "__main__":
    main()
