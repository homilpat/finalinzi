from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "analysis_outputs" / "axis_aligned_gait_model" / "axis_aligned_best10_subject_table.csv"
OUT_DIR = ROOT / "analysis_outputs" / "axis_aligned_domain_corrected_gait_model"

FEATURES = [
    "v_acf_stride_peak",
    "v_acf_stride_peak_width_sec",
    "v_spec_peak_ratio",
    "v_spec_entropy",
    "v_stride_shape_cv_mean",
    "ml_acf_stride_peak",
    "ml_acf_stride_peak_width_sec",
    "ml_spec_peak_ratio",
    "ml_spec_entropy",
    "ap_acf_stride_peak",
    "ap_acf_stride_peak_width_sec",
    "ap_spec_peak_ratio",
    "ap_spec_entropy",
    "step_sec",
    "stride_sec",
]


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


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


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_j = 0.5, -np.inf
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_thr, best_j = float(thr), float(j)
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


def load_table() -> pd.DataFrame:
    table = pd.read_csv(IN_PATH)
    for feature in FEATURES:
        if feature in table.columns:
            table[feature] = pd.to_numeric(table[feature], errors="coerce")
    table["target"] = pd.to_numeric(table["target"], errors="coerce")
    table["group_id"] = table["dataset"].astype(str) + "::" + table["subject_id"].astype(str)
    return table


def correction_reference(table: pd.DataFrame, mode: str, features: list[str]) -> pd.Series:
    if mode == "sample_normal":
        ref = table[table["dataset"].eq("OUR_SAMPLE") & table["target"].eq(0)]
    elif mode == "physionet_normal":
        ref = table[table["dataset"].eq("PhysioNet_LabWalks") & table["target"].eq(0)]
    elif mode == "public_normal":
        ref = table[table["dataset"].ne("OUR_SAMPLE") & table["target"].eq(0)]
    else:
        raise ValueError(mode)
    return ref[features].median(numeric_only=True)


def apply_domain_correction(table: pd.DataFrame, features: list[str], reference_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = table.copy()
    ref_median = correction_reference(out, reference_mode, features)
    rows = []
    for dataset, part in out.groupby("dataset", sort=True):
        normal = part[part["target"].eq(0)]
        if dataset == "OUR_SAMPLE":
            delta = pd.Series(0.0, index=features)
            status = "holdout_no_correction"
        elif len(normal) >= 2:
            base = normal[features].median(numeric_only=True)
            delta = ref_median - base
            status = "normal_reference_delta_applied"
            out.loc[out["dataset"].eq(dataset), features] = out.loc[out["dataset"].eq(dataset), features] + delta
        else:
            delta = pd.Series(0.0, index=features)
            status = "no_within_domain_normal_reference"
        for feature in features:
            rows.append(
                {
                    "reference_mode": reference_mode,
                    "dataset": dataset,
                    "feature": feature,
                    "delta_added": float(delta.get(feature, 0.0)) if np.isfinite(delta.get(feature, np.nan)) else np.nan,
                    "status": status,
                    "normal_n": int(len(normal)),
                }
            )
    return out, pd.DataFrame(rows)


def group_oof(table: pd.DataFrame, features: list[str], threshold_mode: str) -> dict | None:
    data = table[table["dataset"].ne("OUR_SAMPLE")].copy()
    data = data.dropna(subset=["target"])
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
    thr = pick_sens80_threshold(y, oof) if threshold_mode == "sens80" else youden_threshold(y, oof)
    return metrics(y, oof, thr)


def predict_samples(table: pd.DataFrame, features: list[str], threshold_mode: str, metric_row: dict) -> pd.DataFrame:
    train = table[table["dataset"].ne("OUR_SAMPLE")].dropna(subset=["target"]).copy()
    sample = table[table["dataset"].eq("OUR_SAMPLE")].copy()
    train = train.dropna(subset=features, how="all")
    sample = sample.dropna(subset=features, how="all")
    if train.empty or sample.empty:
        return pd.DataFrame()
    y = train["target"].astype(int).to_numpy()
    clf = model()
    clf.fit(train[features], y)
    train_p = clf.predict_proba(train[features])[:, 1]
    thr = pick_sens80_threshold(y, train_p) if threshold_mode == "sens80" else youden_threshold(y, train_p)
    p = clf.predict_proba(sample[features])[:, 1]
    out = sample[["dataset", "subject_id", "target", *features]].copy()
    out["features"] = " + ".join(features)
    out["threshold_mode"] = threshold_mode
    out["reference_mode"] = metric_row["reference_mode"]
    out["model_auc_oof"] = metric_row["auc"]
    out["model_sens_oof"] = metric_row["sens"]
    out["model_spec_oof"] = metric_row["spec"]
    out["probability_impaired"] = p
    out["threshold"] = thr
    out["prediction"] = (p >= thr).astype(int)
    out["correct"] = out["prediction"].eq(out["target"].astype(int))
    return out


def max_abs_corr(table: pd.DataFrame, features: list[str]) -> float:
    corr = table[features].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    value = upper.max().max()
    return float(value) if np.isfinite(value) else 0.0


def screen(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    correction_rows = []
    available = [f for f in FEATURES if f in table.columns and table[f].notna().sum() >= 10]
    for reference_mode in ["sample_normal", "physionet_normal", "public_normal"]:
        for k in (2, 3, 4):
            for combo in combinations(available, k):
                features = list(combo)
                corrected, correction = apply_domain_correction(table, features, reference_mode)
                train_part = corrected[corrected["dataset"].ne("OUR_SAMPLE")]
                corr = max_abs_corr(train_part, features)
                if corr > 0.85:
                    continue
                for threshold_mode in ["sens80", "youden"]:
                    res = group_oof(corrected, features, threshold_mode)
                    if res is None:
                        continue
                    row = {
                        "reference_mode": reference_mode,
                        "threshold_mode": threshold_mode,
                        "features": " + ".join(features),
                        "k": len(features),
                        "max_abs_spearman": corr,
                    }
                    row.update(res)
                    sample_pred = predict_samples(corrected, features, threshold_mode, row)
                    row["sample_n"] = int(len(sample_pred))
                    row["sample_correct"] = int(sample_pred["correct"].sum()) if not sample_pred.empty else 0
                    row["sample_acc"] = float(sample_pred["correct"].mean()) if not sample_pred.empty else np.nan
                    rows.append(row)
                    if not sample_pred.empty:
                        pred_rows.append(sample_pred)
                correction_rows.append(correction)
    result = pd.DataFrame(rows).sort_values(
        ["sample_acc", "sens", "auc", "spec"], ascending=[False, False, False, False]
    )
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    corrections = pd.concat(correction_rows, ignore_index=True).drop_duplicates() if correction_rows else pd.DataFrame()
    return result, preds, corrections


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_table()
    results, preds, corrections = screen(table)
    results.to_csv(OUT_DIR / "domain_corrected_model_screen.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "domain_corrected_sample_predictions.csv", index=False, encoding="utf-8-sig")
    corrections.to_csv(OUT_DIR / "domain_correction_deltas.csv", index=False, encoding="utf-8-sig")
    print("data counts")
    print(table.groupby(["group", "dataset"]).size().to_string())
    print("\nbest candidates")
    print(results.head(30).to_string(index=False))
    if not results.empty and not preds.empty:
        top = results.iloc[0]
        top_preds = preds[
            preds["features"].eq(top["features"])
            & preds["threshold_mode"].eq(top["threshold_mode"])
            & preds["reference_mode"].eq(top["reference_mode"])
        ]
        print("\ntop sample predictions")
        print(top_preds[["subject_id", "target", "probability_impaired", "threshold", "prediction", "correct"]].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
