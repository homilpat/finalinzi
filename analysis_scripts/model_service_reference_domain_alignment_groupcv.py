from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
PATTERN_PATH = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison" / "normal_impaired_pattern_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "service_reference_domain_alignment_groupcv"

FEATURE_SETS = {
    "acf_step": ["acf_stride_peak", "step_sec"],
    "acf_step_sample_entropy": ["acf_stride_peak", "step_sec", "sample_entropy"],
    "acf_step_harmonic": ["acf_stride_peak", "step_sec", "harmonic_ratio"],
    "acf_width_step_harmonic": ["acf_stride_peak", "acf_stride_peak_width_sec", "step_sec", "harmonic_ratio"],
    "entropy_acf_peakratio": ["sample_entropy", "acf_stride_peak", "spec_peak_ratio", "dominant_peak_prominence"],
    "acf_step_entropy": ["acf_stride_peak", "step_sec", "spec_entropy"],
    "acf_width_step": ["acf_stride_peak", "acf_stride_peak_width_sec", "step_sec"],
}


def load_subject_table() -> pd.DataFrame:
    df = pd.read_csv(PATTERN_PATH)
    df = df[df["dataset"].ne("PD_TURNING_IMU")].copy()
    df = df[~df["source_note"].astype(str).str.contains("existing fixed_best10_quality", na=False)].copy()
    df["_quality"] = pd.to_numeric(df["acf_stride_peak"], errors="coerce")
    idx = df.sort_values("_quality", ascending=False).groupby(["dataset", "subject_id"], sort=False).head(1).index
    subj = df.loc[idx].copy().reset_index(drop=True)
    subj["group_id"] = subj["dataset"].astype(str) + "::" + subj["subject_id"].astype(str)
    return subj


def robust_domain_params(train: pd.DataFrame, features: list[str], reference_mode: str) -> dict:
    normals = train[train["target"].eq(0)].copy()
    if reference_mode == "pooled_normals":
        ref_med = normals[features].median()
        ref_iqr = normals[features].quantile(0.75) - normals[features].quantile(0.25)
    elif reference_mode == "service_like_normals":
        ref_domains = {"UCI_HAR", "GEOTEC_SP"}
        ref = normals[normals["dataset"].isin(ref_domains)]
        if ref.empty:
            ref = normals
        ref_med = ref[features].median()
        ref_iqr = ref[features].quantile(0.75) - ref[features].quantile(0.25)
    else:
        raise ValueError(reference_mode)
    ref_iqr = ref_iqr.replace(0, np.nan)
    params = {}
    for dataset, part in train.groupby("dataset"):
        norm_part = normals[normals["dataset"].eq(dataset)]
        basis = norm_part if len(norm_part) >= 5 else part
        med = basis[features].median()
        iqr = (basis[features].quantile(0.75) - basis[features].quantile(0.25)).replace(0, np.nan)
        scale = (ref_iqr / iqr).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 4.0)
        shift = ref_med - med * scale
        params[dataset] = {"scale": scale.to_dict(), "shift": shift.to_dict()}
    return params


def apply_params(df: pd.DataFrame, features: list[str], params: dict) -> pd.DataFrame:
    out = df.copy()
    for dataset, p in params.items():
        mask = out["dataset"].eq(dataset)
        for feature in features:
            out.loc[mask, feature] = out.loc[mask, feature] * float(p["scale"].get(feature, 1.0)) + float(p["shift"].get(feature, 0.0))
    return out


def pick_threshold(y: np.ndarray, prob: np.ndarray, min_sens: float = 0.80) -> float:
    best_thr = 0.5
    best_spec = -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= min_sens and spec > best_spec:
            best_thr, best_spec = float(thr), spec
    return best_thr


def eval_oof(df: pd.DataFrame, features: list[str], correction: str, splitter: str) -> dict:
    data = df.dropna(subset=features + ["target"]).copy()
    y = data["target"].astype(int).to_numpy()
    groups = data["group_id"].astype(str).to_numpy()
    if splitter == "group":
        cv = GroupKFold(n_splits=5)
        splits = cv.split(data[features], y, groups)
    elif splitter == "stratified":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260719)
        splits = cv.split(data[features], y)
    else:
        raise ValueError(splitter)
    oof = np.zeros(len(data), dtype=float)
    for tr, te in splits:
        train = data.iloc[tr].copy()
        test = data.iloc[te].copy()
        if correction != "none":
            params = robust_domain_params(train, features, correction)
            train = apply_params(train, features, params)
            test = apply_params(test, features, params)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        )
        pipe.fit(train[features], y[tr])
        oof[te] = pipe.predict_proba(test[features])[:, 1]
    thr = pick_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
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


def eval_leave_one_domain(df: pd.DataFrame, features: list[str], correction: str) -> pd.DataFrame:
    data = df.dropna(subset=features + ["target"]).copy()
    rows = []
    for heldout in sorted(data["dataset"].unique()):
        train = data[data["dataset"].ne(heldout)].copy()
        test = data[data["dataset"].eq(heldout)].copy()
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            # AUC is undefined when a held-out domain only has one label.
            continue
        y_train = train["target"].astype(int).to_numpy()
        y_test = test["target"].astype(int).to_numpy()
        if correction != "none":
            params = robust_domain_params(train, features, correction)
            train = apply_params(train, features, params)
            test = apply_params(test, features, params)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        )
        pipe.fit(train[features], y_train)
        p_train = pipe.predict_proba(train[features])[:, 1]
        p_test = pipe.predict_proba(test[features])[:, 1]
        thr = pick_threshold(y_train, p_train)
        pred = (p_test >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "heldout_domain": heldout,
                "n": int(len(test)),
                "normal_n": int((y_test == 0).sum()),
                "impaired_n": int((y_test == 1).sum()),
                "auc": float(roc_auc_score(y_test, p_test)),
                "acc": float(accuracy_score(y_test, pred)),
                "sens": float(tp / (tp + fn)) if tp + fn else np.nan,
                "spec": float(tn / (tn + fp)) if tn + fp else np.nan,
                "f1": float(f1_score(y_test, pred)),
                "threshold": float(thr),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_subject_table()
    rows = []
    lodo_rows = []
    for name, features in FEATURE_SETS.items():
        for correction in ["none", "pooled_normals", "service_like_normals"]:
            for splitter in ["stratified", "group"]:
                res = {
                    "feature_set": name,
                    "features": " + ".join(features),
                    "correction": correction,
                    "splitter": splitter,
                }
                res.update(eval_oof(df, features, correction, splitter))
                rows.append(res)
            lodo = eval_leave_one_domain(df, features, correction)
            if not lodo.empty:
                for _, lrow in lodo.iterrows():
                    item = {
                        "feature_set": name,
                        "features": " + ".join(features),
                        "correction": correction,
                    }
                    item.update(lrow.to_dict())
                    lodo_rows.append(item)
    out = pd.DataFrame(rows).sort_values(["splitter", "auc"], ascending=[True, False])
    lodo_out = pd.DataFrame(lodo_rows)
    if not lodo_out.empty:
        lodo_summary = (
            lodo_out.groupby(["feature_set", "features", "correction"], as_index=False)
            .agg(
                domains=("heldout_domain", "nunique"),
                mean_auc=("auc", "mean"),
                min_auc=("auc", "min"),
                mean_acc=("acc", "mean"),
                mean_sens=("sens", "mean"),
                mean_spec=("spec", "mean"),
            )
            .sort_values(["mean_auc", "min_auc"], ascending=[False, False])
        )
    else:
        lodo_summary = pd.DataFrame()
    out.to_csv(OUT_DIR / "domain_alignment_groupcv_results.csv", index=False, encoding="utf-8-sig")
    lodo_out.to_csv(OUT_DIR / "leave_one_domain_results.csv", index=False, encoding="utf-8-sig")
    lodo_summary.to_csv(OUT_DIR / "leave_one_domain_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "subject_table.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("\nleave-one-domain summary")
    print(lodo_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
