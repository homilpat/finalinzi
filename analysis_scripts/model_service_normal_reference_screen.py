from __future__ import annotations

from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison" / "normal_impaired_pattern_features.csv"
SAMPLE_PATH = ROOT / "analysis_outputs" / "pattern4_domain_corrected_model" / "final_pattern4_no_turning_our_sample_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "service_normal_reference_model_screen"

FEATURES = [
    "sample_entropy",
    "acf_stride_peak",
    "acf_stride_peak_width_sec",
    "acf_stride_peak_sharpness",
    "dominant_peak_prominence",
    "spec_peak_ratio",
    "spec_entropy",
    "step_sec",
    "stride_sec",
    "sig_rms",
    "sig_iqr",
    "sig_range",
    "bandpower_mid_total_ratio",
    "jerk_entropy",
]

CORE_FEATURES = [
    "sample_entropy",
    "acf_stride_peak",
    "dominant_peak_prominence",
    "spec_peak_ratio",
    "spec_entropy",
    "step_sec",
]


def sample_label(subject_id: str) -> int:
    return 1 if "20260716" in str(subject_id) else 0


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    public = pd.read_csv(PUBLIC_PATH)
    public = public[public["dataset"].ne("PD_TURNING_IMU")].copy()
    public = public[~public["source_note"].astype(str).str.contains("existing fixed_best10_quality", na=False)].copy()
    public["_quality"] = pd.to_numeric(public["acf_stride_peak"], errors="coerce")
    idx = public.sort_values("_quality", ascending=False).groupby(["dataset", "subject_id"], sort=False).head(1).index
    public = public.loc[idx].reset_index(drop=True)
    public["target"] = public["target"].astype(int)
    public["group_id"] = public["dataset"].astype(str) + "::" + public["subject_id"].astype(str)

    sample = pd.read_csv(SAMPLE_PATH)
    sample["target"] = sample["subject_id"].map(sample_label).astype(int)
    sample["dataset"] = "OUR_SAMPLE_RAW"
    sample["group_id"] = sample["dataset"].astype(str) + "::" + sample["subject_id"].astype(str)
    return public, sample


def reference_transform(public: pd.DataFrame, sample: pd.DataFrame, features: list[str], mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    public = public.copy()
    sample = sample.copy()
    if mode == "none":
        return public, sample, {"mode": mode}

    public_norm = public[public["target"].eq(0)]
    sample_norm = sample[sample["target"].eq(0)]
    pub_med = public_norm[features].median()
    samp_med = sample_norm[features].median()
    if mode == "shift":
        delta = samp_med - pub_med
        public.loc[:, features] = public[features] + delta
        return public, sample, {"mode": mode, "delta": delta.to_dict()}

    if mode == "shift_scale":
        pub_iqr = (public_norm[features].quantile(0.75) - public_norm[features].quantile(0.25)).replace(0, np.nan)
        samp_iqr = (sample_norm[features].quantile(0.75) - sample_norm[features].quantile(0.25)).replace(0, np.nan)
        scale = (samp_iqr / pub_iqr).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 4.0)
        public.loc[:, features] = (public[features] - pub_med) * scale + samp_med
        return public, sample, {"mode": mode, "public_median": pub_med.to_dict(), "sample_median": samp_med.to_dict(), "scale": scale.to_dict()}

    raise ValueError(mode)


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


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


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


def oof_eval(df: pd.DataFrame, features: list[str], splitter: str) -> dict | None:
    data = df.dropna(subset=features + ["target"]).copy()
    if data["target"].nunique() < 2 or len(data) < 20:
        return None
    y = data["target"].to_numpy()
    if splitter == "group":
        cv = GroupKFold(n_splits=5)
        splits = cv.split(data[features], y, data["group_id"])
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260720)
        splits = cv.split(data[features], y)
    oof = np.zeros(len(data))
    for tr, te in splits:
        model = make_model()
        model.fit(data.iloc[tr][features], y[tr])
        train_prob = model.predict_proba(data.iloc[tr][features])[:, 1]
        thr = pick_threshold(y[tr], train_prob)
        oof[te] = model.predict_proba(data.iloc[te][features])[:, 1]
    return metrics(y, oof, pick_threshold(y, oof))


def sample_predictions(train: pd.DataFrame, sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = train.dropna(subset=features + ["target"]).copy()
    y = data["target"].to_numpy()
    model = make_model()
    model.fit(data[features], y)
    p_train = model.predict_proba(data[features])[:, 1]
    thr = pick_threshold(y, p_train)
    pred = sample.dropna(subset=features).copy()
    prob = model.predict_proba(pred[features])[:, 1]
    out = pred[["subject_id", "target", *features]].copy()
    out["probability"] = prob
    out["threshold"] = thr
    out["prediction"] = (prob >= thr).astype(int)
    out["correct"] = out["prediction"].eq(out["target"])
    return out


def feature_combos(public: pd.DataFrame) -> list[list[str]]:
    candidates = [f for f in FEATURES if f in public.columns]
    rows = []
    for k in range(2, 5):
        for combo in combinations(candidates, k):
            combo = list(combo)
            if not any(f in combo for f in CORE_FEATURES):
                continue
            corr = public[combo].corr(method="spearman").abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            max_corr = float(upper.max().max()) if k > 1 else 0.0
            if max_corr < 0.80:
                rows.append(combo)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    public, sample = load_tables()
    combos = feature_combos(public)
    rows = []
    sample_rows = []
    for mode in ["none", "shift", "shift_scale"]:
        for combo in combos:
            public_x, sample_x, params = reference_transform(public, sample, combo, mode)
            for splitter in ["group", "stratified"]:
                res = oof_eval(public_x, combo, splitter)
                if res is None:
                    continue
                sp = sample_predictions(public_x, sample_x, combo)
                sample_acc = float(sp["correct"].mean()) if len(sp) else np.nan
                row = {
                    "mode": mode,
                    "splitter": splitter,
                    "features": " + ".join(combo),
                    "k": len(combo),
                    "n": int(public_x.dropna(subset=combo + ["target"]).shape[0]),
                    "sample_acc": sample_acc,
                    "sample_correct": int(sp["correct"].sum()),
                    "sample_n": int(len(sp)),
                }
                row.update(res)
                rows.append(row)
                sp.insert(0, "mode", mode)
                sp.insert(1, "splitter", splitter)
                sp.insert(2, "features", " + ".join(combo))
                sample_rows.append(sp)

    results = pd.DataFrame(rows)
    results = results.sort_values(["sample_acc", "auc", "spec", "sens"], ascending=[False, False, False, False])
    sample_pred = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    results.to_csv(OUT_DIR / "service_normal_reference_model_screen.csv", index=False, encoding="utf-8-sig")
    sample_pred.to_csv(OUT_DIR / "service_normal_reference_sample_predictions.csv", index=False, encoding="utf-8-sig")
    public.to_csv(OUT_DIR / "public_subject_table_used.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(OUT_DIR / "sample_table_used.csv", index=False, encoding="utf-8-sig")

    best = results.head(30)
    print("public counts")
    print(public.groupby(["dataset", "target"]).size().to_string())
    print("\nsample labels")
    print(sample[["subject_id", "target"]].to_string(index=False))
    print("\nbest models")
    print(best[["mode", "splitter", "features", "n", "auc", "acc", "sens", "spec", "sample_acc", "sample_correct", "sample_n"]].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
