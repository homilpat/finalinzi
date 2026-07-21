from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from compare_normal_gait_pattern_features import pattern_features


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison" / "normal_impaired_pattern_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "pattern4_domain_corrected_model"

FEATURES = [
    "spec_peak_ratio",
    "acf_stride_peak",
    "spec_entropy",
]


def read_hazi_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Timestamp" in line and "," in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no CSV header found: {path}")
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(lines[header_idx:])))


def best10_features(acc: np.ndarray, fs: float, dataset: str, subject_id: str, source_id: str) -> dict | None:
    win = int(round(10 * fs))
    if len(acc) < win:
        return None
    best = None
    for start in range(0, len(acc) - win + 1, max(1, int(round(1 * fs)))):
        feat = pattern_features(acc[start : start + win], fs, dataset, subject_id, f"{source_id}_start{start / fs:.1f}s")
        score = feat.get("acf_stride_peak", np.nan)
        if np.isfinite(score) and (best is None or score > best.get("acf_stride_peak", -np.inf)):
            best = feat
    return best


def sample_rows() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "보행SAMPLE").glob("*.csv")):
        try:
            df = read_hazi_csv(path)
        except Exception:
            continue
        if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(df.columns):
            acc = df[["Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
        elif {"Acc_X", "Acc_Y", "Acc_Z"}.issubset(df.columns):
            # Raw calibrated sample before anatomical columns existed.
            acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float) / 9.80665
        else:
            continue
        if len(acc) < 300:
            continue
        fs = 100.0
        feat = best10_features(acc, fs, "OUR_SAMPLE_RAW", path.stem, path.name)
        if feat is None:
            continue
        feat["group"] = "normal"
        feat["target"] = 0
        feat["source_note"] = "local raw sample, held out from training"
        rows.append(feat)
    return rows


def subject_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(IN_PATH)
    raw_only = df[~df["source_note"].astype(str).str.contains("existing fixed_best10_quality", na=False)].copy()
    raw_only = raw_only[raw_only["dataset"].ne("PD_TURNING_IMU")].copy()
    raw_only = raw_only.dropna(subset=FEATURES + ["target"])
    raw_only["_quality"] = pd.to_numeric(raw_only["acf_stride_peak"], errors="coerce")
    idx = raw_only.sort_values("_quality", ascending=False).groupby(["dataset", "subject_id"], sort=False).head(1).index
    subj = raw_only.loc[idx, ["dataset", "subject_id", "group", "target", *FEATURES]].reset_index(drop=True)
    samples = pd.DataFrame(sample_rows())
    if not samples.empty:
        samples = samples.dropna(subset=FEATURES)
    return subj, samples


def correction_params(train: pd.DataFrame, reference: pd.Series | None = None) -> dict:
    if reference is None:
        normals = train[train["target"].eq(0)]
        ref = normals[FEATURES].median()
    else:
        ref = reference[FEATURES]
    params = {}
    for dataset, part in train.groupby("dataset"):
        med = part[FEATURES].median()
        params[dataset] = (ref - med).to_dict()
    return params


def apply_correction(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    for dataset, delta in params.items():
        mask = out["dataset"].eq(dataset)
        for feature in FEATURES:
            out.loc[mask, feature] = out.loc[mask, feature] + float(delta.get(feature, 0.0))
    return out


def pick_threshold(y_true: np.ndarray, prob: np.ndarray, min_sens: float = 0.80) -> float:
    best_thr = 0.5
    best_spec = -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_thr = float(thr)
    return best_thr


def metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob),
        "acc": accuracy_score(y_true, pred),
        "sens": tp / (tp + fn) if (tp + fn) else np.nan,
        "spec": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(thr),
    }


def cv_eval(subj: pd.DataFrame, use_domain_correction: bool) -> dict:
    y = subj["target"].astype(int).to_numpy()
    oof = np.zeros(len(subj), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260719)
    for tr, te in skf.split(subj[FEATURES], y):
        train = subj.iloc[tr].copy()
        test = subj.iloc[te].copy()
        if use_domain_correction:
            params = correction_params(train)
            train = apply_correction(train, params)
            test = apply_correction(test, params)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        )
        pipe.fit(train[FEATURES], y[tr])
        oof[te] = pipe.predict_proba(test[FEATURES])[:, 1]
    thr = pick_threshold(y, oof, 0.80)
    return {"mode": "domain_corrected" if use_domain_correction else "global_robust", **metrics(y, oof, thr)}


def fit_final(subj: pd.DataFrame, samples: pd.DataFrame, use_domain_correction: bool) -> tuple[Pipeline, dict, pd.DataFrame]:
    train = subj.copy()
    params = correction_params(train) if use_domain_correction else {}
    if use_domain_correction:
        train = apply_correction(train, params)
    y = train["target"].astype(int).to_numpy()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )
    pipe.fit(train[FEATURES], y)
    train_prob = pipe.predict_proba(train[FEATURES])[:, 1]
    thr = pick_threshold(y, train_prob, 0.80)
    pred_samples = pd.DataFrame()
    if not samples.empty:
        samples_x = samples.copy()
        if use_domain_correction:
            # OUR_SAMPLE_RAW is treated as service domain. No automatic label-derived correction is applied.
            pass
        samples_x["probability"] = pipe.predict_proba(samples_x[FEATURES])[:, 1]
        samples_x["prediction"] = (samples_x["probability"] >= thr).astype(int)
        pred_samples = samples_x[["dataset", "subject_id", "source_id", *FEATURES, "probability", "prediction"]]
    artifact = {"pipeline": pipe, "features": FEATURES, "threshold": thr, "domain_correction": params}
    return pipe, artifact, pred_samples


def sample_reference_correction_eval(subj: pd.DataFrame, samples: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if samples.empty:
        return {}, pd.DataFrame()
    ref = samples[FEATURES].median()
    params = correction_params(subj, reference=ref)
    corrected = apply_correction(subj, params)
    y = corrected["target"].astype(int).to_numpy()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )
    pipe.fit(corrected[FEATURES], y)
    prob_train = pipe.predict_proba(corrected[FEATURES])[:, 1]
    thr = pick_threshold(y, prob_train, 0.80)
    row = {"mode": "sample_reference_corrected_fit_all", **metrics(y, prob_train, thr)}
    sample_pred = samples.copy()
    sample_pred["probability"] = pipe.predict_proba(sample_pred[FEATURES])[:, 1]
    sample_pred["prediction"] = (sample_pred["probability"] >= thr).astype(int)
    return row, sample_pred[["dataset", "subject_id", "source_id", *FEATURES, "probability", "prediction"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    subj, samples = subject_table()
    scenario = "final_pattern4_no_turning"
    subj.to_csv(OUT_DIR / f"{scenario}_subject_table.csv", index=False, encoding="utf-8-sig")
    if not samples.empty:
        samples.to_csv(OUT_DIR / f"{scenario}_our_sample_features.csv", index=False, encoding="utf-8-sig")
    for use_corr in [False, True]:
        row = {"scenario": scenario, "n": len(subj), "normal_n": int((subj.target == 0).sum()), "impaired_n": int((subj.target == 1).sum())}
        row.update(cv_eval(subj, use_corr))
        results.append(row)
    _, artifact, pred = fit_final(subj, samples, use_domain_correction=False)
    joblib.dump(artifact, OUT_DIR / f"{scenario}_global_robust.joblib")
    if not pred.empty:
        pred.to_csv(OUT_DIR / f"{scenario}_our_sample_predictions.csv", index=False, encoding="utf-8-sig")
    sample_ref_row, sample_ref_pred = sample_reference_correction_eval(subj, samples)
    if sample_ref_row:
        sample_ref_row = {"scenario": scenario, "n": len(subj), "normal_n": int((subj.target == 0).sum()), "impaired_n": int((subj.target == 1).sum()), **sample_ref_row}
        results.append(sample_ref_row)
    if not sample_ref_pred.empty:
        sample_ref_pred.to_csv(OUT_DIR / f"{scenario}_sample_reference_our_sample_predictions.csv", index=False, encoding="utf-8-sig")
    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "pattern4_cv_results.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
