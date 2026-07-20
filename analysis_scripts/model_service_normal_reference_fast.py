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
PUBLIC_PATH = ROOT / "analysis_outputs" / "normal_vs_impaired_gait_pattern_comparison" / "normal_impaired_pattern_features.csv"
SAMPLE_PATH = ROOT / "analysis_outputs" / "pattern4_domain_corrected_model" / "final_pattern4_no_turning_our_sample_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "service_normal_reference_fast"

FEATURES = [
    "spec_peak_ratio",
    "spec_entropy",
    "dominant_peak_prominence",
    "acf_stride_peak_width_sec",
    "acf_stride_peak",
    "step_sec",
]


def sample_label(subject_id: str) -> int:
    return 1 if "20260716" in str(subject_id) else 0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PUBLIC_PATH)
    df = df[df["dataset"].ne("PD_TURNING_IMU")].copy()
    df = df[~df["source_note"].astype(str).str.contains("existing fixed_best10_quality", na=False)].copy()
    df["_quality"] = pd.to_numeric(df["acf_stride_peak"], errors="coerce")
    idx = df.sort_values("_quality", ascending=False).groupby(["dataset", "subject_id"], sort=False).head(1).index
    df = df.loc[idx].reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)

    sample = pd.read_csv(SAMPLE_PATH)
    sample["target"] = sample["subject_id"].map(sample_label).astype(int)
    sample["group_id"] = "OUR_SAMPLE_RAW::" + sample["subject_id"].astype(str)
    return df, sample


def transform_to_sample_reference(train: pd.DataFrame, sample: pd.DataFrame, features: list[str], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "none":
        return train.copy(), sample.copy()
    out = train.copy()
    sample_out = sample.copy()
    ref = sample_out[sample_out["target"].eq(0)]
    base = out[out["target"].eq(0)]
    delta = ref[features].median() - base[features].median()
    out.loc[:, features] = out[features] + delta
    return out, sample_out


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_spec = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens >= 0.80 and spec > best_spec:
            best_thr, best_spec = float(thr), spec
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


def group_oof(df: pd.DataFrame, sample: pd.DataFrame, features: list[str], mode: str) -> dict:
    data, _ = transform_to_sample_reference(df, sample, features, mode)
    data = data.dropna(subset=features + ["target"]).copy()
    y = data["target"].to_numpy()
    oof = np.zeros(len(data))
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(data[features], y, data["group_id"]):
        m = model()
        m.fit(data.iloc[tr][features], y[tr])
        train_p = m.predict_proba(data.iloc[tr][features])[:, 1]
        _ = pick_threshold(y[tr], train_p)
        oof[te] = m.predict_proba(data.iloc[te][features])[:, 1]
    return metrics(y, oof, pick_threshold(y, oof))


def sample_pred(df: pd.DataFrame, sample: pd.DataFrame, features: list[str], mode: str) -> pd.DataFrame:
    train, sample_x = transform_to_sample_reference(df, sample, features, mode)
    train = train.dropna(subset=features + ["target"]).copy()
    sample_x = sample_x.dropna(subset=features).copy()
    y = train["target"].to_numpy()
    m = model()
    m.fit(train[features], y)
    train_p = m.predict_proba(train[features])[:, 1]
    thr = pick_threshold(y, train_p)
    p = m.predict_proba(sample_x[features])[:, 1]
    out = sample_x[["subject_id", "target", *features]].copy()
    out["probability"] = p
    out["threshold"] = thr
    out["prediction"] = (p >= thr).astype(int)
    out["correct"] = out["prediction"].eq(out["target"])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, sample = load_data()
    combos = []
    for k in [2, 3]:
        combos.extend(combinations(FEATURES, k))
    rows = []
    pred_rows = []
    for mode in ["none", "sample_shift"]:
        for combo in combos:
            features = list(combo)
            res = group_oof(df, sample, features, mode)
            sp = sample_pred(df, sample, features, mode)
            row = {
                "mode": mode,
                "features": " + ".join(features),
                "k": len(features),
                "sample_acc": float(sp["correct"].mean()),
                "sample_correct": int(sp["correct"].sum()),
                "sample_n": int(len(sp)),
            }
            row.update(res)
            rows.append(row)
            sp.insert(0, "mode", mode)
            sp.insert(1, "features", " + ".join(features))
            pred_rows.append(sp)
    results = pd.DataFrame(rows).sort_values(["sample_acc", "auc", "spec"], ascending=[False, False, False])
    preds = pd.concat(pred_rows, ignore_index=True)
    results.to_csv(OUT_DIR / "fast_model_screen.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "fast_sample_predictions.csv", index=False, encoding="utf-8-sig")
    print(results.head(30).to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
