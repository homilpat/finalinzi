from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = ROOT / "analysis_outputs" / "service_reference_domain_alignment_groupcv" / "subject_table.csv"
SAMPLE_PATH = ROOT / "analysis_outputs" / "pattern4_domain_corrected_model" / "final_pattern4_no_turning_our_sample_features.csv"
OUT_DIR = ROOT / "analysis_outputs" / "final_low_domain_risk_gait_candidate_validation"

FEATURES = [
    "sample_entropy",
    "acf_stride_peak",
    "dominant_peak_prominence",
    "step_sec",
]


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def pick_threshold(y_true: np.ndarray, prob: np.ndarray, min_sens: float = 0.80) -> float:
    best_thr = 0.5
    best_spec = -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= min_sens and spec > best_spec:
            best_thr = float(thr)
            best_spec = spec
    return best_thr


def metric_row(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "acc": float(accuracy_score(y_true, pred)),
        "sens": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "spec": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SUBJECT_PATH)
    df = df.dropna(subset=FEATURES + ["target"]).copy()
    df["target"] = df["target"].astype(int)
    if "group_id" not in df.columns:
        df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)
    return df


def group_oof(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = df["target"].to_numpy()
    groups = df["group_id"].astype(str).to_numpy()
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    cv = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(cv.split(df[FEATURES], y, groups), start=1):
        model = make_model()
        model.fit(df.iloc[tr][FEATURES], y[tr])
        train_prob = model.predict_proba(df.iloc[tr][FEATURES])[:, 1]
        threshold = pick_threshold(y[tr], train_prob)
        test_prob = model.predict_proba(df.iloc[te][FEATURES])[:, 1]
        oof[te] = test_prob
        row = {"fold": fold, "n_train": len(tr), "n_test": len(te)}
        row.update(metric_row(y[te], test_prob, threshold))
        fold_rows.append(row)
    oof_thr = pick_threshold(y, oof)
    return pd.DataFrame(fold_rows), metric_row(y, oof, oof_thr)


def stratified_oof(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = df["target"].to_numpy()
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260720)
    for fold, (tr, te) in enumerate(cv.split(df[FEATURES], y), start=1):
        model = make_model()
        model.fit(df.iloc[tr][FEATURES], y[tr])
        train_prob = model.predict_proba(df.iloc[tr][FEATURES])[:, 1]
        threshold = pick_threshold(y[tr], train_prob)
        test_prob = model.predict_proba(df.iloc[te][FEATURES])[:, 1]
        oof[te] = test_prob
        row = {"fold": fold, "n_train": len(tr), "n_test": len(te)}
        row.update(metric_row(y[te], test_prob, threshold))
        fold_rows.append(row)
    oof_thr = pick_threshold(y, oof)
    return pd.DataFrame(fold_rows), metric_row(y, oof, oof_thr)


def repeated_8020(df: pd.DataFrame, repeats: int = 200) -> pd.DataFrame:
    y = df["target"].to_numpy()
    rows = []
    for seed in range(20260720, 20260720 + repeats):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
        model = make_model()
        model.fit(df.iloc[tr][FEATURES], y[tr])
        train_prob = model.predict_proba(df.iloc[tr][FEATURES])[:, 1]
        threshold = pick_threshold(y[tr], train_prob)
        test_prob = model.predict_proba(df.iloc[te][FEATURES])[:, 1]
        row = {"seed": seed, "n_train": len(tr), "n_test": len(te)}
        row.update(metric_row(y[te], test_prob, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def train_final_and_predict_samples(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = df["target"].to_numpy()
    model = make_model()
    model.fit(df[FEATURES], y)
    train_prob = model.predict_proba(df[FEATURES])[:, 1]
    threshold = pick_threshold(y, train_prob)
    artifact = {"pipeline": model, "features": FEATURES, "threshold": threshold, "label_meaning": {"0": "normal", "1": "motor_impaired"}}
    sample = pd.read_csv(SAMPLE_PATH)
    sample = sample.dropna(subset=FEATURES).copy()
    sample_prob = model.predict_proba(sample[FEATURES])[:, 1]
    out = sample[["dataset", "subject_id", "source_id", *FEATURES]].copy()
    out["probability"] = sample_prob
    out["threshold"] = threshold
    out["prediction"] = (sample_prob >= threshold).astype(int)
    out["label"] = np.where(out["prediction"].eq(1), "motor_impaired_possible", "normal_range_possible")
    return artifact, out


def summarize_repeated(rep: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "acc", "sens", "spec", "f1"]
    rows = []
    for metric in metrics:
        x = rep[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "p05": float(x.quantile(0.05)),
                "median": float(x.median()),
                "p95": float(x.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT_DIR / "validation_subject_table.csv", index=False, encoding="utf-8-sig")

    group_folds, group_summary = group_oof(df)
    strat_folds, strat_summary = stratified_oof(df)
    rep = repeated_8020(df)
    rep_summary = summarize_repeated(rep)
    artifact, sample_pred = train_final_and_predict_samples(df)

    group_folds.to_csv(OUT_DIR / "groupkfold_fold_metrics.csv", index=False, encoding="utf-8-sig")
    strat_folds.to_csv(OUT_DIR / "stratifiedkfold_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**{"validation": "groupkfold_oof"}, **group_summary}, {**{"validation": "stratifiedkfold_oof"}, **strat_summary}]).to_csv(
        OUT_DIR / "oof_summary.csv", index=False, encoding="utf-8-sig"
    )
    rep.to_csv(OUT_DIR / "repeated_8020_metrics.csv", index=False, encoding="utf-8-sig")
    rep_summary.to_csv(OUT_DIR / "repeated_8020_summary.csv", index=False, encoding="utf-8-sig")
    sample_pred.to_csv(OUT_DIR / "our_sample_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(artifact, OUT_DIR / "final_low_domain_risk_candidate.joblib")

    print("data counts")
    print(df.groupby(["dataset", "target"]).size().to_string())
    print("\nOOF summary")
    print(pd.DataFrame([{**{"validation": "groupkfold_oof"}, **group_summary}, {**{"validation": "stratifiedkfold_oof"}, **strat_summary}]).to_string(index=False))
    print("\nRepeated 8:2 summary")
    print(rep_summary.to_string(index=False))
    print("\nOUR SAMPLE predictions")
    print(sample_pred.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
