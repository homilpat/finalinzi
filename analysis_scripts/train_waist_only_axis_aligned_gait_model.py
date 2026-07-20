from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from model_axis_aligned_domain_corrected_gait import (
    apply_domain_correction,
    load_table,
    metrics,
    youden_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

from gait_axis_aligned_core import FEATURES


OUT_DIR = ROOT / "analysis_outputs" / "waist_only_axis_aligned_gait_model"
MODEL_DIR = ROOT / "MOCA" / "models"
WAIST_FEATURES = [
    "ap_acf_stride_peak_width_sec",
    "ap_spec_entropy",
]
WAIST_DATASETS = {
    "PhysioNet_LabWalks",      # lower-back/L5 IMU
    "FoG_STAR_BACK_WALK",      # back sensor walking episodes
    "OUR_SAMPLE",              # waist-belt APK samples, held out
}
REFERENCE_MODE = "physionet_normal"


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, pred)),
        "sens": float(tp / (tp + fn)) if tp + fn else np.nan,
        "spec": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def group_oof(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = data[data["dataset"].ne("OUR_SAMPLE")].dropna(subset=["target"]).copy()
    y = train["target"].astype(int).to_numpy()
    groups = train["group_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros(len(train))
    fold_rows = []
    cv = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(cv.split(train[WAIST_FEATURES], y, groups), start=1):
        clf = model()
        clf.fit(train.iloc[tr][WAIST_FEATURES], y[tr])
        train_prob = clf.predict_proba(train.iloc[tr][WAIST_FEATURES])[:, 1]
        test_prob = clf.predict_proba(train.iloc[te][WAIST_FEATURES])[:, 1]
        threshold = youden_threshold(y[tr], train_prob)
        fold_metric = _metrics(y[te], test_prob, threshold)
        fold_metric.update({"fold": fold, "train_n": int(len(tr)), "test_n": int(len(te))})
        fold_rows.append(fold_metric)
        oof[te] = test_prob
    oof_threshold = youden_threshold(y, oof)
    pred = train[["dataset", "subject_id", "group_id", "target", *WAIST_FEATURES]].copy()
    pred["probability"] = oof
    pred["threshold"] = oof_threshold
    pred["prediction"] = (oof >= oof_threshold).astype(int)
    pred["correct"] = pred["prediction"].eq(pred["target"].astype(int))
    return pred, {"folds": fold_rows, "oof": _metrics(y, oof, oof_threshold)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_table()
    raw = raw[raw["dataset"].isin(WAIST_DATASETS)].copy()
    corrected, correction = apply_domain_correction(raw, WAIST_FEATURES, REFERENCE_MODE)
    train = corrected[corrected["dataset"].ne("OUR_SAMPLE")].dropna(subset=["target"]).copy()
    y = train["target"].astype(int).to_numpy()

    oof_pred, cv = group_oof(corrected)
    final_model = model()
    final_model.fit(train[WAIST_FEATURES], y)
    train_prob = final_model.predict_proba(train[WAIST_FEATURES])[:, 1]
    threshold = float(youden_threshold(y, train_prob))
    train_metric = metrics(y, train_prob, threshold)

    sample = corrected[corrected["dataset"].eq("OUR_SAMPLE")].copy()
    sample_prob = final_model.predict_proba(sample[WAIST_FEATURES])[:, 1]
    sample_pred = sample[["dataset", "subject_id", "target", *WAIST_FEATURES]].copy()
    sample_pred["probability"] = sample_prob
    sample_pred["threshold"] = threshold
    sample_pred["prediction"] = (sample_prob >= threshold).astype(int)
    sample_pred["correct"] = sample_pred["prediction"].eq(sample_pred["target"].astype(int))

    artifact = {
        "pipeline": final_model,
        "features": WAIST_FEATURES,
        "threshold": threshold,
        "threshold_strategy": "waist_only_ap_shape_entropy_final_train_youden",
        "model_mode": "axis_aligned_waist_only_ap_shape_entropy",
    }
    metadata = {
        "features": WAIST_FEATURES,
        "waist_datasets": sorted(WAIST_DATASETS),
        "reference_mode": REFERENCE_MODE,
        "threshold_mode": "youden",
        "threshold": threshold,
        "cv_metrics": cv["oof"],
        "train_metrics_at_deploy_threshold": train_metric,
        "fold_metrics": cv["folds"],
        "dataset_counts": {
            f"{dataset}::target_{int(target)}": int(count)
            for (dataset, target), count in train.groupby(["dataset", "target"]).size().items()
        },
        "service_note": "Waist/back-only candidate. OUR_SAMPLE is held out and not used for fitting.",
    }

    joblib.dump(artifact, MODEL_DIR / "gait_axis_aligned_waist_youden.joblib")
    (MODEL_DIR / "gait_axis_aligned_waist_youden_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    oof_pred.to_csv(OUT_DIR / "group_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv["folds"]).to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    sample_pred.to_csv(OUT_DIR / "heldout_sample_predictions.csv", index=False, encoding="utf-8-sig")
    correction.to_csv(OUT_DIR / "domain_correction_deltas.csv", index=False, encoding="utf-8-sig")

    print("COUNTS")
    print(train.groupby(["dataset", "target"]).size().to_string())
    print("\nOOF", cv["oof"])
    print("TRAIN_DEPLOY_THRESHOLD", train_metric)
    print("\nfolds")
    print(pd.DataFrame(cv["folds"]).to_string(index=False))
    print("\nsamples")
    print(sample_pred[["subject_id", "target", "probability", "threshold", "prediction", "correct"]].to_string(index=False))
    print("\nmodel", MODEL_DIR / "gait_axis_aligned_waist_youden.joblib")


if __name__ == "__main__":
    main()
