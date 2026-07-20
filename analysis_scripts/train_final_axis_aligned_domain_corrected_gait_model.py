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

OUT_DIR = ROOT / "analysis_outputs" / "final_axis_aligned_physionet_normal_gait_model"
MODEL_DIR = ROOT / "MOCA" / "models"

REFERENCE_MODE = "physionet_normal"
THRESHOLD_MODE = "youden"


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def group_oof(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = data[data["dataset"].ne("OUR_SAMPLE")].dropna(subset=["target"]).copy()
    y = train["target"].astype(int).to_numpy()
    groups = train["group_id"].astype(str).to_numpy()
    oof = np.zeros(len(train))
    fold_rows = []
    cv = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(cv.split(train[FEATURES], y, groups), start=1):
        clf = model()
        clf.fit(train.iloc[tr][FEATURES], y[tr])
        train_prob = clf.predict_proba(train.iloc[tr][FEATURES])[:, 1]
        test_prob = clf.predict_proba(train.iloc[te][FEATURES])[:, 1]
        thr = youden_threshold(y[tr], train_prob)
        fold_metric = metrics(y[te], test_prob, thr)
        fold_metric.update({"fold": fold, "threshold": thr, "train_n": int(len(tr)), "test_n": int(len(te))})
        fold_rows.append(fold_metric)
        oof[te] = test_prob
    oof_thr = youden_threshold(y, oof)
    oof_metric = metrics(y, oof, oof_thr)
    pred = train[["dataset", "subject_id", "group_id", "target", *FEATURES]].copy()
    pred["probability"] = oof
    pred["threshold"] = oof_thr
    pred["prediction"] = (oof >= oof_thr).astype(int)
    pred["correct"] = pred["prediction"].eq(pred["target"].astype(int))
    return pred, {"folds": fold_rows, "oof": oof_metric}


def sample_predictions(data: pd.DataFrame, clf: Pipeline, threshold: float) -> pd.DataFrame:
    sample = data[data["dataset"].eq("OUR_SAMPLE")].copy()
    prob = clf.predict_proba(sample[FEATURES])[:, 1]
    out = sample[["dataset", "subject_id", "target", *FEATURES]].copy()
    out["probability"] = prob
    out["threshold"] = threshold
    out["prediction"] = (prob >= threshold).astype(int)
    out["correct"] = out["prediction"].eq(out["target"].astype(int))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_table()
    corrected, correction = apply_domain_correction(raw, FEATURES, REFERENCE_MODE)
    train = corrected[corrected["dataset"].ne("OUR_SAMPLE")].dropna(subset=["target"]).copy()
    y = train["target"].astype(int).to_numpy()

    oof_pred, cv = group_oof(corrected)
    final_model = model()
    final_model.fit(train[FEATURES], y)
    train_prob = final_model.predict_proba(train[FEATURES])[:, 1]
    threshold = float(youden_threshold(y, train_prob))
    train_metric = metrics(y, train_prob, threshold)
    sample_pred = sample_predictions(corrected, final_model, threshold)

    artifact = {
        "pipeline": final_model,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_strategy": "physionet_normal_domain_corrected_final_train_youden",
        "model_mode": "axis_aligned_physionet_normal_domain_corrected",
    }
    metadata = {
        "features": FEATURES,
        "reference_mode": REFERENCE_MODE,
        "threshold_mode": THRESHOLD_MODE,
        "threshold": threshold,
        "cv_metrics": cv["oof"],
        "train_metrics_at_deploy_threshold": train_metric,
        "oof_threshold": cv["oof"]["threshold"],
        "deploy_threshold_note": "OOF threshold is reported for validation. Deployed threshold is selected on the final public training set by Youden after model selection; held-out OUR_SAMPLE is not used for fitting.",
        "fold_metrics": cv["folds"],
        "dataset_counts": {
            f"{dataset}::target_{int(target)}": int(count)
            for (dataset, target), count in train.groupby(["dataset", "target"]).size().items()
        },
        "service_note": "Training domains with available normal controls are median-shift corrected to PhysioNet LabWalks normal reference. OUR_SAMPLE is held out and not used for fitting or correction.",
    }

    joblib.dump(artifact, MODEL_DIR / "gait_axis_aligned_physionet_youden.joblib")
    (MODEL_DIR / "gait_axis_aligned_physionet_youden_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    oof_pred.to_csv(OUT_DIR / "group_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv["folds"]).to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    sample_pred.to_csv(OUT_DIR / "heldout_sample_predictions.csv", index=False, encoding="utf-8-sig")
    correction.to_csv(OUT_DIR / "domain_correction_deltas.csv", index=False, encoding="utf-8-sig")

    print("OOF", cv["oof"])
    print("TRAIN_DEPLOY_THRESHOLD", train_metric)
    print("\nfolds")
    print(pd.DataFrame(cv["folds"]).to_string(index=False))
    print("\nsamples")
    print(sample_pred[["subject_id", "target", "probability", "threshold", "prediction", "correct"]].to_string(index=False))
    print("\nmodel", MODEL_DIR / "gait_axis_aligned_physionet_youden.joblib")


if __name__ == "__main__":
    main()
