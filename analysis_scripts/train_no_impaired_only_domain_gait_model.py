from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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

from gait_axis_aligned_core import FEATURES  # noqa: E402


OUT_DIR = ROOT / "analysis_outputs" / "no_impaired_only_domain_gait_model"
REFERENCE_MODE = "physionet_normal"
EXCLUDED_TRAIN_DATASETS = {"Chapman_PD_OFF_RAW", "FoG_STAR_BACK_WALK"}


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def _training_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        data["dataset"].ne("OUR_SAMPLE")
        & ~data["dataset"].isin(EXCLUDED_TRAIN_DATASETS)
    ].dropna(subset=["target"]).copy()


def group_oof(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = _training_rows(data)
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

    raw = load_table()
    corrected, correction = apply_domain_correction(raw, FEATURES, REFERENCE_MODE)
    train = _training_rows(corrected)
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
        "threshold_strategy": "physionet_normal_no_impaired_only_domains_final_train_youden",
        "model_mode": "axis_aligned_physionet_normal_no_impaired_only_domains",
        "excluded_train_datasets": sorted(EXCLUDED_TRAIN_DATASETS),
    }
    metadata = {
        "features": FEATURES,
        "reference_mode": REFERENCE_MODE,
        "threshold_mode": "youden",
        "threshold": threshold,
        "excluded_train_datasets": sorted(EXCLUDED_TRAIN_DATASETS),
        "cv_metrics": cv["oof"],
        "train_metrics_at_deploy_threshold": train_metric,
        "oof_threshold": cv["oof"]["threshold"],
        "fold_metrics": cv["folds"],
        "dataset_counts": {
            f"{dataset}::target_{int(target)}": int(count)
            for (dataset, target), count in train.groupby(["dataset", "target"]).size().items()
        },
        "excluded_dataset_counts": {
            f"{dataset}::target_{int(target)}": int(count)
            for (dataset, target), count in raw[raw["dataset"].isin(EXCLUDED_TRAIN_DATASETS)]
            .groupby(["dataset", "target"])
            .size()
            .items()
        },
        "note": "A-option model: excludes impaired-only Chapman/FoG domains to reduce dataset-label confounding risk. Not deployed.",
    }

    joblib.dump(artifact, OUT_DIR / "no_impaired_only_domain_gait_model.joblib")
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_pred.to_csv(OUT_DIR / "group_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv["folds"]).to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    sample_pred.to_csv(OUT_DIR / "heldout_sample_predictions.csv", index=False, encoding="utf-8-sig")
    correction.to_csv(OUT_DIR / "domain_correction_deltas.csv", index=False, encoding="utf-8-sig")

    print("TRAIN_COUNTS")
    print(pd.Series(metadata["dataset_counts"]).to_string())
    print("\nEXCLUDED_COUNTS")
    print(pd.Series(metadata["excluded_dataset_counts"]).to_string())
    print("\nOOF")
    print(cv["oof"])
    print("\nTRAIN_DEPLOY_THRESHOLD")
    print(train_metric)
    print("\nFOLDS")
    print(pd.DataFrame(cv["folds"]).to_string(index=False))
    print("\nHELDOUT OUR_SAMPLE")
    print(sample_pred[["subject_id", "target", "probability", "threshold", "prediction", "correct"]].to_string(index=False))

    if sample_pred["target"].notna().all():
        tn, fp, fn, tp = confusion_matrix(
            sample_pred["target"].astype(int),
            sample_pred["prediction"].astype(int),
            labels=[0, 1],
        ).ravel()
        print("\nHELDOUT_CONFUSION", {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
