from __future__ import annotations

from itertools import combinations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from model_axis_aligned_domain_corrected_gait import apply_domain_correction, load_table, youden_threshold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

OUT_DIR = ROOT / "analysis_outputs" / "gait_model_domain_audit_options"
REFERENCE_MODE = "physionet_normal"
EXCLUDED_IMPAIRED_ONLY = {"Chapman_PD_OFF_RAW", "FoG_STAR_BACK_WALK"}
BASE_FEATURES = [
    "v_acf_stride_peak",
    "v_acf_stride_peak_width_sec",
    "ap_acf_stride_peak_width_sec",
    "ap_spec_entropy",
]
OPTIONAL_FEATURES = [
    "ap_acf_stride_peak",
    "step_sec",
    "stride_sec",
    "v_spec_entropy",
    "v_spec_peak_ratio",
    "ap_spec_peak_ratio",
    "ml_acf_stride_peak",
    "ml_acf_stride_peak_width_sec",
    "ml_spec_entropy",
]


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def metric_dict(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "acc": float(accuracy_score(y, pred)),
        "sens": float(tp / (tp + fn)) if tp + fn else np.nan,
        "spec": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else np.nan,
        "threshold": float(thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def oof_groupkfold(data: pd.DataFrame, features: list[str]) -> dict | None:
    data = data.dropna(subset=["target"]).copy()
    if data["target"].nunique() < 2 or len(data) < 20:
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
    return metric_dict(y, oof, youden_threshold(y, oof))


def leave_dataset_out(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    logo = LeaveOneGroupOut()
    y_all = data["target"].astype(int).to_numpy()
    datasets = data["dataset"].astype(str).to_numpy()
    for tr, te in logo.split(data[features], y_all, datasets):
        train = data.iloc[tr]
        test = data.iloc[te]
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            continue
        clf = model()
        clf.fit(train[features], train["target"].astype(int))
        train_p = clf.predict_proba(train[features])[:, 1]
        thr = youden_threshold(train["target"].astype(int).to_numpy(), train_p)
        test_p = clf.predict_proba(test[features])[:, 1]
        row = metric_dict(test["target"].astype(int).to_numpy(), test_p, thr)
        row["left_out_dataset"] = str(test["dataset"].iloc[0])
        row["n"] = int(len(test))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_sample(data: pd.DataFrame, features: list[str], train_mask: pd.Series) -> pd.DataFrame:
    train = data[train_mask].dropna(subset=["target"]).copy()
    sample = data[data["dataset"].eq("OUR_SAMPLE")].copy()
    clf = model()
    y = train["target"].astype(int).to_numpy()
    clf.fit(train[features], y)
    train_p = clf.predict_proba(train[features])[:, 1]
    thr = youden_threshold(y, train_p)
    sample_p = clf.predict_proba(sample[features])[:, 1]
    out = sample[["subject_id", "target", *features]].copy()
    out["probability"] = sample_p
    out["threshold"] = thr
    out["prediction"] = (sample_p >= thr).astype(int)
    out["correct"] = out["prediction"].eq(out["target"].astype(int))
    return out


def candidate_sets(columns: set[str]) -> list[list[str]]:
    available_optional = [feature for feature in OPTIONAL_FEATURES if feature in columns]
    sets = [BASE_FEATURES]
    for k in range(2, 6):
        for combo in combinations([f for f in BASE_FEATURES + available_optional if f in columns], k):
            if "v_acf_stride_peak" not in combo and "ap_spec_entropy" not in combo:
                continue
            sets.append(list(combo))
    unique = []
    seen = set()
    for features in sets:
        key = tuple(features)
        if key not in seen and set(features).issubset(columns):
            seen.add(key)
            unique.append(features)
    return unique


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_table()
    columns = set(raw.columns)
    correction_features = [feature for feature in BASE_FEATURES + OPTIONAL_FEATURES if feature in columns]
    corrected, correction = apply_domain_correction(raw, correction_features, REFERENCE_MODE)
    columns = set(corrected.columns)

    masks = {
        "current_all_public": corrected["dataset"].ne("OUR_SAMPLE"),
        "a_no_impaired_only": corrected["dataset"].ne("OUR_SAMPLE") & ~corrected["dataset"].isin(EXCLUDED_IMPAIRED_ONLY),
        "physionet_only": corrected["dataset"].eq("PhysioNet_LabWalks"),
    }

    rows = []
    sample_rows = []
    lodo_rows = []
    for name, mask in masks.items():
        data = corrected[mask].copy()
        for features in candidate_sets(columns):
            if len(data.dropna(subset=features, how="all")) < 20:
                continue
            res = oof_groupkfold(data, features)
            if res is None:
                continue
            row = {
                "option": name,
                "features": " + ".join(features),
                "k": len(features),
                "train_n": int(len(data.dropna(subset=["target"]))),
            }
            row.update(res)
            rows.append(row)
            sample = predict_sample(corrected, features, mask)
            sample["option"] = name
            sample["features_used"] = row["features"]
            sample_rows.append(sample)
            if name == "current_all_public":
                lodo = leave_dataset_out(data.dropna(subset=["target"]).copy(), features)
                if not lodo.empty:
                    lodo["features_used"] = row["features"]
                    lodo_rows.append(lodo)

    results = pd.DataFrame(rows).sort_values(
        ["option", "auc", "sens", "spec", "f1"],
        ascending=[True, False, False, False, False],
    )
    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    lodo_df = pd.concat(lodo_rows, ignore_index=True) if lodo_rows else pd.DataFrame()
    results.to_csv(OUT_DIR / "option_feature_screen.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT_DIR / "option_heldout_sample_predictions.csv", index=False, encoding="utf-8-sig")
    lodo_df.to_csv(OUT_DIR / "leave_dataset_out_current_public.csv", index=False, encoding="utf-8-sig")
    correction.to_csv(OUT_DIR / "domain_correction_deltas.csv", index=False, encoding="utf-8-sig")

    summary = {
        "options": list(masks),
        "reference_mode": REFERENCE_MODE,
        "excluded_impaired_only": sorted(EXCLUDED_IMPAIRED_ONLY),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for option in masks:
        print("\n==", option, "==")
        cols = ["features", "auc", "acc", "sens", "spec", "f1", "threshold", "tn", "fp", "fn", "tp"]
        print(results[results["option"].eq(option)].head(10)[cols].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
