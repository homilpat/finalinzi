"""
GroupKFold fold별 train AUC vs test AUC 비교 → 과적합 확인
subject 단위 집계 기준
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))

BEST3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
ALL10 = ["v_harmonic_ratio_median","ap_harmonic_ratio_median","v_stride_freq_hz_median",
         "ap_spec_entropy_median","v_jerk_rms_median",
         "v_harmonic_ratio_iqr","ap_harmonic_ratio_iqr","v_stride_freq_hz_iqr",
         "ap_spec_entropy_iqr","v_jerk_rms_iqr"]

table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id","motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")
train = table.dropna(subset=ALL10).copy()
y      = train["clinical_target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()

def subj_auc(df_idx, prob, df):
    """윈도우 인덱스 → subject 단위 평균 확률 → AUC"""
    sub = df.iloc[df_idx][["subject_id","clinical_target"]].copy()
    sub["prob"] = prob
    agg = sub.groupby("subject_id").agg(target=("clinical_target","first"), prob=("prob","mean"))
    if agg["target"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(agg["target"], agg["prob"]))

print(f"{'모델':12s}  {'Fold':>4}  {'Train-subj AUC':>14}  {'Test-subj AUC':>13}  {'Gap':>6}  Test subjects")
print("-" * 75)

for model_name, feats in [("3피처", BEST3), ("10피처", ALL10)]:
    fold_trains, fold_tests = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(train[feats], y, groups)):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        pipe.fit(train.iloc[tr][feats].to_numpy(), y[tr])

        prob_tr = pipe.predict_proba(train.iloc[tr][feats].to_numpy())[:, 1]
        prob_te = pipe.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]

        auc_tr = subj_auc(tr, prob_tr, train)
        auc_te = subj_auc(te, prob_te, train)
        gap    = auc_tr - auc_te if np.isfinite(auc_tr) and np.isfinite(auc_te) else np.nan
        n_te_subj = train.iloc[te]["subject_id"].nunique()

        fold_trains.append(auc_tr)
        fold_tests.append(auc_te)
        print(f"{model_name:12s}  {fold+1:>4}  {auc_tr:14.4f}  {auc_te:13.4f}  {gap:+6.3f}  n={n_te_subj}")

    mean_tr = np.nanmean(fold_trains)
    mean_te = np.nanmean(fold_tests)
    print(f"{model_name:12s}  {'평균':>4}  {mean_tr:14.4f}  {mean_te:13.4f}  {mean_tr-mean_te:+6.3f}")
    print()

print("[완료]")
