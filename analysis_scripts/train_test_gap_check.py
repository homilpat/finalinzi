"""
GroupKFold fold별 train vs test AUC/Sen/Spec 비교 → 과적합 확인
- 단일 5-fold 결과 출력
- 100회 반복 StratifiedGroupKFold로 안정화 평균/std 출력
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
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))

BEST3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
THR = 0.470

table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id","motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")
df = table.dropna(subset=BEST3).copy()
y      = df["clinical_target"].astype(int).to_numpy()
groups = df["subject_id"].to_numpy()


def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])


def subj_metrics(idx, prob, thr=THR):
    sub = df.iloc[idx][["subject_id", "clinical_target"]].copy()
    sub["prob"] = prob
    agg = sub.groupby("subject_id").agg(t=("clinical_target", "first"), p=("prob", "mean"))
    if agg["t"].nunique() < 2:
        return None
    auc = float(roc_auc_score(agg["t"], agg["p"]))
    pred = (agg["p"] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(agg["t"], pred, labels=[0, 1]).ravel()
    sen  = float(tp / (tp + fn)) if tp + fn else 0.0
    spec = float(tn / (tn + fp)) if tn + fp else 0.0
    return auc, sen, spec


# ── 1. 단일 5-fold (GroupKFold, fixed) ──────────────────────────────────────
print("=== 단일 5-fold 결과 (GroupKFold, fixed) ===")
print(f"{'Fold':>4}  {'Tr-AUC':>7}  {'Tr-Sen':>7}  {'Tr-Spec':>8}  "
      f"{'Te-AUC':>7}  {'Te-Sen':>7}  {'Te-Spec':>8}")
print("-" * 65)

fold_rows = []
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(df[BEST3], y, groups)):
    pipe = make_pipe()
    pipe.fit(df.iloc[tr][BEST3].to_numpy(), y[tr])
    m_tr = subj_metrics(tr, pipe.predict_proba(df.iloc[tr][BEST3].to_numpy())[:, 1])
    m_te = subj_metrics(te, pipe.predict_proba(df.iloc[te][BEST3].to_numpy())[:, 1])
    if m_tr and m_te:
        print(f"{fold+1:>4}  {m_tr[0]:7.4f}  {m_tr[1]:7.4f}  {m_tr[2]:8.4f}  "
              f"{m_te[0]:7.4f}  {m_te[1]:7.4f}  {m_te[2]:8.4f}")
        fold_rows.append({"tr_auc": m_tr[0], "tr_sen": m_tr[1], "tr_spec": m_tr[2],
                          "te_auc": m_te[0], "te_sen": m_te[1], "te_spec": m_te[2]})

r1 = pd.DataFrame(fold_rows)
print(f"{'평균':>4}  {r1.tr_auc.mean():7.4f}  {r1.tr_sen.mean():7.4f}  {r1.tr_spec.mean():8.4f}  "
      f"{r1.te_auc.mean():7.4f}  {r1.te_sen.mean():7.4f}  {r1.te_spec.mean():8.4f}")
print(f"{'Gap':>4}  {(r1.tr_auc-r1.te_auc).mean():+7.4f}  "
      f"{(r1.tr_sen-r1.te_sen).mean():+7.4f}  {(r1.tr_spec-r1.te_spec).mean():+8.4f}")

# ── 2. 100회 반복 StratifiedGroupKFold ──────────────────────────────────────
print("\n=== 100회 반복 StratifiedGroupKFold 5-fold ===")
N_REPS = 100
results = []

for seed in range(N_REPS):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_te = []
    for tr, te in sgkf.split(df[BEST3], y, groups):
        pipe = make_pipe()
        pipe.fit(df.iloc[tr][BEST3].to_numpy(), y[tr])
        m_tr = subj_metrics(tr, pipe.predict_proba(df.iloc[tr][BEST3].to_numpy())[:, 1])
        m_te = subj_metrics(te, pipe.predict_proba(df.iloc[te][BEST3].to_numpy())[:, 1])
        if m_tr and m_te:
            fold_te.append({"tr_auc": m_tr[0], "tr_sen": m_tr[1], "tr_spec": m_tr[2],
                            "te_auc": m_te[0], "te_sen": m_te[1], "te_spec": m_te[2]})
    if fold_te:
        arr = {k: np.mean([f[k] for f in fold_te]) for k in fold_te[0]}
        results.append(arr)

r = pd.DataFrame(results)
print(f"(n={len(r)}회 유효)\n")
print(f"           Train          Test           Gap")
print(f"AUC  : {r.tr_auc.mean():.4f}±{r.tr_auc.std():.4f}  "
      f"{r.te_auc.mean():.4f}±{r.te_auc.std():.4f}  "
      f"{(r.tr_auc-r.te_auc).mean():+.4f}±{(r.tr_auc-r.te_auc).std():.4f}")
print(f"Sen  : {r.tr_sen.mean():.4f}±{r.tr_sen.std():.4f}  "
      f"{r.te_sen.mean():.4f}±{r.te_sen.std():.4f}  "
      f"{(r.tr_sen-r.te_sen).mean():+.4f}±{(r.tr_sen-r.te_sen).std():.4f}")
print(f"Spec : {r.tr_spec.mean():.4f}±{r.tr_spec.std():.4f}  "
      f"{r.te_spec.mean():.4f}±{r.te_spec.std():.4f}  "
      f"{(r.tr_spec-r.te_spec).mean():+.4f}±{(r.tr_spec-r.te_spec).std():.4f}")
print(f"\nTest Sen  범위: {r.te_sen.min():.3f} ~ {r.te_sen.max():.3f}")
print(f"Test Spec 범위: {r.te_spec.min():.3f} ~ {r.te_spec.max():.3f}")
print("\n[완료]")
