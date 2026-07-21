"""
1) VIF 다중공선성 검증
2) stride_width_sec 피처 추가 테스트 (v2 CSV에서 직접 가져오기 or window_features 재활용)
3) subject 단위 AUC도 함께 보고
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))

BASE  = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz",
         "ap_spec_entropy", "v_jerk_rms"]
FEATS = [f"{f}_median" for f in BASE] + [f"{f}_iqr" for f in BASE]

table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id","motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")
train = table.dropna(subset=FEATS).copy()
y      = train["clinical_target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()

# ── 0) 스케일링 후 피처 행렬 ─────────────────────────────
X_raw = train[FEATS].to_numpy()
imp = SimpleImputer(strategy="median")
sc  = RobustScaler()
X   = sc.fit_transform(imp.fit_transform(X_raw))
X_df = pd.DataFrame(X, columns=FEATS)

# ── 1) 피처 간 Pearson 상관 행렬 (절댓값) ───────────────
print("[1] 피처 간 상관계수 (절댓값 ≥ 0.60인 쌍)")
corr = X_df.corr().abs()
pairs = []
for i, c1 in enumerate(FEATS):
    for j, c2 in enumerate(FEATS):
        if j <= i: continue
        r = corr.loc[c1, c2]
        if r >= 0.60:
            pairs.append((r, c1, c2))
pairs.sort(reverse=True)
if pairs:
    for r, c1, c2 in pairs:
        print(f"  r={r:.3f}  {c1}  ↔  {c2}")
else:
    print("  없음 (r < 0.60)")

# ── 2) VIF 계산 ────────────────────────────────────────
print("\n[2] VIF (Variance Inflation Factor)")
print(f"  {'피처':40s}  VIF")
vifs = {}
for i, col in enumerate(FEATS):
    X_i  = X_df.drop(columns=[col]).to_numpy()
    y_i  = X_df[col].to_numpy()
    lr   = LinearRegression().fit(X_i, y_i)
    r2   = lr.score(X_i, y_i)
    vif  = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    vifs[col] = vif
    flag = "  ← 주의" if vif > 5 else ("  ← 심각" if vif > 10 else "")
    print(f"  {col:40s}  {vif:6.2f}{flag}")

# ── 3) 윈도우 단위 vs subject 단위 AUC ──────────────────
def oof_auc(feats, y_, groups_):
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[feats], y_, groups_):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        pipe.fit(train.iloc[tr][feats].to_numpy(), y_[tr])
        oof[te] = pipe.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]
    return oof

BEST3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

print("\n[3] 윈도우 단위 OOF AUC")
for label, feats in [("전체 10피처", FEATS), ("최적 3피처", BEST3)]:
    oof = oof_auc(feats, y, groups)
    auc = roc_auc_score(y, oof)
    print(f"  [{label}]  AUC={auc:.4f}")

print("\n[4] Subject 단위 AUC (각 subject의 윈도우 평균 확률)")
oof_full = oof_auc(FEATS, y, groups)
oof_b3   = oof_auc(BEST3, y, groups)

sub_df = train[["subject_id","clinical_target"]].copy()
sub_df["prob_full"] = oof_full
sub_df["prob_b3"]   = oof_b3
sub_agg = sub_df.groupby("subject_id").agg(
    target=("clinical_target","first"),
    prob_full=("prob_full","mean"),
    prob_b3=("prob_b3","mean"),
).reset_index()

auc_sub_full = roc_auc_score(sub_agg["target"], sub_agg["prob_full"])
auc_sub_b3   = roc_auc_score(sub_agg["target"], sub_agg["prob_b3"])
print(f"  [전체 10피처]  subject AUC={auc_sub_full:.4f}")
print(f"  [최적  3피처]  subject AUC={auc_sub_b3:.4f}")

# youden threshold on subject level
def subj_perf(prob_col, thr):
    pred = (sub_agg[prob_col] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(sub_agg["target"], pred, labels=[0,1]).ravel()
    s  = tp/(tp+fn) if tp+fn else 0.
    sp = tn/(tn+fp) if tn+fp else 0.
    return s, sp, tp, fn, tn, fp

for label, col, auc_v in [("전체 10피처", "prob_full", auc_sub_full),
                           ("최적  3피처", "prob_b3",   auc_sub_b3)]:
    best_j, best_thr = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        s, sp, *_ = subj_perf(col, t)
        if s + sp - 1 > best_j: best_j, best_thr = s+sp-1, t
    s, sp, tp, fn, tn, fp = subj_perf(col, best_thr)
    print(f"  [{label}]  Youden thr={best_thr:.3f}  sens={s:.3f}  spec={sp:.3f}  TP={tp} FN={fn} TN={tn} FP={fp}")

print("\n[완료]")
