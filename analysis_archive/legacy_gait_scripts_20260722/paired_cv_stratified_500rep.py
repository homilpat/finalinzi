"""
500회 반복 Paired 비교 (StratifiedGroupKFold 버전)

모델 3가지:
  A) 2피처:   v_jerk_rms_median + v_harmonic_ratio_iqr
  B) 기존3피처: v_jerk_rms_median + v_jerk_rms_iqr + v_harmonic_ratio_iqr
  C) 신규3피처: v_jerk_rms_median + v_harmonic_ratio_iqr + pitch_band_rms__iqr

[데이터]
  - 가속도 피처(jerk/hr): subwindow_median_iqr_table.csv → subject-level median
  - pitch_band_rms__iqr:  subject_features_with_clinical.csv
  - 라벨:                 motor_impairment_score >= 0.5

[CV]
  - StratifiedGroupKFold(5, shuffle=True) — fold 클래스 비율 보장
  - 500회 반복: rep마다 다른 random_state → 같은 rep 내 3모델 동일 fold (paired)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# ── 경로 ──────────────────────────────────────────────────────────
SUBWIN_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(
    Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"),
    None,
)

N_REP   = 500
N_SPLIT = 5
SEED    = 42

ACCEL_FEATS = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
GYRO_FEAT   = "pitch_band_rms__iqr"

# ── 데이터 로드 ───────────────────────────────────────────────────
# 1) subwindow → subject-level median (가속도 피처)
sub = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)

# 라벨 교체: Faller/Control → motor_impairment_score 기반
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id")
sub = sub.merge(clin_lbl, on="subject_id", how="inner", suffixes=("_drop", ""))
# target_drop 제거
drop_cols = [c for c in sub.columns if c.endswith("_drop")]
sub = sub.drop(columns=drop_cols)

# subject-level median 집계
subj_accel = (
    sub.groupby("subject_id")[ACCEL_FEATS + ["target"]]
    .agg({f: "median" for f in ACCEL_FEATS} | {"target": "first"})
    .reset_index()
)

# 2) clinical CSV → pitch_band_rms__iqr
clin_gyro = clin[["subject_id", GYRO_FEAT]].drop_duplicates("subject_id")

# 3) 합치기
df = subj_accel.merge(clin_gyro, on="subject_id", how="inner")
df = df.dropna(subset=ACCEL_FEATS + [GYRO_FEAT]).reset_index(drop=True)

y      = df["target"].values.astype(int)
groups = df["subject_id"].values
n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"데이터: {len(y)}명 | 정상 {n0} / 저하 {n1}")

FEAT_A = ["v_jerk_rms_median", "v_harmonic_ratio_iqr"]
FEAT_B = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
FEAT_C = ["v_jerk_rms_median", "v_harmonic_ratio_iqr", "pitch_band_rms__iqr"]

print(f"모델 A ({len(FEAT_A)}피처): {FEAT_A}")
print(f"모델 B ({len(FEAT_B)}피처, 기존): {FEAT_B}")
print(f"모델 C ({len(FEAT_C)}피처, 신규): {FEAT_C}")
print(f"\n500회 × StratifiedGroupKFold({N_SPLIT}) 시작 ...")

# ── CV AUC 함수 ───────────────────────────────────────────────────
pipe_factory = lambda: Pipeline([
    ("imp",   SimpleImputer(strategy="median")),
    ("scale", RobustScaler()),
    ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
])

def cv_auc(feats: list[str], random_state: int) -> float:
    X    = df[feats].values.astype(float)
    pipe = pipe_factory()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLIT, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan

# ── 500회 반복 ────────────────────────────────────────────────────
auc_a = np.full(N_REP, np.nan)
auc_b = np.full(N_REP, np.nan)
auc_c = np.full(N_REP, np.nan)

for rep in range(N_REP):
    rs = SEED + rep * 7
    auc_a[rep] = cv_auc(FEAT_A, rs)
    auc_b[rep] = cv_auc(FEAT_B, rs)
    auc_c[rep] = cv_auc(FEAT_C, rs)
    if (rep + 1) % 100 == 0:
        print(f"  [{rep+1:3d}/500]  A={np.nanmean(auc_a[:rep+1]):.4f}  "
              f"B={np.nanmean(auc_b[:rep+1]):.4f}  C={np.nanmean(auc_c[:rep+1]):.4f}")

# ── 결과 ─────────────────────────────────────────────────────────
def ci95(arr):
    return np.nanpercentile(arr, [2.5, 97.5])

def paired_t(a, b):
    delta = np.nanmean(b - a)
    _, p  = stats.ttest_rel(a, b, nan_policy="omit")
    return delta, float(p)

print("\n" + "=" * 65)
print("결과: 500회 StratifiedGroupKFold (per-rep mean AUC)")
print("=" * 65)
for tag, arr in [("A 2피처    ", auc_a), ("B 기존3피처 ", auc_b), ("C 신규3피처 ", auc_c)]:
    lo, hi = ci95(arr)
    print(f"  {tag}: {np.nanmean(arr):.4f}  95CI [{lo:.3f}, {hi:.3f}]  std={np.nanstd(arr):.4f}")

print()
print("Paired t-test")
for tag, (a, b) in [("B vs A (기존3 vs 2피처)", (auc_a, auc_b)),
                     ("C vs A (신규3 vs 2피처)", (auc_a, auc_c)),
                     ("C vs B (신규3 vs 기존3)", (auc_b, auc_c))]:
    delta, p = paired_t(a, b)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {tag}: delta={delta:+.4f}  p={p:.4f}  {sig}")

# ── 저장 ─────────────────────────────────────────────────────────
out = ROOT / "analysis_outputs" / "bout_duration_feature_selection"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "auc_A_2feat":       auc_a,
    "auc_B_3feat_orig":  auc_b,
    "auc_C_3feat_gyro":  auc_c,
}).to_csv(out / "paired_stratified_500rep.csv", index=False)
print(f"\n저장: {out / 'paired_stratified_500rep.csv'}")
