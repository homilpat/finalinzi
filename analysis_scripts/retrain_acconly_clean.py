"""
Acc-only 3피처 모델 클린 재훈련 → gait_daily_clinical_3feat.joblib 덮어쓰기

[피처] v_jerk_rms_median, v_jerk_rms_iqr, v_harmonic_ratio_iqr (자이로 없음)
[이유] 허리 자유 착용 + 자동 20s 측정 → 자이로 방향 불안정 → acc-only 우월
[성능] 클린 AUC 0.861 vs pitch 포함 0.848
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

SUBWIN_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
MODEL_SRC    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat.joblib"
MODEL_DST    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat.joblib"
META_DST     = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_metadata.json"

FEATS = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
N_SPLIT = 5
N_SEED  = 100

# ── 1. 데이터 ─────────────────────────────────────────────────────
print("[1] 데이터 구성")
sub  = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id")

sub  = sub.merge(clin_lbl, on="subject_id", how="inner", suffixes=("_drop", ""))
sub  = sub.drop(columns=[c for c in sub.columns if c.endswith("_drop")])
# Subject-level 집계 (pitch model과 동일한 방식)
df = (
    sub.groupby("subject_id")[FEATS + ["target"]]
    .agg({f: "median" for f in FEATS} | {"target": "first"})
    .reset_index()
)
df = df.dropna(subset=FEATS).reset_index(drop=True)

y      = df["target"].values.astype(int)
groups = df["subject_id"].values
n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"  {len(y)}명 | 정상 {n0} / 저하 {n1}")

# ── 2. OOF AUC (100회 StratifiedGroupKFold) ───────────────────────
print("\n[2] OOF AUC (StratifiedGroupKFold 5-fold, 100회)")

def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])

X = df[FEATS].values.astype(float)

auc_list = []
for seed in range(N_SEED):
    sgkf = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        p = make_pipe()
        p.fit(X[tr], y[tr])
        fold_aucs.append(roc_auc_score(y[te], p.predict_proba(X[te])[:, 1]))
    if fold_aucs:
        auc_list.append(float(np.mean(fold_aucs)))

oof_auc = float(np.mean(auc_list))
print(f"  Acc-only AUC: {oof_auc:.4f} ± {np.std(auc_list):.4f}  "
      f"95CI [{np.percentile(auc_list,2.5):.3f}, {np.percentile(auc_list,97.5):.3f}]")
print(f"  (pitch 포함 모델: 0.848)")

# ── 3. Youden 임계값 (seed=42) ────────────────────────────────────
print("\n[3] Youden 임계값 (seed=42)")
sgkf42 = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=42)
oof_prob = np.zeros(len(y))
for tr, te in sgkf42.split(X, y, groups):
    p = make_pipe(); p.fit(X[tr], y[tr])
    oof_prob[te] = p.predict_proba(X[te])[:, 1]

best_j, best_thr = -np.inf, 0.5
for t in np.linspace(0.05, 0.95, 181):
    pred = (oof_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    j = (tp/(tp+fn) if tp+fn else 0) + (tn/(tn+fp) if tn+fp else 0) - 1
    if j > best_j:
        best_j, best_thr = j, t

pred_thr = (oof_prob >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_thr, labels=[0, 1]).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp)
oof_subj_auc = roc_auc_score(y, oof_prob)
print(f"  OOF AUC (seed=42): {oof_subj_auc:.4f}")
print(f"  threshold={best_thr:.3f}  sens={sens:.3f}  spec={spec:.3f}")

# ── 4. 전체 데이터로 최종 모델 훈련 ─────────────────────────────
print("\n[4] 전체 데이터로 최종 모델 훈련")
final_pipe = make_pipe()
final_pipe.fit(X, y)

# ── 5. 기존 signal_correction 이어받기 ───────────────────────────
old_art = joblib.load(MODEL_SRC)
signal_correction = old_art.get("signal_correction")
print(f"[5] signal_correction 이어받기: alpha={signal_correction.get('alpha'):.4f}")

# ── 6. 저장 ───────────────────────────────────────────────────────
print("\n[6] 모델 저장")
artifact = {
    "pipeline":           final_pipe,
    "features":           FEATS,
    "threshold":          float(best_thr),
    "threshold_strategy": "StratifiedGroupKFold_Youden_subject_level",
    "model_mode":         "daily_3feat_acconly",
    "label_source":       "motor_impairment_score >= 0.5",
    "train_n_subjects":   int(len(y)),
    "train_n_normal":     int(n0),
    "train_n_impaired":   int(n1),
    "oof_subject_auc":    round(oof_auc, 4),
    "oof_subject_auc_std":round(float(np.std(auc_list)), 4),
    "oof_subject_sens":   round(sens, 4),
    "oof_subject_spec":   round(spec, 4),
    "signal_correction":  signal_correction,
}
joblib.dump(artifact, MODEL_DST)

meta = {k: v for k, v in artifact.items() if k != "pipeline"}
META_DST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"  저장: {MODEL_DST.name}")
print(f"  저장: {META_DST.name}")

# ── 최종 요약 ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("최종 모델 (acc-only, 클린 재학습)")
print("="*55)
print(f"  피처: {FEATS}")
print(f"  AUC:  {oof_auc:.4f} ± {np.std(auc_list):.4f}")
print(f"  threshold={best_thr:.3f}  sens={sens:.3f}  spec={spec:.3f}")
print(f"  signal_correction alpha={signal_correction.get('alpha'):.4f}")
