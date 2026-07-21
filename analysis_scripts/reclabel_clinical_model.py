"""
subwindow_median_iqr_table.csv (기존 캐시) + 임상 OR 라벨 → 재모델링
motor_impairment_score >= 0.5 → impaired(1), else normal(0)
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ── 경로 ─────────────────────────────────────────────────
TABLE_CSV = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(
    (ROOT.parent / p for p in [
        "파이널 보행 프로젝트/physionet_AWS/strict_preprocessing_runs/clinical_motor_label_modeling/subject_features_with_clinical.csv",
    ]),
    None
)
if CLINICAL_CSV is None or not CLINICAL_CSV.exists():
    # fallback 탐색
    import glob
    hits = list(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))
    CLINICAL_CSV = hits[0] if hits else None

MODEL_DIR = ROOT / "MOCA" / "models"

BASE = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz",
        "ap_spec_entropy", "v_jerk_rms"]
FEATS = [f"{f}_median" for f in BASE] + [f"{f}_iqr" for f in BASE]

# ── 로드 ─────────────────────────────────────────────────
table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin  = clin[["subject_id", "motor_impairment_score"]].copy()
clin["motor_impairment_score"] = pd.to_numeric(clin["motor_impairment_score"], errors="coerce")

# 임상 라벨 기준: motor_impairment_score >= 0.5 → impaired
THRESHOLD = 0.5
clin["clinical_target"] = (clin["motor_impairment_score"] >= THRESHOLD).astype(int)

table = table.merge(clin, on="subject_id", how="inner")

# ── 임상 라벨 vs 원래 CO/FL 라벨 비교 ────────────────────
subj = table.drop_duplicates("subject_id")[["subject_id","group","motor_impairment_score","clinical_target"]]
orig_n1 = (subj["group"] == "Faller").sum()
new_n1  = subj["clinical_target"].sum()
changed = subj[subj["group"].map({"Control":0,"Faller":1}) != subj["clinical_target"]]
print(f"[라벨 비교]")
print(f"  원래 Faller(CO/FL): n={orig_n1} / {len(subj)}")
print(f"  임상 impaired(≥{THRESHOLD}): n={new_n1} / {len(subj)}")
print(f"  라벨 변경된 subject: {len(changed)}명")
if len(changed) > 0:
    for _, r in changed.iterrows():
        orig = "FL" if r["group"]=="Faller" else "CO"
        print(f"    {r['subject_id']:8s}  원래={orig}  score={r['motor_impairment_score']:.3f}  → clinical={'impaired' if r['clinical_target'] else 'normal'}")

# ── 피처 분포 (임상 라벨 기준) ───────────────────────────
print(f"\n[피처 분포] (clinical_target별 중앙값, 윈도우 수준)")
subj_med = table.groupby(["subject_id","clinical_target"])[FEATS].median().reset_index()
print(f"  {'피처':38s}  {'정상(0)':>8}  {'저하(1)':>8}  {'diff':>8}")
for f in FEATS:
    g = subj_med.groupby("clinical_target")[f].median()
    v0, v1 = g.get(0, np.nan), g.get(1, np.nan)
    diff = v1 - v0 if np.isfinite(v0) and np.isfinite(v1) else np.nan
    flag = " *" if np.isfinite(diff) and abs(diff) > 0.03 else ""
    print(f"  {f:38s}  {v0:8.4f}  {v1:8.4f}  {diff:+8.4f}{flag}")

# ── OOF GroupKFold (group=subject_id) ────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0,1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.
        sp = tn/(tn+fp) if tn+fp else 0.
        if s + sp - 1 > best_j: best_j, best_t = s + sp - 1, t
    return float(best_t)

train  = table.dropna(subset=FEATS).copy()
y      = train["clinical_target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()
n0     = int((y == 0).sum())
n1     = int((y == 1).sum())

print(f"\n[OOF] 윈도우={len(train)}개  정상={n0}  저하={n1}  (subject={table['subject_id'].nunique()}명)")

# 피처 조합별 비교
results = {}
for label, feats in [
    ("MEDIAN만",   [f for f in FEATS if "_median" in f]),
    ("IQR만",      [f for f in FEATS if "_iqr"    in f]),
    ("MEDIAN+IQR", FEATS),
]:
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[feats], y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        pipe.fit(train.iloc[tr][feats].to_numpy(), y[tr])
        oof[te] = pipe.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]
    thr  = youden_thr(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y, pred, labels=[0,1]).ravel()
    s  = tp_/(tp_+fn) if tp_+fn else 0.
    sp = tn/(tn+fp)   if tn+fp  else 0.
    auc = roc_auc_score(y, oof)
    results[label] = {"oof": oof, "auc": auc, "sens": s, "spec": sp, "thr": thr}
    print(f"  [{label:12s}]  AUC={auc:.4f}  sens={s:.4f}  spec={sp:.4f}  thr={thr:.3f}")

# ── MEDIAN+IQR 최종: sens≥0.80 threshold ─────────────────
oof_best = results["MEDIAN+IQR"]["oof"]
best_thr, best_spec = 0.5, -1.
for t in np.linspace(0.05, 0.95, 181):
    pred_t = (oof_best >= t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y, pred_t, labels=[0,1]).ravel()
    s2  = tp2/(tp2+fn2) if tp2+fn2 else 0.
    sp2 = tn2/(tn2+fp2) if tn2+fp2 else 0.
    if s2 >= 0.80 and sp2 > best_spec:
        best_thr, best_spec = float(t), float(sp2)

pred80 = (oof_best >= best_thr).astype(int)
tn80, fp80, fn80, tp80 = confusion_matrix(y, pred80, labels=[0,1]).ravel()
s80  = tp80/(tp80+fn80) if tp80+fn80 else 0.
sp80 = tn80/(tn80+fp80) if tn80+fp80 else 0.
auc_best = results["MEDIAN+IQR"]["auc"]
print(f"\n[sens≥0.80 최적 thr={best_thr:.3f}]  sens={s80:.4f}  spec={sp80:.4f}  AUC={auc_best:.4f}")
print(f"  TP={tp80} FN={fn80} TN={tn80} FP={fp80}  (정상={n0//max(n0,1)*0+int((y==0).sum())}행 저하={int((y==1).sum())}행)")

# ── 최종 모델 저장 ────────────────────────────────────────
pipe_final = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  RobustScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
])
pipe_final.fit(train[FEATS].to_numpy(), y)

artifact = {
    "pipeline":  pipe_final,
    "features":  FEATS,
    "threshold": best_thr,
    "threshold_strategy": "sens_geq_0.80_max_spec",
    "model_mode": "daily_walk_75h_subwindow_clinical_label",
    "label_source": f"motor_impairment_score >= {THRESHOLD}",
    "train_n_normal":   int((y==0).sum()),
    "train_n_impaired": int((y==1).sum()),
    "oof_auc":  round(auc_best, 4),
    "oof_sens": round(s80, 4),
    "oof_spec": round(sp80, 4),
    "reference": "Moe-Nilssen & Helbostad 2004; PhysioNet 75h, clinical OR label",
}
out_path = MODEL_DIR / "gait_daily_clinical_subwindow.joblib"
joblib.dump(artifact, out_path)

meta = {
    "features": FEATS, "threshold": best_thr,
    "label_source": f"motor_impairment_score >= {THRESHOLD}",
    "oof": {"auc": round(auc_best,4), "sens": round(s80,4), "spec": round(sp80,4), "thr": best_thr},
}
(MODEL_DIR / "gait_daily_clinical_subwindow_metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\n[저장] {out_path}")
print("[완료]")
