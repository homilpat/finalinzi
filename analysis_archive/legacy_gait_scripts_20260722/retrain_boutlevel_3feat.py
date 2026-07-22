"""
Bout-level 3피처 모델 재훈련 (데이터 누수 없음)

[핵심 아이디어]
  subject-level 집계(median/IQR → 1행/subject) 대신
  20s bout 행 그대로 사용(~100행/subject) → N=6400
  GroupKFold(subject_id)로 subject leakage 완전 차단

[피처] v_jerk_rms_median + v_harmonic_ratio_iqr + pitch_band_rms_median
  - 가속도 2개: subwindow_median_iqr_table.csv (per 20s bout)
  - pitch:      subwindow_pitch_table.csv (per subject, broadcast)

[Youden 임계값]
  OOF bout 확률 → subject별 평균 → subject-level Youden
  (추론 시 세션 1개 확률과 동일한 스케일)

[도메인 보정]
  기존 gait_daily_clinical_3feat_pitch.joblib에서 signal_correction + gyro_alpha 이어받기
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# ── 경로 ──────────────────────────────────────────────────────────
SUBWIN_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
PITCH_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_pitch_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
MODEL_SRC    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_pitch.joblib"
MODEL_DST    = ROOT / "MOCA" / "models" / "gait_daily_clinical_boutlevel.joblib"
META_DST     = ROOT / "MOCA" / "models" / "gait_daily_clinical_boutlevel_metadata.json"

NEW_FEATS   = ["v_jerk_rms_median", "v_harmonic_ratio_iqr", "pitch_band_rms_median"]
N_SPLIT     = 5
N_SEED      = 100


# ── 1. 데이터 구성 ────────────────────────────────────────────────
print("[1] Bout-level 데이터 구성")
sub  = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id")

# subject-level 집계 없이 bout 행 그대로 유지
df = sub.merge(clin_lbl, on="subject_id", how="inner", suffixes=("_drop", ""))
df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])

# pitch: subject-level broadcast (동일 subject의 모든 bout에 같은 값)
pitch_df = pd.read_csv(PITCH_CSV)[["subject_id", "pitch_band_rms_median"]]
df = df.merge(pitch_df, on="subject_id", how="inner", suffixes=("_drop", ""))
df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])
df = df.dropna(subset=NEW_FEATS).reset_index(drop=True)

X      = df[NEW_FEATS].values.astype(float)
y      = df["target"].values.astype(int)
groups = df["subject_id"].values

n_subj = df["subject_id"].nunique()
n0s    = (df.groupby("subject_id")["target"].first() == 0).sum()
n1s    = (df.groupby("subject_id")["target"].first() == 1).sum()
print(f"  {len(y)} bouts | {n_subj}명 (정상 {n0s} / 저하 {n1s})")
print(f"  bout 분포: 정상 {(y==0).sum()} / 저하 {(y==1).sum()}")


# ── 2. OOF AUC 평가 (GroupKFold, 100회) ──────────────────────────
print("\n[2] OOF AUC 평가 (GroupKFold, 100회)")

def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])

bout_aucs   = []   # per-bout AUC
subj_aucs   = []   # subject-averaged-prob AUC

for seed in range(N_SEED):
    rng  = np.random.default_rng(seed)
    # GroupKFold는 shuffle 미지원 → subject 순서를 seed마다 달리 섞어 GroupKFold 적용
    unique_subj = np.unique(groups)
    rng.shuffle(unique_subj)
    perm_map = {s: i for i, s in enumerate(unique_subj)}
    g_perm   = np.array([perm_map[g] for g in groups])

    gkf = GroupKFold(n_splits=N_SPLIT)
    bout_prob = np.full(len(y), np.nan)

    valid = True
    for tr, te in gkf.split(X, y, g_perm):
        if len(np.unique(y[te])) < 2:
            valid = False; break
        p = make_pipe()
        p.fit(X[tr], y[tr])
        bout_prob[te] = p.predict_proba(X[te])[:, 1]

    if not valid or np.any(np.isnan(bout_prob)):
        continue

    # subject-averaged probability
    subj_df_tmp = pd.DataFrame({"subject_id": groups, "prob": bout_prob, "label": y})
    subj_avg    = subj_df_tmp.groupby("subject_id").agg({"prob": "mean", "label": "first"})

    if len(np.unique(subj_avg["label"])) < 2:
        continue

    bout_aucs.append(roc_auc_score(y, bout_prob))
    subj_aucs.append(roc_auc_score(subj_avg["label"], subj_avg["prob"]))

bout_auc = float(np.mean(bout_aucs))
subj_auc = float(np.mean(subj_aucs))
print(f"  Bout-level  AUC: {bout_auc:.4f} ± {np.std(bout_aucs):.4f}  95CI [{np.percentile(bout_aucs,2.5):.3f}, {np.percentile(bout_aucs,97.5):.3f}]")
print(f"  Subject-avg AUC: {subj_auc:.4f} ± {np.std(subj_aucs):.4f}  95CI [{np.percentile(subj_aucs,2.5):.3f}, {np.percentile(subj_aucs,97.5):.3f}]")
print(f"  (기존 subject-level 모델 AUC: 0.8480)")


# ── 3. Youden 임계값 (subject-level OOF, seed=42) ─────────────────
print("\n[3] Youden 임계값 (subject-averaged OOF, seed=42)")
rng42 = np.random.default_rng(42)
unique_subj = np.unique(groups)
rng42.shuffle(unique_subj)
perm_map42  = {s: i for i, s in enumerate(unique_subj)}
g_perm42    = np.array([perm_map42[g] for g in groups])

gkf42 = GroupKFold(n_splits=N_SPLIT)
bout_prob42 = np.zeros(len(y))
for tr, te in gkf42.split(X, y, g_perm42):
    p = make_pipe(); p.fit(X[tr], y[tr])
    bout_prob42[te] = p.predict_proba(X[te])[:, 1]

# subject-level 집계
subj_df42 = pd.DataFrame({"subject_id": groups, "prob": bout_prob42, "label": y})
subj_avg42 = subj_df42.groupby("subject_id").agg({"prob": "mean", "label": "first"}).reset_index()
subj_probs  = subj_avg42["prob"].values
subj_labels = subj_avg42["label"].values

best_j, best_thr = -np.inf, 0.5
for t in np.linspace(0.05, 0.95, 181):
    pred = (subj_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(subj_labels, pred, labels=[0, 1]).ravel()
    j = (tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0) - 1
    if j > best_j:
        best_j, best_thr = j, t

pred_thr = (subj_probs >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(subj_labels, pred_thr, labels=[0, 1]).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp)
oof_subj_auc = roc_auc_score(subj_labels, subj_probs)
print(f"  OOF subject-level AUC (seed=42): {oof_subj_auc:.4f}")
print(f"  threshold={best_thr:.3f}  sens={sens:.3f}  spec={spec:.3f}")


# ── 4. 최종 모델 전체 데이터로 재훈련 ────────────────────────────
print("\n[4] 전체 6400 bouts로 최종 모델 훈련")
final_pipe = make_pipe()
final_pipe.fit(X, y)


# ── 5. 기존 보정값 이어받기 ───────────────────────────────────────
old_art = joblib.load(MODEL_SRC)
signal_correction = old_art.get("signal_correction")
gyro_alpha        = float(old_art.get("gyro_alpha", 1.0))
print(f"\n[5] 보정 이어받기: accel_alpha={signal_correction.get('alpha'):.4f}, gyro_alpha={gyro_alpha:.4f}")


# ── 6. 저장 ───────────────────────────────────────────────────────
print("\n[6] 모델 저장")
artifact = {
    "pipeline":           final_pipe,
    "features":           NEW_FEATS,
    "threshold":          float(best_thr),
    "threshold_strategy": "boutlevel_GroupKFold_Youden_subjectavg",
    "model_mode":         "daily_boutlevel_3feat_pitch_gyro",
    "label_source":       "motor_impairment_score >= 0.5",
    "train_n_normal":     int(n0s),
    "train_n_impaired":   int(n1s),
    "train_n_bouts":      int(len(y)),
    "oof_bout_auc":       round(bout_auc, 4),
    "oof_subject_auc":    round(subj_auc, 4),
    "oof_subject_sens":   round(sens, 4),
    "oof_subject_spec":   round(spec, 4),
    "signal_correction":  signal_correction,
    "gyro_alpha":         gyro_alpha,
    "gyro_alpha_method":  "feature_level_moment_matching",
}
joblib.dump(artifact, MODEL_DST)

meta = {k: v for k, v in artifact.items() if k != "pipeline"}
META_DST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"  저장: {MODEL_DST.name}")
print(f"  저장: {META_DST.name}")
print(f"\n=== 완료 ===")
print(f"  피처: {NEW_FEATS}")
print(f"  학습 bouts: {len(y)} ({n_subj}명)")
print(f"  Bout AUC: {bout_auc:.4f}  Subject AUC: {subj_auc:.4f}")
print(f"  threshold: {best_thr:.3f}  sens: {sens:.3f}  spec: {spec:.3f}")
print(f"  gyro_alpha: {gyro_alpha:.4f}  accel_alpha: {signal_correction.get('alpha'):.4f}")
