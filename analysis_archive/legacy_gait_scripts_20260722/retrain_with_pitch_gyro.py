"""
pitch_band_rms__iqr 추가 3피처 모델 재훈련 + 자이로 도메인 보정

[피처]
  v_jerk_rms_median   : subwindow_median_iqr_table.csv (가속도, alpha 보정)
  v_harmonic_ratio_iqr: subwindow_median_iqr_table.csv (가속도, alpha 보정)
  pitch_band_rms__iqr : subject_features_with_clinical.csv (자이로, gyro_alpha 보정)

[도메인 보정]
  - 가속도 (signal-level): 기존 alpha=1.9705, tau=1.0 유지
  - 자이로 (feature-level moment matching):
      gyro_alpha = mean(PhysioNet 정상 pitch_band_rms__iqr) /
                   mean(우리 앱 정상 pitch_band_rms__iqr)
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
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from gait_axis_aligned_core import (
    load_sensor_csv_with_metadata, _acc_columns, align_to_vmlap,
    resample_array_to_100hz, bandpass,
    _DAILY_WIN20, _DAILY_SUB_WIN, _DAILY_STEP, TARGET_FS_HZ,
)

# ── 경로 ──────────────────────────────────────────────────────────
SUBWIN_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
PITCH_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_pitch_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
SAMPLE_DIR   = ROOT / "보행SAMPLE"
MODEL_SRC    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat.joblib"
MODEL_DST    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_pitch.joblib"
META_DST     = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_pitch_metadata.json"

# pitch_band_rms_median 또는 pitch_band_rms_iqr 중 선택
# median: sub-window 방식으로 뽑아서 세션 길이에 덜 민감
PITCH_FEAT = "pitch_band_rms_median"   # deg/s (sub-window 방식, PhysioNet LabWalk에서 추출)
NEW_FEATS  = ["v_jerk_rms_median", "v_harmonic_ratio_iqr", PITCH_FEAT]
ACCEL_FEATS = ["v_jerk_rms_median", "v_harmonic_ratio_iqr"]

NORMAL_FILES = [
    "hazi_gait_calibrated_20s_20260715_155029.csv",
    "hazi_gait_anatomical_14cols_20260715_163129.csv",
    "hazi_gait_anatomical_14cols_20260719_161127_80대_2.csv",
    "hazi_gait_anatomical_14cols_20260721_125400.csv",
    "hazi_gait_anatomical_14cols_20260721_125240.csv",
    "hazi_gait_anatomical_14cols_20260721_130032.csv",
]


# ── 1. 훈련 데이터 구성 ───────────────────────────────────────────
print("[1] 훈련 데이터 구성")
sub = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id")

sub = sub.merge(clin_lbl, on="subject_id", how="inner", suffixes=("_drop", ""))
sub = sub.drop(columns=[c for c in sub.columns if c.endswith("_drop")])

subj_accel = (
    sub.groupby("subject_id")[ACCEL_FEATS + ["target"]]
    .agg({f: "median" for f in ACCEL_FEATS} | {"target": "first"})
    .reset_index()
)

# pitch: sub-window 방식으로 새로 추출한 PhysioNet pitch (deg/s)
pitch_df = pd.read_csv(PITCH_CSV)[["subject_id", PITCH_FEAT]]
df_train = subj_accel.merge(pitch_df, on="subject_id", how="inner")
df_train = df_train.dropna(subset=NEW_FEATS).reset_index(drop=True)

y      = df_train["target"].values.astype(int)
groups = df_train["subject_id"].values
n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"  {len(y)}명 | 정상 {n0} / 저하 {n1}")


# ── 2. OOF AUC 평가 (StratifiedGroupKFold) ───────────────────────
print("\n[2] OOF AUC 평가 (StratifiedGroupKFold 5-fold, 100회)")
pipe = lambda: Pipeline([
    ("imp",  SimpleImputer(strategy="median")),
    ("sc",   RobustScaler()),
    ("clf",  LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
])
X = df_train[NEW_FEATS].values.astype(float)

auc_list = []
for seed in range(100):
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        p = pipe()
        p.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X[te])[:, 1]))
    auc_list.append(float(np.mean(aucs)))

oof_auc = float(np.mean(auc_list))
print(f"  AUC: {oof_auc:.4f} ± {np.std(auc_list):.4f}  95CI [{np.percentile(auc_list,2.5):.3f}, {np.percentile(auc_list,97.5):.3f}]")


# ── 3. Youden 임계값 (전체 OOF) ───────────────────────────────────
print("\n[3] Youden 임계값 계산")
sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
oof_prob = np.zeros(len(y))
for tr, te in sgkf.split(X, y, groups):
    p = pipe(); p.fit(X[tr], y[tr])
    oof_prob[te] = p.predict_proba(X[te])[:, 1]

best_j, best_thr = -np.inf, 0.5
for t in np.linspace(0.05, 0.95, 181):
    pred = (oof_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    j = (tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0) - 1
    if j > best_j:
        best_j, best_thr = j, t

pred_thr = (oof_prob >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_thr, labels=[0, 1]).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp)
print(f"  threshold={best_thr:.3f}  sens={sens:.3f}  spec={spec:.3f}")


# ── 4. 최종 모델 전체 데이터로 재훈련 ────────────────────────────
print("\n[4] 전체 데이터로 최종 모델 훈련")
final_pipe = pipe()
final_pipe.fit(X, y)


# ── 5. 자이로 도메인 보정: 앱 정상 샘플에서 pitch_band_rms__iqr ──
print("\n[5] 자이로 도메인 보정 (feature-level moment matching)")

def extract_pitch_band_rms_iqr_from_csv(csv_path: Path) -> float | None:
    df, metadata = load_sensor_csv_with_metadata(str(csv_path))
    gyro_cols = ["Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z"]
    if not all(c in df.columns for c in gyro_cols):
        return None

    # ML axis 결정
    raw_acc_cols = ["Acc_X", "Acc_Y", "Acc_Z"]
    if all(c in df.columns for c in raw_acc_cols):
        acc = df[raw_acc_cols].to_numpy(float)
        try:
            t = df["Timestamp_ns"].to_numpy(float)
            dur = (t.max() - t.min()) / 1e9
            obs_fs = len(df) / dur if dur > 0 else TARGET_FS_HZ
            _, align_info = align_to_vmlap(acc, already_vmlap=False, fs=obs_fs)
            ml_idx = align_info.get("ml_raw_axis")
        except Exception:
            ml_idx = None
    else:
        ml_idx = None

    if ml_idx is None:
        return None

    t = df["Timestamp_ns"].to_numpy(float)
    dur = (t.max() - t.min()) / 1e9
    obs_fs = len(df) / dur if dur > 0 else TARGET_FS_HZ

    gyro_raw = df[gyro_cols].to_numpy(float)
    pitch_raw = gyro_raw[:, int(ml_idx)]
    pitch_100hz = resample_array_to_100hz(pitch_raw.reshape(-1, 1), obs_fs)[:, 0]

    n = len(pitch_100hz)
    seg_starts = list(range(0, max(1, n - _DAILY_WIN20 + 1), _DAILY_WIN20 // 2)) or [0]
    rms_list = []
    for w0 in seg_starts:
        seg = pitch_100hz[w0 : w0 + _DAILY_WIN20]
        if len(seg) < int(0.5 * _DAILY_WIN20):
            continue
        for s in range(0, max(1, len(seg) - _DAILY_SUB_WIN + 1), _DAILY_STEP):
            sub = seg[s : s + _DAILY_SUB_WIN]
            if len(sub) < int(0.8 * _DAILY_SUB_WIN):
                continue
            bp = bandpass(sub, TARGET_FS_HZ)
            rms_list.append(float(np.sqrt(np.nanmean(bp ** 2))))

    if len(rms_list) < 2:
        return None
    # rad/s → deg/s 변환 (앱 gyro는 rad/s, PhysioNet은 deg/s)
    rms_arr_dps = np.array(rms_list) * 57.2958
    return float(np.median(rms_arr_dps))   # pitch_band_rms_median (deg/s)


our_pitch_vals = []
for fn in NORMAL_FILES:
    p = SAMPLE_DIR / fn
    if not p.exists():
        print(f"  [없음] {fn}")
        continue
    val = extract_pitch_band_rms_iqr_from_csv(p)
    if val is not None:
        our_pitch_vals.append(val)
        print(f"  OK  {fn[-30:]}  pitch_band_rms__iqr={val:.5f} rad/s")
    else:
        print(f"  NG  {fn[-30:]}  (gyro 없음 또는 추출 실패)")

if len(our_pitch_vals) < 2:
    print("  [경고] 정상 샘플 부족 → gyro_alpha=1.0 (보정 없음)")
    gyro_alpha = 1.0
else:
    our_mean = float(np.mean(our_pitch_vals))
    pn_normals = df_train[df_train["target"] == 0][PITCH_FEAT]
    pn_mean = float(pn_normals.mean())
    gyro_alpha = pn_mean / our_mean
    print(f"\n  PhysioNet 정상 평균: {pn_mean:.3f} deg/s")
    print(f"  우리 앱 정상 평균:   {our_mean:.3f} deg/s  (rad/s→deg/s 변환 포함)")
    print(f"  gyro_alpha = {gyro_alpha:.4f}")


# ── 6. 기존 가속도 보정 이어받기 ─────────────────────────────────
old_art = joblib.load(MODEL_SRC)
signal_correction = old_art.get("signal_correction")
print(f"\n[6] 가속도 보정 이어받기: alpha={signal_correction.get('alpha'):.4f}, tau={signal_correction.get('tau'):.4f}")


# ── 7. 모델 아티팩트 저장 ─────────────────────────────────────────
print("\n[7] 모델 저장")
artifact = {
    "pipeline":          final_pipe,
    "features":          NEW_FEATS,
    "threshold":         float(best_thr),
    "threshold_strategy":"StratifiedGroupKFold_Youden",
    "model_mode":        "daily_3feat_pitch_gyro",
    "label_source":      "motor_impairment_score >= 0.5",
    "train_n_normal":    int(n0),
    "train_n_impaired":  int(n1),
    "oof_subject_auc":   round(oof_auc, 4),
    "oof_subject_sens":  round(sens, 4),
    "oof_subject_spec":  round(spec, 4),
    "signal_correction": signal_correction,
    "gyro_alpha":        float(gyro_alpha),
    "gyro_alpha_method": "feature_level_moment_matching",
}
joblib.dump(artifact, MODEL_DST)

meta = {
    "features":          NEW_FEATS,
    "threshold":         float(best_thr),
    "threshold_strategy":"StratifiedGroupKFold_Youden",
    "oof_subject_auc":   round(oof_auc, 4),
    "oof_subject_sens":  round(sens, 4),
    "oof_subject_spec":  round(spec, 4),
    "signal_correction": signal_correction,
    "gyro_alpha":        float(gyro_alpha),
    "gyro_alpha_method": "feature_level_moment_matching",
    "n_our_normals_gyro": len(our_pitch_vals),
}
META_DST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"  저장: {MODEL_DST.name}")
print(f"  저장: {META_DST.name}")
print(f"\n=== 완료 ===")
print(f"  피처: {NEW_FEATS}")
print(f"  AUC: {oof_auc:.4f}  threshold: {best_thr:.3f}")
print(f"  gyro_alpha: {gyro_alpha:.4f}  accel_alpha: {signal_correction.get('alpha'):.4f}")
