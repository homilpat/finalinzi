"""
Best-10s 방식 3피처 재훈련 (추론 파이프라인 완전 일치)

[훈련 파이프라인]
  75h .dat → 20s 보행 창 → best quality 10s sub-window 선택
  → v_jerk_rms, v_harmonic_ratio (acc)
  → subject별 median 집계 → LR 모델

[pitch]
  75h .dat에 gyro 없음 → subwindow_pitch_table.csv (LabWalk best10 대신
  동일 subject-level pitch_band_rms_median) 병합

[추론 파이프라인 (일치)]
  앱 20s 녹화 → extract_best10_daily_features_from_vmlap
  → v_jerk_rms, v_harmonic_ratio, pitch_band_rms (best 10s 창 기준)
"""
from __future__ import annotations
import re, sys, json
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

from gait_axis_aligned_core import (
    TARGET_FS_HZ, window_features, bandpass,
    load_sensor_csv_with_metadata, _acc_columns, align_to_vmlap,
    resample_array_to_100hz, BEST10_FEATURES,
)

# ── 경로 ──────────────────────────────────────────────────────────
GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR      = GAIT_PROJECT / "physionet_AWS"
V2_CSV       = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
PITCH_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_pitch_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
MODEL_SRC    = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_pitch.joblib"
MODEL_DST    = ROOT / "MOCA" / "models" / "gait_daily_best10_3feat.joblib"
META_DST     = ROOT / "MOCA" / "models" / "gait_daily_best10_3feat_metadata.json"
SAMPLE_DIR   = ROOT / "보행SAMPLE"

FS       = int(TARGET_FS_HZ)
WIN20    = int(20 * FS)
SUB_WIN  = int(10 * FS)
SUB_STEP = int(2  * FS)
MAX_SEGS = 100

ACC_FEATS   = ["v_jerk_rms", "v_harmonic_ratio"]   # best10 창 단위 피처 (suffix 없음)
N_SPLIT = 5
N_SEED  = 100

NORMAL_FILES = [
    "hazi_gait_calibrated_20s_20260715_155029.csv",
    "hazi_gait_anatomical_14cols_20260715_163129.csv",
    "hazi_gait_anatomical_14cols_20260719_161127_80대_2.csv",
    "hazi_gait_anatomical_14cols_20260721_125400.csv",
    "hazi_gait_anatomical_14cols_20260721_125240.csv",
    "hazi_gait_anatomical_14cols_20260721_130032.csv",
]


# ── hea / dat 읽기 (기존 build_75h 스크립트와 동일) ───────────────
def parse_hea(sid: str) -> dict:
    lines = (RAW_DIR / f"{sid}.hea").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    n, ch = int(parts[3]), int(parts[1])
    gains, baselines = [], []
    for line in lines[1:1+ch]:
        m = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(m.group(1))); baselines.append(float(m.group(2)))
    return {"n": n, "ch": ch,
            "gains": np.array(gains[:3]), "baselines": np.array(baselines[:3])}


def read_20s(sid: str, start_sec: float, hea: dict) -> np.ndarray | None:
    dat = RAW_DIR / f"{sid}.dat"
    if not dat.exists(): return None
    s = int(round(start_sec * FS))
    e = s + WIN20
    if e > hea["n"]: return None
    try:
        raw = np.memmap(dat, dtype="<i2", mode="r", shape=(hea["n"], hea["ch"]))
        seg = raw[s:e, :3].astype(float)
        return (seg - hea["baselines"]) / hea["gains"]
    except Exception:
        return None


# ── 핵심 변경: median/IQR → best 10s sub-window ──────────────────
def subwindow_best10(vmlap_20s: np.ndarray) -> dict | None:
    """20s 창에서 quality_score(=v_stride_peak) 최고인 10s 창 하나 선택."""
    best_feat  = None
    best_score = -np.inf
    for s in range(0, WIN20 - SUB_WIN + 1, SUB_STEP):
        sub = vmlap_20s[s:s + SUB_WIN]
        if len(sub) < int(0.8 * SUB_WIN):
            continue
        try:
            feat  = window_features(sub)
            score = feat.get("quality_score", -np.inf)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_feat  = feat
        except Exception:
            continue
    if best_feat is None:
        return None
    return {
        "v_jerk_rms":       float(best_feat.get("v_jerk_rms",       np.nan)),
        "v_harmonic_ratio": float(best_feat.get("v_harmonic_ratio", np.nan)),
    }


# ── 1. 75h 데이터에서 best10 피처 추출 ───────────────────────────
print("[1] 75h 데이터 → 20s 창 → best 10s sub-window 피처 추출")
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id").set_index("subject_id")

v2 = pd.read_csv(V2_CSV)
v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
v2 = v2.dropna(subset=["start_sec"])
subjects = v2[["subject_id", "group"]].drop_duplicates().sort_values("subject_id")

all_rows = []
for _, s in subjects.iterrows():
    sid, group = s["subject_id"], s["group"]
    if sid not in clin_lbl.index:
        continue
    segs = v2[v2["subject_id"].eq(sid)]
    if len(segs) > MAX_SEGS:
        segs = segs.sample(MAX_SEGS, random_state=42)

    try:
        hea = parse_hea(sid)
    except Exception:
        print(f"  [skip] {sid} hea 오류")
        continue

    seg_rows = []
    for _, row in segs.iterrows():
        vmlap = read_20s(sid, float(row["start_sec"]), hea)
        if vmlap is None: continue
        feat = subwindow_best10(vmlap)
        if feat is None: continue
        feat["subject_id"] = sid
        feat["group"]      = group
        feat["target"]     = int(clin_lbl.loc[sid, "target"])
        seg_rows.append(feat)

    all_rows.extend(seg_rows)
    if seg_rows:
        ex = seg_rows[0]
        print(f"  {sid:8s} ({'저하' if ex['target'] else '정상':2s})  "
              f"{len(seg_rows):3d}개 20s  "
              f"jerk_best={ex['v_jerk_rms']:.3f}  "
              f"hr_best={ex['v_harmonic_ratio']:.3f}")
    else:
        print(f"  [skip] {sid} 유효 구간 없음")

table = pd.DataFrame(all_rows)
print(f"\n  {table['subject_id'].nunique()}명  {len(table)}개 20s 창 추출")


# ── 2. subject-level 집계 (best10 창들의 median) ─────────────────
print("\n[2] subject-level 집계 (per-20s-bout best10 median)")
subj_acc = (
    table.groupby("subject_id")[ACC_FEATS + ["target"]]
    .agg({f: "median" for f in ACC_FEATS} | {"target": "first"})
    .reset_index()
)

# pitch: LabWalk에서 추출한 subject-level pitch_band_rms_median 병합
pitch_df = pd.read_csv(PITCH_CSV)[["subject_id", "pitch_band_rms_median"]]
df_train = subj_acc.merge(pitch_df, on="subject_id", how="inner")
df_train = df_train.rename(columns={"pitch_band_rms_median": "pitch_band_rms"})
df_train = df_train.dropna(subset=BEST10_FEATURES).reset_index(drop=True)

y      = df_train["target"].values.astype(int)
groups = df_train["subject_id"].values
n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"  최종 {len(y)}명 | 정상 {n0} / 저하 {n1}")

# 피처 분포
print(f"\n  {'피처':25s}  {'정상':>8}  {'저하':>8}")
for f in BEST10_FEATURES:
    g0 = df_train[df_train.target==0][f].median()
    g1 = df_train[df_train.target==1][f].median()
    print(f"  {f:25s}  {g0:8.4f}  {g1:8.4f}")


# ── 3. OOF AUC (StratifiedGroupKFold 100회) ──────────────────────
print("\n[3] OOF AUC 평가 (StratifiedGroupKFold, 100회)")
X = df_train[BEST10_FEATURES].values.astype(float)

def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])

auc_list = []
for seed in range(N_SEED):
    sgkf = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2: continue
        p = make_pipe(); p.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X[te])[:, 1]))
    if aucs:
        auc_list.append(float(np.mean(aucs)))

oof_auc = float(np.mean(auc_list))
print(f"  AUC: {oof_auc:.4f} ± {np.std(auc_list):.4f}  "
      f"95CI [{np.percentile(auc_list,2.5):.3f}, {np.percentile(auc_list,97.5):.3f}]")
print(f"  (기존 subject-level 모델: 0.8480)")


# ── 4. Youden 임계값 (OOF seed=42) ───────────────────────────────
print("\n[4] Youden 임계값")
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
    if j > best_j: best_j, best_thr = j, t

pred_thr = (oof_prob >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_thr, labels=[0, 1]).ravel()
sens = tp/(tp+fn); spec = tn/(tn+fp)
print(f"  threshold={best_thr:.3f}  sens={sens:.3f}  spec={spec:.3f}")


# ── 5. 최종 모델 훈련 ─────────────────────────────────────────────
print("\n[5] 전체 데이터로 최종 모델 훈련")
final_pipe = make_pipe()
final_pipe.fit(X, y)


# ── 6. gyro_alpha: 앱 정상 샘플에서 best10 pitch 추출 ────────────
print("\n[6] gyro_alpha 계산")

def extract_best10_pitch_from_csv(csv_path: Path) -> float | None:
    df, metadata = load_sensor_csv_with_metadata(str(csv_path))
    gyro_cols    = ["Gyro_Clean_X", "Gyro_Clean_Y", "Gyro_Clean_Z"]
    raw_acc_cols = ["Acc_X", "Acc_Y", "Acc_Z"]
    if not all(c in df.columns for c in gyro_cols + raw_acc_cols):
        return None
    t = df["Timestamp_ns"].to_numpy(float)
    dur = (t.max() - t.min()) / 1e9
    obs_fs = len(df) / dur if dur > 0 else TARGET_FS_HZ

    acc = df[raw_acc_cols].to_numpy(float)
    try:
        _, align_info = align_to_vmlap(acc, already_vmlap=False, fs=obs_fs)
        ml_idx = align_info.get("ml_raw_axis")
    except Exception:
        return None
    if ml_idx is None:
        return None

    gyro_raw   = df[gyro_cols].to_numpy(float)
    pitch_raw  = gyro_raw[:, int(ml_idx)]
    pitch_100  = resample_array_to_100hz(pitch_raw.reshape(-1, 1), obs_fs)[:, 0]

    acc_arr, already_vmlap, _, _ = _acc_columns(df, metadata)
    aligned, _ = align_to_vmlap(acc_arr, already_vmlap=already_vmlap, fs=obs_fs)
    vmlap_100  = resample_array_to_100hz(aligned, obs_fs)

    # best 10s 창 위치 찾기 (vmlap 기준)
    win = SUB_WIN
    best_start = 0; best_score = -np.inf
    for s in range(0, max(1, len(vmlap_100) - win + 1), SUB_STEP):
        sub = vmlap_100[s:s + win]
        if len(sub) < int(0.8 * win): continue
        try:
            feat  = window_features(sub)
            score = feat.get("quality_score", -np.inf)
            if np.isfinite(score) and score > best_score:
                best_score = score; best_start = s
        except Exception:
            continue

    sub_pitch = pitch_100[best_start:best_start + win]
    if len(sub_pitch) < int(0.8 * win):
        return None
    bp  = bandpass(sub_pitch, TARGET_FS_HZ)
    rms = float(np.sqrt(np.nanmean(bp ** 2)))   # rad/s
    return rms * 57.2958                          # → deg/s

our_pitch_vals = []
for fn in NORMAL_FILES:
    p = SAMPLE_DIR / fn
    if not p.exists():
        print(f"  [없음] {fn}"); continue
    val = extract_best10_pitch_from_csv(p)
    if val is not None:
        our_pitch_vals.append(val)
        print(f"  OK  {fn[-35:]}  pitch={val:.3f} dps")
    else:
        print(f"  NG  {fn[-35:]}  (추출 실패)")

if len(our_pitch_vals) < 2:
    print("  [경고] 정상 샘플 부족 → gyro_alpha=1.0")
    gyro_alpha = 1.0
else:
    our_mean = float(np.mean(our_pitch_vals))
    pn_mean  = float(df_train[df_train.target==0]["pitch_band_rms"].mean())
    gyro_alpha = pn_mean / our_mean
    print(f"\n  PhysioNet 정상 평균: {pn_mean:.3f} dps")
    print(f"  우리 앱 정상 평균:   {our_mean:.3f} dps")
    print(f"  gyro_alpha = {gyro_alpha:.4f}")


# ── 7. 가속도 보정 이어받기 ───────────────────────────────────────
old_art = joblib.load(MODEL_SRC)
signal_correction = old_art.get("signal_correction")
print(f"\n[7] 가속도 보정 이어받기: alpha={signal_correction.get('alpha'):.4f}")


# ── 8. 저장 ───────────────────────────────────────────────────────
print("\n[8] 모델 저장")
artifact = {
    "pipeline":           final_pipe,
    "features":           BEST10_FEATURES,
    "threshold":          float(best_thr),
    "threshold_strategy": "best10_StratifiedGroupKFold_Youden",
    "model_mode":         "daily_best10_75h_3feat_pitch_gyro",
    "label_source":       "motor_impairment_score >= 0.5",
    "train_n_normal":     int(n0),
    "train_n_impaired":   int(n1),
    "oof_subject_auc":    round(oof_auc, 4),
    "oof_subject_sens":   round(sens, 4),
    "oof_subject_spec":   round(spec, 4),
    "signal_correction":  signal_correction,
    "gyro_alpha":         float(gyro_alpha),
    "gyro_alpha_method":  "best10_feature_level_moment_matching",
}
joblib.dump(artifact, MODEL_DST)
meta = {k: v for k, v in artifact.items() if k != "pipeline"}
META_DST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"  저장: {MODEL_DST.name}")
print(f"\n=== 완료 ===")
print(f"  피처: {BEST10_FEATURES}")
print(f"  AUC: {oof_auc:.4f}  threshold: {best_thr:.3f}  sens: {sens:.3f}  spec: {spec:.3f}")
print(f"  gyro_alpha: {gyro_alpha:.4f}  accel_alpha: {signal_correction.get('alpha'):.4f}")
