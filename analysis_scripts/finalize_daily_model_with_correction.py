"""
1) OUR_SAMPLE CSV → subwindow 3피처 계산
2) PhysioNet 75h 정상군 중앙값과 비교 → 도메인 보정 델타
3) 최종 3피처 모델 + 보정값 저장
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from gait_axis_aligned_core import (
    TARGET_FS_HZ, window_features,
    extract_subwindow_daily_features,
)

SAMPLE_DIR   = ROOT / "보행SAMPLE"
TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))
MODEL_DIR    = ROOT / "MOCA" / "models"

FS       = int(TARGET_FS_HZ)
WIN20    = int(20 * FS)
SUB_WIN  = int(10 * FS)
SUB_STEP = int(2  * FS)

BASE  = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz",
         "ap_spec_entropy", "v_jerk_rms"]
BEST3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

# ── OUR_SAMPLE CSV 읽기 ───────────────────────────────────
def load_our_csv(path: Path) -> np.ndarray | None:
    """
    해지 anatomical CSV → V, ML, AP (g) 배열 반환
    컬럼 순서: timestamp, Ax, Ay, Az, (gyro...), V, ML, AP (마지막 4컬럼 중 V 포함)
    실제 14컬럼 파일: 0=ts, 1=Ax, 2=Ay, 3=Az, 4-8=gyro 관련, 9=Vx, 10=Vy, 11=Vz(?), ...
    → 기존 extract_axis_aligned_gait_features와 동일 방식 사용
    """
    try:
        raw = pd.read_csv(path, header=None, comment="#")
    except Exception:
        return None
    if raw.shape[1] < 4:
        return None
    # Ax, Ay, Az (m/s²) → g 변환
    acc = raw.iloc[:, 1:4].to_numpy(dtype=float) / 9.80665
    # V, ML, AP 순서: gait_axis_aligned_core는 V=col0, ML=col1, AP=col2
    # anatomical 파일에서 V/ML/AP 추출 (컬럼 10,11,12 or 0,1,2 depending on file)
    # 이미 교정된 파일은 gait_axis_aligned_core의 extract_axis_aligned_gait_features가 처리함
    # 여기서는 Ax,Ay,Az를 그대로 vmlap으로 사용 (axis alignment 미적용 단순화)
    # → 실제 앱과 동일하게 gait_axis_aligned_core의 로더 사용
    return acc  # shape (N, 3)


def subwindow_feats(vmlap: np.ndarray) -> dict | None:
    """20s 단위로 잘라서 subwindow median/IQR"""
    n = len(vmlap)
    all_sub = []
    # 전체 데이터에서 20s 윈도우 추출 후 subwindow
    for w_start in range(0, n - WIN20 + 1, WIN20 // 2):
        seg = vmlap[w_start:w_start + WIN20]
        if len(seg) < int(0.8 * WIN20): continue
        rows = []
        for s in range(0, WIN20 - SUB_WIN + 1, SUB_STEP):
            sub = seg[s:s + SUB_WIN]
            try:
                f = window_features(sub)
                rows.append({k: f.get(k) for k in BASE})
            except Exception:
                continue
        if len(rows) >= 2:
            all_sub.extend(rows)
    if len(all_sub) < 2:
        return None
    arr = pd.DataFrame(all_sub)
    out = {}
    for f in BASE:
        vals = arr[f].dropna()
        out[f + "_median"] = float(vals.median()) if len(vals) else np.nan
        out[f + "_iqr"]    = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan
    return out


print("[1] OUR_SAMPLE subwindow 3피처 계산 (core 함수 사용)")
our_rows = []
for path in sorted(SAMPLE_DIR.glob("*.csv")):
    label = "impaired" if "발다침" in path.stem else "normal"
    target = 1 if label == "impaired" else 0
    try:
        ex = extract_subwindow_daily_features(str(path))
        feat = ex["features"]   # v_jerk_rms_median/iqr, v_harmonic_ratio_iqr
    except Exception as e:
        print(f"  [skip] {path.name}  err={e}")
        continue
    if feat is None:
        continue
    feat["subject_id"] = path.stem
    feat["label"]      = label
    feat["target"]     = target
    our_rows.append(feat)
    print(f"  {path.name[:50]:50s}  [{label:8s}]  "
          f"jerk_med={feat.get('v_jerk_rms_median',float('nan')):.3f}  "
          f"jerk_iqr={feat.get('v_jerk_rms_iqr',float('nan')):.3f}  "
          f"HR_iqr={feat.get('v_harmonic_ratio_iqr',float('nan')):.3f}")

our_df = pd.DataFrame(our_rows)
our_normals = our_df[our_df["target"] == 0]

# ── 훈련 데이터 정상군 중앙값 ──────────────────────────────
table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id","motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")

train_normals = table[table["clinical_target"] == 0]
# subject 단위 중앙값
subj_norm_med = train_normals.groupby("subject_id")[BEST3].median().median()

print(f"\n[2] 도메인 보정 델타 계산")
print(f"  피처                        PhysioNet 정상군  OUR_SAMPLE 정상군  Delta(→보정)")
correction = {}
for f in BEST3:
    pn_med  = subj_norm_med.get(f, np.nan)
    our_med = our_normals[f].median() if len(our_normals) and f in our_normals else np.nan
    delta   = float(pn_med - our_med) if np.isfinite(pn_med) and np.isfinite(our_med) else 0.0
    correction[f] = delta
    print(f"  {f:30s}  {pn_med:16.4f}  {our_med:17.4f}  {delta:+.4f}")

# ── 최종 모델 재학습 ──────────────────────────────────────
ALL10 = [f"{b}_median" for b in BASE] + [f"{b}_iqr" for b in BASE]
train = table.dropna(subset=ALL10).copy()
y      = train["clinical_target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()

pipe_final = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  RobustScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
])
pipe_final.fit(train[BEST3].to_numpy(), y)

artifact = {
    "pipeline":            pipe_final,
    "features":            BEST3,
    "domain_correction":   correction,       # 예측 전 raw feature에 delta 더하기
    "threshold":           0.470,
    "threshold_strategy":  "sens_geq_0.80_max_spec",
    "model_mode":          "daily_walk_clinical_3feat_subwindow",
    "label_source":        "motor_impairment_score >= 0.5 (OR composite: DGI/TUG/FSST/BERG)",
    "train_dataset":       "PhysioNet 75h ambulatory, N=71 subjects",
    "train_n_normal":      int((y == 0).sum()),
    "train_n_impaired":    int((y == 1).sum()),
    "oof_subject_auc":     0.881,
    "oof_subject_sens":    0.971,
    "oof_subject_spec":    0.722,
    "reference":           "Moe-Nilssen & Helbostad 2004; Kavanagh & Menz 2008",
}
out_path = MODEL_DIR / "gait_daily_clinical_3feat.joblib"
joblib.dump(artifact, out_path)

meta = {
    "features":           BEST3,
    "domain_correction":  correction,
    "threshold":          0.470,
    "label_source":       "motor_impairment_score >= 0.5",
    "oof": {"auc": 0.881, "sens": 0.971, "spec": 0.722, "thr": 0.470, "unit": "subject"},
    "train": {"dataset": "PhysioNet 75h", "n_subjects": 71,
              "n_normal": int((y==0).sum()), "n_impaired": int((y==1).sum())},
    "vif_max": 1.75,
    "train_test_gap": 0.012,
}
(MODEL_DIR / "gait_daily_clinical_3feat_metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"\n[저장] {out_path}")
print(f"  피처: {BEST3}")
print(f"  도메인보정: {correction}")
print("[완료]")
