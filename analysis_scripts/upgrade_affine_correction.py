"""
Affine 도메인 보정 업그레이드
기존: additive shift (delta = mean_PhysioNet - mean_OUR)
신규: affine calibration (scale + shift)
  a = std(PhysioNet_normals) / std(OUR_normals)   # 분포 폭 맞추기
  b = mean(PhysioNet_normals) - a * mean(OUR_normals)  # 중심 맞추기
  corrected = a * raw + b

참고: Sugiyama et al. 2007 Covariate Shift / moment matching
주의: OUR_SAMPLE 정상 3명으로 std 추정 → 신뢰구간 넓음 (df=2)
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

from gait_axis_aligned_core import extract_subwindow_daily_features

SAMPLE_DIR   = ROOT / "보행SAMPLE"
TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))
MODEL_DIR    = ROOT / "MOCA" / "models"
MODEL_PATH   = MODEL_DIR / "gait_daily_clinical_3feat.joblib"
META_PATH    = MODEL_DIR / "gait_daily_clinical_3feat_metadata.json"

BEST3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

# ── 1. PhysioNet 정상군 subject-level stats ────────────────────
print("[1] PhysioNet 정상군 affine 파라미터 계산")
table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id", "motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")

train_normals = table[table["clinical_target"] == 0]
# Subject 단위 median → 피처 분포의 대표값
subj_med = train_normals.groupby("subject_id")[BEST3].median()

pn_mean = subj_med.mean()
pn_std  = subj_med.std(ddof=1)   # Bessel correction
pn_n    = len(subj_med)

print(f"  PhysioNet 정상군 subject 수: {pn_n}")
print(f"\n  {'피처':35s}  {'mean':8s}  {'std':8s}")
for f in BEST3:
    print(f"  {f:35s}  {pn_mean[f]:8.4f}  {pn_std[f]:8.4f}")

# ── 2. OUR_SAMPLE 정상군 features ─────────────────────────────
print("\n[2] OUR_SAMPLE 정상군 피처 추출")
our_rows = []
for path in sorted(SAMPLE_DIR.glob("*.csv")):
    label = "impaired" if "발다침" in path.stem else "normal"
    if label != "normal":
        continue
    try:
        ex = extract_subwindow_daily_features(str(path))
        feat = ex["features"]
        feat["subject_id"] = path.stem
        our_rows.append(feat)
        print(f"  {path.name[:55]:55s} "
              f"jerk_med={feat.get('v_jerk_rms_median', float('nan')):.3f}  "
              f"jerk_iqr={feat.get('v_jerk_rms_iqr', float('nan')):.3f}  "
              f"HR_iqr={feat.get('v_harmonic_ratio_iqr', float('nan')):.3f}")
    except Exception as e:
        print(f"  [skip] {path.name}  err={e}")

our_df = pd.DataFrame(our_rows)
our_n = len(our_df)
our_mean = our_df[BEST3].mean()
our_std  = our_df[BEST3].std(ddof=1)

print(f"\n  OUR_SAMPLE 정상군 수: {our_n}")
print(f"\n  {'피처':35s}  {'mean':8s}  {'std':8s}")
for f in BEST3:
    print(f"  {f:35s}  {our_mean[f]:8.4f}  {our_std[f]:8.4f}")

# ── 3. Affine 파라미터 계산 ────────────────────────────────────
print("\n[3] Affine 보정 파라미터 (scale a, shift b)")
correction_affine = {}
correction_additive = {}   # 기존 방식 (비교용)

for f in BEST3:
    pm = float(pn_mean[f]); ps = float(pn_std[f])
    om = float(our_mean[f]); os_ = float(our_std[f])

    if our_n < 2 or not np.isfinite(os_) or os_ < 1e-8:
        # std 추정 불가 → additive fallback
        a = 1.0
        b = pm - om
        print(f"  {f:35s}  scale=1.000 (std 추정불가)  shift={b:+.4f}")
    else:
        a = ps / os_
        b = pm - a * om
        print(f"  {f:35s}  scale={a:.4f}  shift={b:+.4f}  "
              f"(PhysioNet σ={ps:.4f} / OUR σ={os_:.4f})")

    correction_affine[f] = {"scale": float(a), "shift": float(b)}
    correction_additive[f] = float(pm - om)   # 기존 delta

# ── 4. 모델 아티팩트 업데이트 ──────────────────────────────────
print("\n[4] 모델 아티팩트에 affine 보정 추가 (기존 additive 유지)")
artifact = joblib.load(MODEL_PATH)
artifact["domain_correction_affine"] = correction_affine
# 기존 domain_correction(additive)는 그대로 유지 (폴백)
joblib.dump(artifact, MODEL_PATH)
print(f"  저장: {MODEL_PATH}")

# metadata JSON 업데이트
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
meta["domain_correction_affine"] = correction_affine
meta["domain_correction_additive"] = correction_additive   # 기존 delta 명시
meta["domain_correction_n_our_normals"] = our_n
meta["domain_correction_n_pn_normals"] = pn_n
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  저장: {META_PATH}")

# ── 5. 검증: OUR_SAMPLE에 affine 적용 후 예측 ─────────────────
print("\n[5] OUR_SAMPLE 전체 affine 보정 후 예측 검증")
pipeline = artifact["pipeline"]
threshold = float(artifact.get("threshold", 0.5))
all_rows = []
for path in sorted(SAMPLE_DIR.glob("*.csv")):
    label = "impaired" if "발다침" in path.stem else "normal"
    target = 1 if label == "impaired" else 0
    try:
        ex = extract_subwindow_daily_features(str(path))
        raw = ex["features"]
    except Exception as e:
        print(f"  [skip] {path.name}  err={e}")
        continue
    # Affine 보정 적용
    corrected = {
        f: correction_affine[f]["scale"] * raw.get(f, float("nan")) + correction_affine[f]["shift"]
        for f in BEST3
    }
    X = np.array([[corrected.get(f, float("nan")) for f in BEST3]])
    prob = float(pipeline.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)
    correct = (pred == target)
    tag = "✓" if correct else "✗"
    print(f"  {tag} {path.stem[:45]:45s} [{label:8s}] prob={prob:.3f} pred={'저하' if pred else '정상'}")
    all_rows.append({"correct": correct, "target": target})

n_correct = sum(r["correct"] for r in all_rows)
print(f"\n  정답률: {n_correct}/{len(all_rows)}")

print("\n[완료]")
