"""
75h 일상보행 데이터 → 우리 파이프라인 적용
- v2 CSV의 start_sec/end_sec 좌표로 20s 구간만 memmap 접근
- 각 20s → extract_best10_from_vmlap() → 최고 quality_score 윈도우 선택
- subject당 best 1개 (또는 bout당 1개) → HR 피처 → 모델 학습
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from gait_axis_aligned_core import (
    FEATURES, TARGET_FS_HZ,
    extract_best10_from_vmlap,
)

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR  = GAIT_PROJECT / "physionet_AWS"
V2_CSV   = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
OUT_DIR  = ROOT / "analysis_outputs" / "daily_walk_model"
MODEL_DIR = ROOT / "MOCA" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 100
WIN_SAMPLES = int(20 * FS)  # 2000 samples per 20s window

print(f"피처: {FEATURES}")


# ── hea 파일 파싱 ─────────────────────────────────────────
def parse_hea(subject_id: str) -> dict:
    hea_path = RAW_DIR / f"{subject_id}.hea"
    lines = hea_path.read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    n_samples = int(parts[3])
    n_channels = int(parts[1])
    gains, baselines = [], []
    for line in lines[1:1 + n_channels]:
        m = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(m.group(1)))
        baselines.append(float(m.group(2)))
    return {"n_samples": n_samples, "n_channels": n_channels,
            "gains": gains, "baselines": baselines}


# ── 20s 구간 읽기 (memmap으로 필요한 부분만) ──────────────
def read_20s_vmlap(subject_id: str, start_sec: float, hea: dict) -> np.ndarray | None:
    dat_path = RAW_DIR / f"{subject_id}.dat"
    if not dat_path.exists():
        return None
    start_sample = int(round(start_sec * FS))
    end_sample   = start_sample + WIN_SAMPLES
    if end_sample > hea["n_samples"]:
        return None
    try:
        raw = np.memmap(dat_path, dtype="<i2", mode="r",
                        shape=(hea["n_samples"], hea["n_channels"]))
        seg = raw[start_sample:end_sample, :3].astype(float)  # V, ML, AP only
        gains     = np.array(hea["gains"][:3])
        baselines = np.array(hea["baselines"][:3])
        return (seg - baselines) / gains  # 단위: g
    except Exception:
        return None


# ── subject별 best window 추출 ────────────────────────────
def extract_subject_best(subject_id: str, windows_df: pd.DataFrame) -> dict | None:
    try:
        hea = parse_hea(subject_id)
    except Exception:
        return None

    best_feat   = None
    best_quality = -np.inf

    for _, row in windows_df.iterrows():
        vmlap = read_20s_vmlap(subject_id, float(row["start_sec"]), hea)
        if vmlap is None or len(vmlap) < int(0.8 * WIN_SAMPLES):
            continue
        try:
            result = extract_best10_from_vmlap(vmlap, duration_sec=20.0)
        except Exception:
            continue
        q = result["all_features"].get("quality_score", -np.inf)
        if q > best_quality:
            best_quality = q
            best_feat    = result

    if best_feat is None:
        return None
    return {
        **{f: best_feat["features"].get(f, np.nan) for f in FEATURES},
        "quality_score": best_quality,
        "n_windows_tried": len(windows_df),
    }


# ── 메인: 피처 테이블 빌드 ───────────────────────────────
v2 = pd.read_csv(V2_CSV)
v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
v2["end_sec"]   = pd.to_numeric(v2["end_sec"],   errors="coerce")
v2 = v2.dropna(subset=["start_sec", "end_sec"])

subjects = v2[["subject_id", "group"]].drop_duplicates().sort_values("subject_id")
print(f"\n[1] 피처 추출 시작 — {len(subjects)}명")
print(f"    각 subject의 20s 윈도우에서 best 10s 선택")

rows = []
for _, s in subjects.iterrows():
    sid   = s["subject_id"]
    group = s["group"]
    wins  = v2[v2["subject_id"].eq(sid)]

    # 너무 많으면 랜덤 샘플 (subject당 최대 200개 윈도우만 시도)
    if len(wins) > 200:
        wins = wins.sample(200, random_state=42)

    feat = extract_subject_best(sid, wins)
    if feat is None:
        print(f"  [skip] {sid}")
        continue
    feat["subject_id"] = sid
    feat["group"]      = group
    feat["target"]     = 0 if group == "Control" else 1
    rows.append(feat)
    print(f"  {sid:8s} ({group:8s})  q={feat['quality_score']:.3f}  "
          f"HR_v={feat.get('v_harmonic_ratio', float('nan')):.3f}  "
          f"HR_ap={feat.get('ap_harmonic_ratio', float('nan')):.3f}  "
          f"freq={feat.get('v_stride_freq_hz', float('nan')):.3f}")

table = pd.DataFrame(rows)
table_path = OUT_DIR / "daily_walk_subject_table.csv"
table.to_csv(table_path, index=False)
n0 = int((table["target"] == 0).sum())
n1 = int((table["target"] == 1).sum())
print(f"\n저장: {table_path}  ({len(table)}명  정상={n0}  낙상군={n1})")


# ── 피처 분포 ─────────────────────────────────────────────
print("\n[2] 피처 분포 (group별 중앙값)")
for feat in FEATURES:
    if feat not in table.columns: continue
    g = table.groupby("target")[feat].median()
    print(f"  {feat:30s}  정상={g.get(0, float('nan')):.4f}  낙상군={g.get(1, float('nan')):.4f}")


# ── OOF + Youden 학습 ─────────────────────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.
        sp = tn/(tn+fp) if tn+fp else 0.
        if s + sp - 1 > best_j:
            best_j, best_t = s + sp - 1, t
    return float(best_t)


train = table.dropna(subset=FEATURES).copy()
y      = train["target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()

oof = np.zeros(len(train))
for tr, te in GroupKFold(n_splits=5).split(train[FEATURES], y, groups):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(train.iloc[tr][FEATURES].to_numpy(), y[tr])
    oof[te] = pipe.predict_proba(train.iloc[te][FEATURES].to_numpy())[:, 1]

thr  = youden_thr(y, oof)
pred = (oof >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
s  = tp/(tp+fn) if tp+fn else 0.
sp = tn/(tn+fp) if tn+fp else 0.

print(f"\n[3] OOF 결과 (Youden threshold={thr:.3f})")
print(f"  AUC={roc_auc_score(y, oof):.4f}  sens={s:.4f}  spec={sp:.4f}  "
      f"f1={f1_score(y, pred):.4f}")
print(f"  TP={tp} FN={fn} TN={tn} FP={fp}  (정상={n0} 낙상군={n1})")

# sens>=0.80 threshold
best_thr, best_spec = 0.5, -1.
for t in np.linspace(0.05, 0.95, 181):
    pred_t = (oof >= t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y, pred_t, labels=[0,1]).ravel()
    s2  = tp2/(tp2+fn2) if tp2+fn2 else 0.
    sp2 = tn2/(tn2+fp2) if tn2+fp2 else 0.
    if s2 >= 0.80 and sp2 > best_spec:
        best_thr, best_spec = float(t), float(sp2)
pred80 = (oof >= best_thr).astype(int)
tn80, fp80, fn80, tp80 = confusion_matrix(y, pred80, labels=[0,1]).ravel()
s80  = tp80/(tp80+fn80) if tp80+fn80 else 0.
sp80 = tn80/(tn80+fp80) if tn80+fp80 else 0.
print(f"\n  [sens>=0.80 최적] thr={best_thr:.3f}  sens={s80:.4f}  spec={sp80:.4f}  "
      f"TP={tp80} FN={fn80} TN={tn80} FP={fp80}")

# 최종 모델 저장
pipe_final = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  RobustScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
])
pipe_final.fit(train[FEATURES].to_numpy(), y)

artifact = {
    "pipeline":  pipe_final,
    "features":  FEATURES,
    "threshold": best_thr,
    "threshold_strategy": "sens_geq_0.80_max_spec",
    "model_mode": "daily_walk_75h_best10_harmonic_ratio",
    "train_n_normal":   n0,
    "train_n_impaired": n1,
    "reference": "Moe-Nilssen & Helbostad 2004; PhysioNet 75h ambulatory CO/FL",
}
model_path = MODEL_DIR / "gait_daily_walk_harmonic_ratio_youden.joblib"
joblib.dump(artifact, model_path)

meta = {"features": FEATURES, "threshold": best_thr,
        "oof": {"auc": round(float(roc_auc_score(y, oof)), 4),
                "sens": round(s80, 4), "spec": round(sp80, 4),
                "thr": best_thr},
        "label": "Control(0) vs Faller(1)"}
(MODEL_DIR / "gait_daily_walk_harmonic_ratio_metadata.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n[저장] {model_path}")
