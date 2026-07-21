"""
75h 일상보행 → 20s 구간 → 내부 10s 슬라이딩 → MEDIAN + IQR
Base 피처 5개: v_harmonic_ratio, ap_harmonic_ratio, v_stride_freq_hz,
               ap_spec_entropy (논문), v_jerk_rms (추가)
집계: MEDIAN×5 + IQR×5 = 10피처
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

from gait_axis_aligned_core import TARGET_FS_HZ, window_features

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR   = GAIT_PROJECT / "physionet_AWS"
V2_CSV    = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
OUT_DIR   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr"
MODEL_DIR = ROOT / "MOCA" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS          = int(TARGET_FS_HZ)
WIN20       = int(20 * FS)   # 20s = 2000 samples
SUB_WIN     = int(10 * FS)   # 10s sub-window
SUB_STEP    = int(2  * FS)   # 2s step → ~6 sub-windows per 20s
MAX_SEGS    = 100             # subject당 최대 20s 윈도우 수

BASE = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz",
        "ap_spec_entropy", "v_jerk_rms"]
FEATS = [f"{f}_median" for f in BASE] + [f"{f}_iqr" for f in BASE]

print(f"Base 피처 ({len(BASE)}개): {BASE}")
print(f"모델 피처 ({len(FEATS)}개): MEDIAN×5 + IQR×5")


# ── hea 파싱 ─────────────────────────────────────────────
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


# ── 20s 구간 읽기 ────────────────────────────────────────
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


# ── 20s 내부 슬라이딩 → 각 10s 피처 → MEDIAN + IQR ──────
def subwindow_median_iqr(vmlap_20s: np.ndarray) -> dict | None:
    starts = range(0, WIN20 - SUB_WIN + 1, SUB_STEP)
    rows = []
    for s in starts:
        sub = vmlap_20s[s:s + SUB_WIN]
        if len(sub) < int(0.8 * SUB_WIN): continue
        try:
            feat = window_features(sub)
            rows.append({k: feat[k] for k in BASE if k in feat})
        except Exception:
            continue
    if len(rows) < 2: return None
    arr = pd.DataFrame(rows)
    out = {}
    for f in BASE:
        vals = arr[f].dropna()
        if len(vals) < 2:
            out[f + "_median"] = np.nan
            out[f + "_iqr"]    = np.nan
        else:
            out[f + "_median"] = float(vals.median())
            out[f + "_iqr"]    = float(vals.quantile(0.75) - vals.quantile(0.25))
    return out


# ── 피처 테이블 빌드 ─────────────────────────────────────
v2 = pd.read_csv(V2_CSV)
v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
v2 = v2.dropna(subset=["start_sec"])

subjects = v2[["subject_id", "group"]].drop_duplicates().sort_values("subject_id")
print(f"\n[1] {len(subjects)}명 처리 중 (subject당 최대 {MAX_SEGS}개 20s 윈도우)")
print(f"    각 20s → 10s 슬라이딩(step=2s, ~{(WIN20-SUB_WIN)//SUB_STEP+1}개) → median/IQR")

all_rows = []
for _, s in subjects.iterrows():
    sid, group = s["subject_id"], s["group"]
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
        feat = subwindow_median_iqr(vmlap)
        if feat is None: continue
        feat["subject_id"] = sid
        feat["group"]      = group
        feat["target"]     = 0 if group == "Control" else 1
        seg_rows.append(feat)

    all_rows.extend(seg_rows)
    if seg_rows:
        ex = seg_rows[0]
        print(f"  {sid:8s} ({group:8s})  {len(seg_rows):3d}개 20s  "
              f"HR_v_med={ex.get('v_harmonic_ratio_median', float('nan')):.3f} "
              f"HR_v_iqr={ex.get('v_harmonic_ratio_iqr', float('nan')):.3f}  "
              f"jerk_med={ex.get('v_jerk_rms_median', float('nan')):.3f} "
              f"jerk_iqr={ex.get('v_jerk_rms_iqr', float('nan')):.3f}")
    else:
        print(f"  [skip] {sid} 유효 구간 없음")

table = pd.DataFrame(all_rows)
table.to_csv(OUT_DIR / "subwindow_median_iqr_table.csv", index=False)
n_subj = table["subject_id"].nunique()
n0 = int((table["target"] == 0).sum())
n1 = int((table["target"] == 1).sum())
print(f"\n저장 완료: {n_subj}명  {len(table)}개 20s 윈도우  정상={n0}  낙상군={n1}")


# ── 피처 분포 ─────────────────────────────────────────────
print("\n[2] 피처 분포 (target별 중앙값)")
subj_med = table.groupby(["subject_id","group","target"])[FEATS].median().reset_index()
print(f"  {'피처':38s}  {'정상':>8}  {'낙상군':>8}  {'차이':>8}")
for f in FEATS:
    g = subj_med.groupby("target")[f].median()
    v0, v1 = g.get(0, np.nan), g.get(1, np.nan)
    diff = v1 - v0 if np.isfinite(v0) and np.isfinite(v1) else np.nan
    flag = " *" if abs(diff) > 0.03 else ""
    print(f"  {f:38s}  {v0:8.4f}  {v1:8.4f}  {diff:+8.4f}{flag}")


# ── OOF GroupKFold (group=subject) ───────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0,1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.
        sp = tn/(tn+fp) if tn+fp else 0.
        if s + sp - 1 > best_j: best_j, best_t = s + sp - 1, t
    return float(best_t)


train = table.dropna(subset=FEATS).copy()
y      = train["target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()
oof    = np.zeros(len(train))

for tr, te in GroupKFold(n_splits=5).split(train[FEATS], y, groups):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(train.iloc[tr][FEATS].to_numpy(), y[tr])
    oof[te] = pipe.predict_proba(train.iloc[te][FEATS].to_numpy())[:, 1]

thr  = youden_thr(y, oof)
pred = (oof >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
s  = tp/(tp+fn) if tp+fn else 0.
sp = tn/(tn+fp) if tn+fp else 0.

print(f"\n[3] OOF (Youden thr={thr:.3f})")
print(f"  AUC={roc_auc_score(y, oof):.4f}  sens={s:.4f}  spec={sp:.4f}  f1={f1_score(y, pred):.4f}")
print(f"  TP={tp} FN={fn} TN={tn} FP={fp}  (정상={n0}행 낙상군={n1}행)")

# MEDIAN만 / IQR만 비교
for label, feats in [("MEDIAN만", [f for f in FEATS if "_median" in f]),
                     ("IQR만",    [f for f in FEATS if "_iqr" in f]),
                     ("MEDIAN+IQR", FEATS)]:
    oof2 = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[feats], y, groups):
        p2 = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", RobustScaler()),
                       ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))])
        p2.fit(train.iloc[tr][feats].to_numpy(), y[tr])
        oof2[te] = p2.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]
    t2 = youden_thr(y, oof2)
    p2pred = (oof2 >= t2).astype(int)
    tn2,fp2,fn2,tp2 = confusion_matrix(y, p2pred, labels=[0,1]).ravel()
    s2  = tp2/(tp2+fn2) if tp2+fn2 else 0.
    sp2 = tn2/(tn2+fp2) if tn2+fp2 else 0.
    print(f"  [{label:12s}]  AUC={roc_auc_score(y, oof2):.4f}  sens={s2:.4f}  spec={sp2:.4f}  thr={t2:.3f}")

print("\n[완료]")
