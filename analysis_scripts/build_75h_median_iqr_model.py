"""
75h 일상보행 → subject당 여러 10s 윈도우 → MEDIAN + IQR 집계 → 모델
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

from gait_axis_aligned_core import FEATURES as BASE_FEATURES, TARGET_FS_HZ, extract_best10_from_vmlap

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR  = GAIT_PROJECT / "physionet_AWS"
V2_CSV   = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
OUT_DIR  = ROOT / "analysis_outputs" / "daily_walk_median_iqr"
MODEL_DIR = ROOT / "MOCA" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 100
WIN_SAMPLES = int(20 * FS)
MAX_WINDOWS_PER_SUBJECT = 150  # 너무 많으면 시간 오래 걸려서 cap

MEDIAN_FEATURES = [f + "_median" for f in BASE_FEATURES]
IQR_FEATURES    = [f + "_iqr"    for f in BASE_FEATURES]
ALL_FEATURES    = MEDIAN_FEATURES + IQR_FEATURES

print(f"기반 피처: {BASE_FEATURES}")
print(f"집계 피처 ({len(ALL_FEATURES)}개): MEDIAN×4 + IQR×4")


def parse_hea(subject_id: str) -> dict:
    lines = (RAW_DIR / f"{subject_id}.hea").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    n_samples, n_ch = int(parts[3]), int(parts[1])
    gains, baselines = [], []
    for line in lines[1:1 + n_ch]:
        m = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(m.group(1))); baselines.append(float(m.group(2)))
    return {"n_samples": n_samples, "n_channels": n_ch,
            "gains": np.array(gains[:3]), "baselines": np.array(baselines[:3])}


def read_20s(subject_id: str, start_sec: float, hea: dict) -> np.ndarray | None:
    dat = RAW_DIR / f"{subject_id}.dat"
    if not dat.exists(): return None
    s = int(round(start_sec * FS))
    e = s + WIN_SAMPLES
    if e > hea["n_samples"]: return None
    try:
        raw = np.memmap(dat, dtype="<i2", mode="r",
                        shape=(hea["n_samples"], hea["n_channels"]))
        seg = raw[s:e, :3].astype(float)
        return (seg - hea["baselines"]) / hea["gains"]
    except Exception:
        return None


def extract_subject_windows(subject_id: str, windows_df: pd.DataFrame) -> list[dict]:
    try:
        hea = parse_hea(subject_id)
    except Exception:
        return []

    # 랜덤 샘플
    if len(windows_df) > MAX_WINDOWS_PER_SUBJECT:
        windows_df = windows_df.sample(MAX_WINDOWS_PER_SUBJECT, random_state=42)

    results = []
    for _, row in windows_df.iterrows():
        vmlap = read_20s(subject_id, float(row["start_sec"]), hea)
        if vmlap is None: continue
        try:
            r = extract_best10_from_vmlap(vmlap, duration_sec=20.0)
            results.append({f: r["features"].get(f, np.nan) for f in BASE_FEATURES})
        except Exception:
            continue
    return results


# ── 메인 ────────────────────────────────────────────────
v2 = pd.read_csv(V2_CSV)
v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
v2 = v2.dropna(subset=["start_sec"])

subjects = v2[["subject_id", "group"]].drop_duplicates().sort_values("subject_id")
print(f"\n[1] {len(subjects)}명 처리 중 (subject당 최대 {MAX_WINDOWS_PER_SUBJECT}개 윈도우)")

rows = []
for _, s in subjects.iterrows():
    sid, group = s["subject_id"], s["group"]
    wins = v2[v2["subject_id"].eq(sid)]
    feat_list = extract_subject_windows(sid, wins)

    if len(feat_list) < 5:
        print(f"  [skip] {sid} (유효 윈도우 {len(feat_list)}개)")
        continue

    arr = pd.DataFrame(feat_list)
    row = {"subject_id": sid, "group": group,
           "target": 0 if group == "Control" else 1,
           "n_windows": len(feat_list)}
    for f in BASE_FEATURES:
        vals = arr[f].dropna()
        row[f + "_median"] = float(vals.median()) if len(vals) else np.nan
        row[f + "_iqr"]    = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan

    rows.append(row)
    print(f"  {sid:8s} ({group:8s})  n={len(feat_list):3d}  "
          f"HR_v_med={row['v_harmonic_ratio_median']:.3f}  "
          f"HR_v_iqr={row['v_harmonic_ratio_iqr']:.3f}  "
          f"freq_med={row['v_stride_freq_hz_median']:.3f}  "
          f"freq_iqr={row['v_stride_freq_hz_iqr']:.3f}")

table = pd.DataFrame(rows)
table.to_csv(OUT_DIR / "daily_walk_median_iqr_table.csv", index=False)
n0 = int((table["target"] == 0).sum())
n1 = int((table["target"] == 1).sum())
print(f"\n저장 완료: {len(table)}명  정상={n0}  낙상군={n1}")


# ── 피처 분포 ─────────────────────────────────────────────
print("\n[2] 피처 분포")
print(f"  {'피처':40s}  {'정상 중앙값':>10}  {'낙상군 중앙값':>12}  {'차이':>8}")
for f in ALL_FEATURES:
    g = table.groupby("target")[f].median()
    v0, v1 = g.get(0, np.nan), g.get(1, np.nan)
    diff = v1 - v0 if np.isfinite(v0) and np.isfinite(v1) else np.nan
    print(f"  {f:40s}  {v0:10.4f}  {v1:12.4f}  {diff:8.4f}")


# ── OOF 모델 ─────────────────────────────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.
        sp = tn/(tn+fp) if tn+fp else 0.
        if s + sp - 1 > best_j:
            best_j, best_t = s + sp - 1, t
    return float(best_t)


train = table.dropna(subset=ALL_FEATURES).copy()
y      = train["target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()
oof    = np.zeros(len(train))

for tr, te in GroupKFold(n_splits=5).split(train[ALL_FEATURES], y, groups):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(train.iloc[tr][ALL_FEATURES].to_numpy(), y[tr])
    oof[te] = pipe.predict_proba(train.iloc[te][ALL_FEATURES].to_numpy())[:, 1]

thr  = youden_thr(y, oof)
pred = (oof >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
s  = tp/(tp+fn) if tp+fn else 0.
sp = tn/(tn+fp) if tn+fp else 0.

print(f"\n[3] OOF (Youden thr={thr:.3f})")
print(f"  AUC={roc_auc_score(y, oof):.4f}  sens={s:.4f}  spec={sp:.4f}  f1={f1_score(y, pred):.4f}")
print(f"  TP={tp} FN={fn} TN={tn} FP={fp}")

# MEDIAN만 / IQR만 / 둘다 비교
for label, feats in [("MEDIAN만", MEDIAN_FEATURES), ("IQR만", IQR_FEATURES), ("MEDIAN+IQR", ALL_FEATURES)]:
    oof2 = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[feats], y, groups):
        pipe2 = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", RobustScaler()),
                          ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))])
        pipe2.fit(train.iloc[tr][feats].to_numpy(), y[tr])
        oof2[te] = pipe2.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]
    t2 = youden_thr(y, oof2)
    p2 = (oof2 >= t2).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y, p2, labels=[0, 1]).ravel()
    s2  = tp2/(tp2+fn2) if tp2+fn2 else 0.
    sp2 = tn2/(tn2+fp2) if tn2+fp2 else 0.
    print(f"  [{label:12s}]  AUC={roc_auc_score(y, oof2):.4f}  sens={s2:.4f}  spec={sp2:.4f}  thr={t2:.3f}")

print("\n[완료]")
