"""
모든 window_features() 출력값 단독 AUC → best combination 탐색
임상 라벨 (motor_impairment_score >= 0.5) 기준
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from itertools import combinations

TABLE_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))

BASE = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz",
        "ap_spec_entropy", "v_jerk_rms"]
FEATS_ALL = [f"{f}_median" for f in BASE] + [f"{f}_iqr" for f in BASE]

table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id","motor_impairment_score"]]
clin["clinical_target"] = (pd.to_numeric(clin["motor_impairment_score"], errors="coerce") >= 0.5).astype(int)
table = table.merge(clin, on="subject_id", how="inner")
train = table.dropna(subset=FEATS_ALL).copy()
y      = train["clinical_target"].astype(int).to_numpy()
groups = train["subject_id"].to_numpy()

def oof_auc(feats):
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[feats], y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        pipe.fit(train.iloc[tr][feats].to_numpy(), y[tr])
        oof[te] = pipe.predict_proba(train.iloc[te][feats].to_numpy())[:, 1]
    return float(roc_auc_score(y, oof))

# ── 단독 AUC ─────────────────────────────────────────────
print("[1] 단독 AUC (각 피처 단독)")
solo = {}
for f in FEATS_ALL:
    a = oof_auc([f])
    solo[f] = a
    print(f"  {f:40s}  AUC={a:.4f}")

# ── 상위 5개 조합 탐색 ─────────────────────────────────────
top5 = sorted(solo, key=lambda k: -solo[k])[:6]
print(f"\n[2] 상위 6개 피처: {top5}")
print("\n[3] 상위 6개 중 2~4개 조합 AUC (상위 15개만)")

combo_results = []
for r in range(2, 5):
    for combo in combinations(top5, r):
        a = oof_auc(list(combo))
        combo_results.append((a, combo))

combo_results.sort(reverse=True)
for a, combo in combo_results[:20]:
    print(f"  AUC={a:.4f}  {combo}")

print(f"\n[현재 모델(MEDIAN+IQR 전체)]  AUC={oof_auc(FEATS_ALL):.4f}")
print("[완료]")
