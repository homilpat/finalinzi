"""
현재 3피처 고정 + 1~2개 추가 시 AUC 변화 분석

[방법]
- 기준 모델: v_jerk_rms_median + v_jerk_rms_iqr + v_harmonic_ratio_iqr (고정)
- 나머지 7개 피처 중 1개 추가 (7가지) → 4피처 조합
- 나머지 7개 피처 중 2개 추가 (21가지) → 5피처 조합
- 각 조합: full / 30s / 60s 측정 시간별 AUC 비교
- CV: GroupKFold(5-fold, subject 단위) — 누수 없음
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

IN_CSV       = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(
    Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"),
    None
)
OUT_DIR = ROOT / "analysis_outputs" / "bout_duration_feature_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 피처 정의 ─────────────────────────────────────────────────────
BASE_3 = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]   # 고정

ALL_FEATS = [
    "v_harmonic_ratio_median",
    "v_harmonic_ratio_iqr",
    "ap_harmonic_ratio_median",
    "ap_harmonic_ratio_iqr",
    "v_stride_freq_hz_median",
    "v_stride_freq_hz_iqr",
    "ap_spec_entropy_median",
    "ap_spec_entropy_iqr",
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
]

EXTRA_CANDIDATES = [f for f in ALL_FEATS if f not in BASE_3]   # 추가 가능 7개

BOUT_CONFIG = {"30s": 2, "60s": 3, "full": None}
N_REPEAT    = 200
RANDOM_SEED = 42


def make_subject_features(df: pd.DataFrame, k: int | None, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for sid, grp in df.groupby("subject_id", sort=False):
        vals = grp[ALL_FEATS].values.astype(float)
        n    = len(vals)
        if k is None or k >= n:
            sel = vals
        else:
            start = rng.integers(0, n - k + 1)
            sel   = vals[start : start + k]
        row = {f: float(np.nanmedian(sel[:, i])) for i, f in enumerate(ALL_FEATS)}
        row["subject_id"] = sid
        row["target"]     = int(grp["target"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def cv_auc(sim_df: pd.DataFrame, feats: list[str]) -> float:
    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])
    X    = sim_df[feats].values
    y    = sim_df["target"].values
    grps = sim_df["subject_id"].values
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, grps):
        if len(np.unique(y[te])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def sim_mean_auc(df: pd.DataFrame, feats: list[str], k: int | None, n_rep: int) -> tuple[float, float]:
    aucs = []
    for rep in range(n_rep):
        rng = np.random.default_rng(RANDOM_SEED + rep * 7)
        sim = make_subject_features(df, k, rng)
        aucs.append(cv_auc(sim, feats))
    arr = np.array([a for a in aucs if not np.isnan(a)])
    return float(np.mean(arr)), float(np.std(arr))


# ── 데이터 로드 ───────────────────────────────────────────────────
df_raw = pd.read_csv(IN_CSV)
if CLINICAL_CSV and CLINICAL_CSV.exists():
    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id", "motor_impairment_score"]]
    clin["clinical_target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
    df = df_raw.merge(clin, on="subject_id", how="inner")
    df["target"] = df["clinical_target"]
    print("라벨: 임상 운동저하 (motor_impairment_score >= 0.5)")
else:
    df = df_raw
    print("라벨: 낙상 이력 (임상 CSV 없음)")

n0 = (df.groupby("subject_id")["target"].first() == 0).sum()
n1 = (df.groupby("subject_id")["target"].first() == 1).sum()
print(f"71명 (정상 {n0} / 저하 {n1})")
print(f"\n기준 3피처: {BASE_3}")
print(f"추가 후보 {len(EXTRA_CANDIDATES)}개: {EXTRA_CANDIDATES}")
print()

# ─────────────────────────────────────────────────────────────────
# 기준 3피처 AUC (full / 30s / 60s)
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("기준 3피처 AUC")
print("=" * 70)
base_aucs = {}
for bout, k in BOUT_CONFIG.items():
    mu, sd = sim_mean_auc(df, BASE_3, k, N_REPEAT)
    base_aucs[bout] = (mu, sd)
    label = "(전체 데이터)" if k is None else f"({bout} 측정)"
    print(f"  {bout:5s} {label}: AUC {mu:.3f} ± {sd:.3f}")
print()

# ─────────────────────────────────────────────────────────────────
# +1 피처 추가 (7가지)
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("+1 피처 추가 (기준 3피처 + 1개 추가, 7가지 조합)")
print("=" * 70)

plus1_rows = []
for extra in EXTRA_CANDIDATES:
    feats = BASE_3 + [extra]
    row = {"추가피처": extra}
    for bout, k in BOUT_CONFIG.items():
        mu, sd = sim_mean_auc(df, feats, k, N_REPEAT)
        row[f"AUC_{bout}"] = round(mu, 4)
        row[f"std_{bout}"] = round(sd, 4)
        row[f"delta_{bout}"] = round(mu - base_aucs[bout][0], 4)
    plus1_rows.append(row)

plus1_df = pd.DataFrame(plus1_rows).sort_values("AUC_full", ascending=False)
print("\n[full 데이터 기준 정렬]")
display_cols = ["추가피처"] + [f"AUC_{b}" for b in BOUT_CONFIG] + [f"delta_{b}" for b in BOUT_CONFIG]
print(plus1_df[display_cols].to_string(index=False))
plus1_df.to_csv(OUT_DIR / "plus1_feature_auc.csv", index=False, encoding="utf-8-sig")

print("\n[delta = 기준 3피처 대비 AUC 변화 (양수 = 향상)]")
for bout in BOUT_CONFIG:
    best = plus1_df.sort_values(f"delta_{bout}", ascending=False).iloc[0]
    print(f"  {bout:5s}: 최고 추가 → '{best['추가피처']}'  delta={best[f'delta_{bout}']:+.4f}")

# ─────────────────────────────────────────────────────────────────
# +2 피처 추가 (21가지)
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("+2 피처 추가 (기준 3피처 + 2개 추가, 21가지 조합)")
print("=" * 70)

plus2_rows = []
for extra1, extra2 in itertools.combinations(EXTRA_CANDIDATES, 2):
    feats = BASE_3 + [extra1, extra2]
    row = {"추가피처": f"{extra1} + {extra2}"}
    for bout, k in BOUT_CONFIG.items():
        mu, sd = sim_mean_auc(df, feats, k, N_REPEAT)
        row[f"AUC_{bout}"] = round(mu, 4)
        row[f"std_{bout}"] = round(sd, 4)
        row[f"delta_{bout}"] = round(mu - base_aucs[bout][0], 4)
    plus2_rows.append(row)

plus2_df = pd.DataFrame(plus2_rows).sort_values("AUC_full", ascending=False)
print("\n[full 데이터 기준 Top 10]")
display_cols2 = ["추가피처"] + [f"AUC_{b}" for b in BOUT_CONFIG] + [f"delta_{b}" for b in BOUT_CONFIG]
print(plus2_df[display_cols2].head(10).to_string(index=False))
plus2_df.to_csv(OUT_DIR / "plus2_feature_auc.csv", index=False, encoding="utf-8-sig")

print("\n[delta = 기준 3피처 대비 AUC 변화]")
for bout in BOUT_CONFIG:
    best = plus2_df.sort_values(f"delta_{bout}", ascending=False).iloc[0]
    print(f"  {bout:5s}: 최고 추가 → '{best['추가피처']}'  delta={best[f'delta_{bout}']:+.4f}")

# ─────────────────────────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("요약: 추가 효과 있는 피처")
print("=" * 70)
print("\n[+1 피처] delta > 0 인 경우 (full 기준)")
pos = plus1_df[plus1_df["delta_full"] > 0][["추가피처", "AUC_full", "delta_full", "AUC_30s", "delta_30s"]]
if len(pos):
    print(pos.to_string(index=False))
else:
    print("  없음 — 추가해도 향상 없음")

print("\n[+2 피처] delta > 0 인 경우 (full 기준, Top 5)")
pos2 = plus2_df[plus2_df["delta_full"] > 0].head(5)[["추가피처", "AUC_full", "delta_full", "AUC_30s", "delta_30s"]]
if len(pos2):
    print(pos2.to_string(index=False))
else:
    print("  없음 — 추가해도 향상 없음")

print("\n완료:", OUT_DIR)
