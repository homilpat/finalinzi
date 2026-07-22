"""
보행 측정 시간(30s / 60s / 전체) 별 피처 안정성 + 최적 피처 조합 탐색

[데이터누수 방지 설계]
- 모든 CV: GroupKFold(subject 단위) — 같은 subject가 train/test에 동시 출현 불가
- 스케일러 / Imputer: Pipeline 내부에서만 fit (test set 정보 유입 없음)
- 피처 선택: Nested CV — 외부 fold 테스트셋은 feature selection에 절대 관여 안 함
- subject별 feature vector 생성 후 CV 실시 (window-level 절대 사용 안 함)
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── 경로 ─────────────────────────────────────────────────────────
IN_CSV      = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(
    Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"),
    None
)
OUT_DIR = ROOT / "analysis_outputs" / "bout_duration_feature_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 피처 정의 (논문 기반) ─────────────────────────────────────────
ALL_FEATS = [
    "v_harmonic_ratio_median",   # 수직 HR — 보행 리듬 (Moe-Nilssen 2004)
    "v_harmonic_ratio_iqr",      # 수직 HR 변동성
    "ap_harmonic_ratio_median",  # 전후 HR (Kavanagh 2006)
    "ap_harmonic_ratio_iqr",
    "v_stride_freq_hz_median",   # 보행 주파수 / 속도 proxy (Hausdorff 2007)
    "v_stride_freq_hz_iqr",      # 보행 주파수 변동성 (GV 지표)
    "ap_spec_entropy_median",    # AP 스펙트럼 엔트로피 (리듬 불규칙성)
    "ap_spec_entropy_iqr",
    "v_jerk_rms_median",         # 수직 jerk (Menz 2003)
    "v_jerk_rms_iqr",
]

CURRENT_3FEAT = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

# 30s ≈ 세그먼트 2개, 60s ≈ 3개, full = 전체
BOUT_CONFIG = {"30s": 2, "60s": 3, "full": None}
N_REPEAT    = 300   # 반복 시뮬레이션 횟수
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────
def make_subject_features(df: pd.DataFrame, k: int | None, rng: np.random.Generator) -> pd.DataFrame:
    """
    subject당 k개 연속 20s 세그먼트 샘플링 → 피처 median 집계
    → 1 subject = 1 row (누수 없음)
    """
    rows = []
    for sid, grp in df.groupby("subject_id", sort=False):
        vals = grp[ALL_FEATS].values.astype(float)
        n    = len(vals)
        if k is None or k >= n:
            sel = vals
        else:
            start = rng.integers(0, n - k + 1)   # 연속 k개
            sel   = vals[start : start + k]
        row = {f: float(np.nanmedian(sel[:, i])) for i, f in enumerate(ALL_FEATS)}
        row["subject_id"] = sid
        row["target"]     = int(grp["target"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def make_pipe() -> Pipeline:
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> float:
    """GroupKFold AUC — subject 단위 분리 보장"""
    cv   = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


# ─────────────────────────────────────────────────────────────────
# 데이터 로드 + 임상 라벨 merge
# ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(IN_CSV)

if CLINICAL_CSV and CLINICAL_CSV.exists():
    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")[["subject_id", "motor_impairment_score"]]
    clin["clinical_target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
    df = df_raw.merge(clin, on="subject_id", how="inner")
    df["target"] = df["clinical_target"]   # 기존 target(낙상이력) 대신 임상 라벨 사용
    label_note = "임상 운동저하 (motor_impairment_score >= 0.5)"
else:
    df = df_raw
    label_note = "낙상 이력 (Faller/Control) — 임상 CSV 미발견"

subjects = sorted(df["subject_id"].unique())
n_subj   = len(subjects)
n0 = (df.groupby("subject_id")["target"].first() == 0).sum()
n1 = (df.groupby("subject_id")["target"].first() == 1).sum()
print(f"데이터: {len(df)}행, {n_subj}명")
print(f"라벨: {label_note}")
print(f"정상 {n0}명 / 저하 {n1}명")
print()

# ─────────────────────────────────────────────────────────────────
# 1. Cohen's d 효과크기 (full 데이터 subject-level median)
# ─────────────────────────────────────────────────────────────────
rng_full = np.random.default_rng(RANDOM_SEED)
full_df  = make_subject_features(df, None, rng_full)
ctrl = full_df[full_df["target"] == 0]
impr = full_df[full_df["target"] == 1]

print("=" * 65)
print("1. 피처별 Cohen's d 효과크기 (subject-level, 내림차순)")
print("=" * 65)
eff_rows = []
for f in ALL_FEATS:
    c, i = ctrl[f].dropna().values, impr[f].dropna().values
    pooled = np.sqrt((np.var(c, ddof=1) + np.var(i, ddof=1)) / 2)
    d = (np.mean(i) - np.mean(c)) / pooled if pooled > 0 else 0.0
    eff_rows.append({"피처": f, "정상_mean": round(np.mean(c),4),
                     "저하_mean": round(np.mean(i),4), "Cohen_d": round(d,3)})

eff_df = pd.DataFrame(eff_rows).sort_values("Cohen_d", key=abs, ascending=False)
print(eff_df.to_string(index=False))
eff_df.to_csv(OUT_DIR / "cohens_d.csv", index=False, encoding="utf-8-sig")
print()

# ─────────────────────────────────────────────────────────────────
# 2. VIF 다중공선성 검정
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("2. VIF 다중공선성 (subject-level, full)")
print("   VIF > 5: 중간 공선성, VIF > 10: 심각")
print("=" * 65)
X_vif = full_df[ALL_FEATS].dropna()
vif_rows = [{"피처": f, "VIF": round(variance_inflation_factor(X_vif.values, i), 2)}
            for i, f in enumerate(ALL_FEATS)]
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
print(vif_df.to_string(index=False))
vif_df.to_csv(OUT_DIR / "vif_all_features.csv", index=False, encoding="utf-8-sig")
print()

# ─────────────────────────────────────────────────────────────────
# 3. 측정 시간별 AUC 안정성 (현재 3피처 모델)
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"3. 측정 시간별 AUC 안정성 — 현재 3피처 ({N_REPEAT}회 반복)")
print(f"   피처: {CURRENT_3FEAT}")
print("=" * 65)

bout_results = {}
for bout_name, k in BOUT_CONFIG.items():
    aucs = []
    for rep in range(N_REPEAT):
        rng  = np.random.default_rng(RANDOM_SEED + rep * 7)
        sim  = make_subject_features(df, k, rng)
        X    = sim[CURRENT_3FEAT].values
        y    = sim["target"].values
        grps = sim["subject_id"].values
        aucs.append(cv_auc(X, y, grps))
    arr = np.array([a for a in aucs if not np.isnan(a)])
    bout_results[bout_name] = arr
    print(f"  {bout_name:5s}: AUC {np.mean(arr):.3f} ± {np.std(arr):.3f}  "
          f"[{np.percentile(arr,5):.3f}~{np.percentile(arr,95):.3f}] 90%CI")

pd.DataFrame({k: v for k, v in bout_results.items()}).to_csv(
    OUT_DIR / "bout_duration_auc_distribution.csv", index=False, encoding="utf-8-sig")
print()

# ─────────────────────────────────────────────────────────────────
# 4. 피처 조합 탐색 (Nested CV, full 데이터)
#    외부 5-fold: 평가 / 내부: 후보 조합 ranking (train만 사용)
#    → 최종 AUC는 외부 fold test로만 계산 — 누수 없음
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("4. 최적 피처 조합 탐색 (Nested CV, full 데이터, 1~3피처)")
print("   [외부 5-fold 평가 / 내부 5-fold feature selection]")
print("=" * 65)

# 전체 후보 조합
CANDIDATES = []
for n in [1, 2, 3]:
    CANDIDATES.extend(itertools.combinations(ALL_FEATS, n))

sim_full = make_subject_features(df, None, np.random.default_rng(RANDOM_SEED))
y_full   = sim_full["target"].values
grp_full = sim_full["subject_id"].values

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=4)

# 각 외부 fold에서: inner CV로 best combo 선택 → outer test AUC 기록
outer_selected = []
outer_aucs     = []

for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(sim_full, y_full, grp_full)):
    X_tr = sim_full.iloc[tr_idx][ALL_FEATS].values
    X_te = sim_full.iloc[te_idx][ALL_FEATS].values
    y_tr, y_te = y_full[tr_idx], y_full[te_idx]
    g_tr = grp_full[tr_idx]

    # inner: train fold 안에서 best combo 선택
    best_inner_auc, best_combo = -1, CANDIDATES[0]
    for combo in CANDIDATES:
        combo = list(combo)
        feat_idx = [ALL_FEATS.index(f) for f in combo]
        inner_aucs = []
        for tr2, te2 in inner_cv.split(X_tr, y_tr, g_tr):
            if len(np.unique(y_tr[te2])) < 2:
                continue
            pipe = make_pipe()
            pipe.fit(X_tr[tr2][:, feat_idx], y_tr[tr2])
            inner_aucs.append(roc_auc_score(y_tr[te2], pipe.predict_proba(X_tr[te2][:, feat_idx])[:, 1]))
        if not inner_aucs:
            continue
        iauc = np.mean(inner_aucs)
        if iauc > best_inner_auc:
            best_inner_auc, best_combo = iauc, combo

    # outer: best combo로 test 평가
    feat_idx = [ALL_FEATS.index(f) for f in best_combo]
    pipe = make_pipe()
    pipe.fit(X_tr[:, feat_idx], y_tr)
    if len(np.unique(y_te)) >= 2:
        auc_te = roc_auc_score(y_te, pipe.predict_proba(X_te[:, feat_idx])[:, 1])
        outer_aucs.append(auc_te)
        outer_selected.append(best_combo)
        print(f"  Fold {fold_i+1}: best={best_combo}  inner={best_inner_auc:.3f}  outer={auc_te:.3f}")

print(f"\n  Nested CV AUC: {np.mean(outer_aucs):.3f} ± {np.std(outer_aucs):.3f}")
from collections import Counter
sel_counter = Counter([tuple(c) for c in outer_selected])
print("  Fold별 선택된 조합:", dict(sel_counter))
print()

# ─────────────────────────────────────────────────────────────────
# 5. 단순 GroupKFold combo 스크리닝 (참고용 — top 20)
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("5. 피처 조합 GroupKFold AUC 스크리닝 (full, 1~3피처, 참고용)")
print("   주의: 조합 선택에 selection bias 있음 → 4번 Nested CV가 정식 평가")
print("=" * 65)

screen_rows = []
for combo in CANDIDATES:
    combo = list(combo)
    feat_idx = [ALL_FEATS.index(f) for f in combo]
    X_c = sim_full.iloc[:, feat_idx].values if False else sim_full[combo].values
    auc = cv_auc(X_c, y_full, grp_full)
    screen_rows.append({"n_feat": len(combo), "피처조합": " + ".join(combo), "AUC": round(auc, 4)})

screen_df = pd.DataFrame(screen_rows).sort_values("AUC", ascending=False)
print("\n[Top 15]")
print(screen_df.head(15).to_string(index=False))
screen_df.to_csv(OUT_DIR / "combo_screening_auc.csv", index=False, encoding="utf-8-sig")

# 현재 모델 위치
cur_str = " + ".join(CURRENT_3FEAT)
cur_row = screen_df[screen_df["피처조합"] == cur_str]
if not cur_row.empty:
    rank = screen_df.reset_index(drop=True).index[screen_df["피처조합"] == cur_str].tolist()[0] + 1
    print(f"\n  현재 모델 [{cur_str}]")
    print(f"  AUC = {cur_row['AUC'].iloc[0]:.4f}, 전체 {len(screen_df)}개 중 {rank}위")

# ─────────────────────────────────────────────────────────────────
# 6. 30s / 60s 별 Top 조합 비교 (screen 상위 + 현재 모델)
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("6. 30s / 60s 측정 시 Top 조합 AUC (100회 반복 평균)")
print("=" * 65)

top_3feat = screen_df[screen_df["n_feat"] == 3].head(15)["피처조합"].tolist()
if cur_str not in top_3feat:
    top_3feat.append(cur_str)

for bout_name, k in [("30s", 2), ("60s", 3)]:
    print(f"\n  [{bout_name}]")
    brows = []
    for combo_str in top_3feat:
        combo = combo_str.split(" + ")
        aucs = []
        for rep in range(100):
            rng = np.random.default_rng(RANDOM_SEED + rep * 13)
            sim = make_subject_features(df, k, rng)
            X_c = sim[combo].values
            y_c = sim["target"].values
            g_c = sim["subject_id"].values
            aucs.append(cv_auc(X_c, y_c, g_c))
        arr = np.array([a for a in aucs if not np.isnan(a)])
        brows.append({"피처조합": combo_str, "AUC_mean": round(np.mean(arr),4),
                      "AUC_std": round(np.std(arr),4)})
    bd = pd.DataFrame(brows).sort_values("AUC_mean", ascending=False)
    print(bd.to_string(index=False))
    bd.to_csv(OUT_DIR / f"top_combos_{bout_name}.csv", index=False, encoding="utf-8-sig")

print()
print("완료:", OUT_DIR)
