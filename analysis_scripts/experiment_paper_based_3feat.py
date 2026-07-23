"""
실험: 논문 기반 3피처 모델
  - v_jerk_rms_mean   : LPF(20Hz) → diff → mean   (Kavanagh & Menz 2008)
  - v_jerk_rms_iqr    : LPF(20Hz) → diff → IQR    (Hausdorff 2001 변동성 개념)
  - v_harmonic_ratio_median : BPF(0.6-3Hz) → ACF → HR → median (Moe-Nilssen 2004)

비교 대상 (현행):
  - v_jerk_rms_median / v_jerk_rms_iqr : BPF(0.6-3Hz) → diff
  - v_harmonic_ratio_iqr               : BPF → ACF → IQR
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, sosfiltfilt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from gait_axis_aligned_core import TARGET_FS_HZ, bandpass, acf, peak_in_range

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR = GAIT_PROJECT / "physionet_AWS"
V2_CSV  = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
OUT_DIR = ROOT / "analysis_outputs" / "experiment_paper_based_3feat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS      = int(TARGET_FS_HZ)
WIN20   = int(20 * FS)
SUB_WIN = int(10 * FS)
SUB_STEP = int(2 * FS)
MAX_SEGS = 100

NEW_FEATS = ["v_jerk_rms_mean", "v_jerk_rms_iqr", "v_harmonic_ratio_median"]
OLD_FEATS = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]

N_SPLIT = 5
N_SEED  = 100


# ── 필터 ──────────────────────────────────────────────────
def lowpass20(x: np.ndarray) -> np.ndarray:
    sos = butter(4, 20.0 / (FS / 2.0), btype="low", output="sos")
    return sosfiltfilt(sos, x)


# ── hea 파싱 ──────────────────────────────────────────────
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


# ── 10s 서브윈도우 → 논문기반 피처 ────────────────────────
def paper_window_features(sub: np.ndarray) -> dict | None:
    if len(sub) < int(0.8 * SUB_WIN):
        return None

    v_raw = sub[:, 0]

    # 1) Jerk: LPF(20Hz) → diff (Kavanagh & Menz 2008)
    v_lp = lowpass20(v_raw)
    jerk = float(np.sqrt(np.mean(np.diff(v_lp) ** 2)) * FS)

    # 2) Harmonic Ratio: BPF(0.6-3Hz) → ACF (Moe-Nilssen & Helbostad 2004)
    v_bp = bandpass(v_raw, float(FS))
    c_v  = acf(v_bp)
    stride_lag, stride_peak, _ = peak_in_range(c_v, float(FS), 0.80, 1.70)
    if np.isfinite(stride_lag) and stride_lag > 0 and np.isfinite(stride_peak) and stride_peak > 1e-6:
        half = stride_lag / 2.0
        _, step_peak, _ = peak_in_range(c_v, float(FS), half * 0.6, half * 1.4)
        hr = float(step_peak / stride_peak) if np.isfinite(step_peak) else np.nan
    else:
        hr = np.nan

    return {"jerk": jerk, "hr": hr}


def subwindow_paper_feats(vmlap_20s: np.ndarray) -> dict | None:
    rows = []
    for s in range(0, WIN20 - SUB_WIN + 1, SUB_STEP):
        sub = vmlap_20s[s:s + SUB_WIN]
        feat = paper_window_features(sub)
        if feat is not None:
            rows.append(feat)
    if len(rows) < 2:
        return None
    jerks = np.array([r["jerk"] for r in rows if np.isfinite(r["jerk"])])
    hrs   = np.array([r["hr"]   for r in rows if np.isfinite(r["hr"])])
    if len(jerks) < 2 or len(hrs) < 2:
        return None
    return {
        "v_jerk_rms_mean":        float(np.mean(jerks)),
        "v_jerk_rms_iqr":         float(np.percentile(jerks, 75) - np.percentile(jerks, 25)),
        "v_harmonic_ratio_median": float(np.median(hrs)),
    }


# ── 피처 테이블 빌드 ──────────────────────────────────────
v2 = pd.read_csv(V2_CSV)
v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
v2 = v2.dropna(subset=["start_sec"])

subjects = v2[["subject_id", "group"]].drop_duplicates().sort_values("subject_id")
print(f"[1] {len(subjects)}명 피처 추출 (LPF-jerk + BPF-HR)")

all_rows = []
for _, s in subjects.iterrows():
    sid, group = s["subject_id"], s["group"]
    segs = v2[v2["subject_id"].eq(sid)]
    if len(segs) > MAX_SEGS:
        segs = segs.sample(MAX_SEGS, random_state=42)
    try:
        hea = parse_hea(sid)
    except Exception:
        print(f"  [skip] {sid} hea 오류"); continue

    seg_rows = []
    for _, row in segs.iterrows():
        vmlap = read_20s(sid, float(row["start_sec"]), hea)
        if vmlap is None: continue
        feat = subwindow_paper_feats(vmlap)
        if feat is None: continue
        feat.update({"subject_id": sid, "group": group,
                     "target": 0 if group == "Control" else 1})
        seg_rows.append(feat)

    all_rows.extend(seg_rows)
    if seg_rows:
        ex = seg_rows[0]
        print(f"  {sid:8s} ({group:8s}) {len(seg_rows):3d}개  "
              f"jerk_mean={ex['v_jerk_rms_mean']:.3f}  "
              f"jerk_iqr={ex['v_jerk_rms_iqr']:.3f}  "
              f"hr_med={ex['v_harmonic_ratio_median']:.3f}")
    else:
        print(f"  [skip] {sid} 유효 구간 없음")

table = pd.DataFrame(all_rows)
table.to_csv(OUT_DIR / "paper_feat_table.csv", index=False)
print(f"\n저장: {OUT_DIR / 'paper_feat_table.csv'}")


# ── OR 라벨 머지 + 집계 ──────────────────────────────────
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
for col in ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)"]:
    clin[col] = pd.to_numeric(clin[col], errors="coerce")
clin["OR_label"] = (
    (clin["TUG"] >= 12) | (clin["FSST"] >= 15) | (clin["BERG"] < 52) |
    (clin["DGI"] <= 19) | (clin["base(velocity)"] < 1.0) | (clin["s3(velocity)"] < 1.0)
).astype(int)

clin_lbl = clin[["subject_id", "OR_label"]].drop_duplicates("subject_id")
merged = table.merge(clin_lbl, on="subject_id", how="inner")

df = (
    merged.groupby("subject_id")[NEW_FEATS + ["OR_label"]]
    .agg({f: "median" for f in NEW_FEATS} | {"OR_label": "first"})
    .reset_index()
    .dropna(subset=NEW_FEATS)
    .reset_index(drop=True)
)

X = df[NEW_FEATS].to_numpy(float)
y = df["OR_label"].to_numpy(int)
groups = df["subject_id"].to_numpy()
n0, n1 = int((y == 0).sum()), int((y == 1).sum())
print(f"\n[2] subjects={len(y)} normal={n0} impaired={n1}")

print("\n[3] 피처 분포 (label별 median)")
for f in NEW_FEATS:
    v0 = float(df.loc[y==0, f].median())
    v1 = float(df.loc[y==1, f].median())
    print(f"  {f:30s}  정상={v0:.4f}  저하={v1:.4f}  diff={v1-v0:+.4f}")


# ── 100-rep CV ───────────────────────────────────────────
def make_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
    ])

print(f"\n[4] 100-rep {N_SPLIT}-fold StratifiedGroupKFold CV")
auc_list = []
for seed in range(N_SEED):
    sgkf = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2: continue
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        fold_aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    if fold_aucs:
        auc_list.append(float(np.mean(fold_aucs)))

new_auc_mean = float(np.mean(auc_list))
new_auc_std  = float(np.std(auc_list))
ci = [float(np.percentile(auc_list, 2.5)), float(np.percentile(auc_list, 97.5))]
print(f"  [논문기반] AUC={new_auc_mean:.4f} ± {new_auc_std:.4f}  95CI [{ci[0]:.3f}, {ci[1]:.3f}]")


# ── seed=42 sensitivity / specificity ───────────────────
sgkf42 = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=42)
oof_prob = np.zeros(len(y))
for tr, te in sgkf42.split(X, y, groups):
    pipe = make_pipe()
    pipe.fit(X[tr], y[tr])
    oof_prob[te] = pipe.predict_proba(X[te])[:, 1]

pred = (oof_prob >= 0.50).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
sens = tp/(tp+fn) if tp+fn else 0.
spec = tn/(tn+fp) if tn+fp else 0.
print(f"  seed42 thr=0.50  sens={sens:.4f}  spec={spec:.4f}")
print(f"  TP={tp} FN={fn} TN={tn} FP={fp}")


# ── 현행 모델과 비교 ─────────────────────────────────────
print("\n" + "="*55)
print("현행 모델  AUC=0.866 ± 0.009  (BPF jerk median/iqr + HR iqr)")
print(f"논문기반   AUC={new_auc_mean:.3f} ± {new_auc_std:.3f}  (LPF jerk mean/iqr + HR median)")
diff = new_auc_mean - 0.866
print(f"차이: {diff:+.3f}")
if abs(diff) < 0.01:
    print("→ 성능 동등: 논문기반 피처로 교체 권장 (방어력 ↑)")
elif diff > 0:
    print("→ 성능 향상: 논문기반 피처 교체 적극 권장")
else:
    print("→ 성능 하락: 현행 피처 유지하되 논문에서 정의 명시")

result = {
    "features": NEW_FEATS,
    "auc_mean": round(new_auc_mean, 4),
    "auc_std":  round(new_auc_std, 4),
    "auc_95ci": [round(ci[0], 4), round(ci[1], 4)],
    "seed42_sens": round(sens, 4),
    "seed42_spec": round(spec, 4),
    "vs_current_auc": 0.866,
    "auc_diff": round(diff, 4),
}
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n결과 저장: {OUT_DIR / 'result.json'}")
