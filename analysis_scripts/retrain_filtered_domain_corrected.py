"""
도메인 보정 가능한 데이터만 쓰는 빠른 재학습 실험
Chapman + FoG_STAR (정상샘플 없어서 보정불가) 제거
class_weight='balanced' + SMOTE 비교
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT    = Path(__file__).resolve().parents[1]
TABLE   = ROOT / "analysis_outputs" / "axis_aligned_gait_model" / "axis_aligned_best10_subject_table.csv"
OUT_DIR = ROOT / "analysis_outputs" / "retrain_filtered"
MODEL_DIR = ROOT / "MOCA" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

FEATURES = [
    "v_acf_stride_peak",
    "v_acf_stride_peak_width_sec",
    "ap_acf_stride_peak_width_sec",
    "ap_spec_entropy",
]
CORRECTABLE   = {"PhysioNet_LabWalks", "UCI_HAR", "GEOTEC_SP"}
PHYSIONET_ONLY = {"PhysioNet_LabWalks"}


# ── 도메인 보정 (physionet_normal 기준 median shift) ────────
def apply_correction(df: pd.DataFrame, domains: set[str]) -> pd.DataFrame:
    df = df.copy()
    ref = df[df["dataset"].eq("PhysioNet_LabWalks") & df["target"].eq(0)][FEATURES].median()
    for ds, part in df.groupby("dataset"):
        if ds not in domains or ds == "PhysioNet_LabWalks":
            continue
        normals = part[part["target"].eq(0)]
        if len(normals) < 2:
            continue
        delta = ref - normals[FEATURES].median()
        df.loc[df["dataset"].eq(ds), FEATURES] += delta
    return df


# ── OOF 평가 ─────────────────────────────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.0
        sp = tn/(tn+fp) if tn+fp else 0.0
        if s + sp - 1 > best_j:
            best_j, best_t = s + sp - 1, t
    return float(best_t)


def oof_metrics(train: pd.DataFrame, smote: bool = False) -> dict:
    y      = train["target"].astype(int).to_numpy()
    groups = (train["dataset"] + "::" + train["subject_id"]).to_numpy()
    oof    = np.zeros(len(train))

    for tr, te in GroupKFold(n_splits=5).split(train[FEATURES], y, groups):
        X_tr, y_tr = train.iloc[tr][FEATURES].to_numpy(), y[tr]
        X_te       = train.iloc[te][FEATURES].to_numpy()

        if smote:
            try:
                from imblearn.over_sampling import SMOTE
                k = min(4, int(y_tr.sum()) - 1)
                if k >= 1:
                    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
            except Exception:
                pass

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                       solver="liblinear", C=1.0)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[te] = pipe.predict_proba(X_te)[:, 1]

    thr  = youden_thr(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_normal":   int((y == 0).sum()),
        "n_impaired": int((y == 1).sum()),
        "auc":  round(float(roc_auc_score(y, oof)), 4),
        "sens": round(float(tp/(tp+fn)) if tp+fn else 0.0, 4),
        "spec": round(float(tn/(tn+fp)) if tn+fp else 0.0, 4),
        "f1":   round(float(f1_score(y, pred)), 4),
        "thr":  round(float(thr), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def final_train(train: pd.DataFrame, smote: bool = False) -> tuple[Pipeline, float]:
    y = train["target"].astype(int).to_numpy()
    X = train[FEATURES].to_numpy()
    if smote:
        try:
            from imblearn.over_sampling import SMOTE
            k = min(4, int(y.sum()) - 1)
            if k >= 1:
                X, y = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
        except Exception:
            pass
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                   solver="liblinear", C=1.0)),
    ])
    pipe.fit(X, y)
    prob = pipe.predict_proba(train[FEATURES].to_numpy())[:, 1]
    thr  = youden_thr(train["target"].astype(int).to_numpy(), prob)
    return pipe, float(thr)


def show_sample_preds(corrected: pd.DataFrame, pipe: Pipeline, thr: float, label: str):
    sample = corrected[corrected["dataset"].eq("OUR_SAMPLE")].copy()
    if sample.empty:
        return
    prob = pipe.predict_proba(sample[FEATURES])[:, 1]
    sample = sample.copy()
    sample["probability"] = prob
    sample["pred"]        = (prob >= thr).astype(int)
    print(f"  [{label}] 우리샘플 (thr={thr:.3f})")
    for _, r in sample.iterrows():
        ok = "O" if r["pred"] == int(r["target"]) else "X"
        print(f"    {ok} {str(r['subject_id'])[:45]:45s}  prob={r['probability']:.3f}  pred={int(r['pred'])}  label={int(r['target'])}")


# ── 메인 ────────────────────────────────────────────────────
raw = pd.read_csv(TABLE)
raw["target"] = pd.to_numeric(raw["target"], errors="coerce")
raw = raw.dropna(subset=["target"])

experiments = [
    ("A_현재모델(전체)",   raw,                                              {"PhysioNet_LabWalks","UCI_HAR","GEOTEC_SP","Chapman_PD_OFF_RAW","FoG_STAR_BACK_WALK"}),
    ("B_필터링_보정가능만", raw[raw["dataset"].isin(CORRECTABLE)].copy(),    CORRECTABLE),
    ("C_PhysioNet만",     raw[raw["dataset"].isin(PHYSIONET_ONLY)].copy(), PHYSIONET_ONLY),
]

summary_rows = []
saved_pipe, saved_thr, saved_corrected = None, None, None

print("=" * 65)
for name, data, domains in experiments:
    corrected = apply_correction(data, domains)
    train     = corrected[corrected["dataset"].ne("OUR_SAMPLE")].copy()

    print(f"\n{'='*65}")
    print(f"[{name}]")
    print(f"  정상={int((train['target']==0).sum())}  저하={int((train['target']==1).sum())}  "
          f"도메인={sorted(train['dataset'].unique())}")

    res_bal   = oof_metrics(train, smote=False)
    res_smote = oof_metrics(train, smote=True)

    print(f"  balanced  AUC={res_bal['auc']}  sens={res_bal['sens']}  spec={res_bal['spec']}  "
          f"thr={res_bal['thr']}  TP={res_bal['tp']} FN={res_bal['fn']} TN={res_bal['tn']} FP={res_bal['fp']}")
    print(f"  +SMOTE    AUC={res_smote['auc']}  sens={res_smote['sens']}  spec={res_smote['spec']}  "
          f"thr={res_smote['thr']}  TP={res_smote['tp']} FN={res_smote['fn']} TN={res_smote['tn']} FP={res_smote['fp']}")

    # 우리 샘플 예측
    for use_smote, label in [(False, "balanced"), (True, "+SMOTE")]:
        pipe, thr = final_train(train, smote=use_smote)
        show_sample_preds(corrected, pipe, thr, label)

    for suffix, res in [("balanced", res_bal), ("smote", res_smote)]:
        summary_rows.append({"실험": name, "방식": suffix, **res})

    # B 모델 저장용
    if name == "B_필터링_보정가능만":
        saved_pipe, saved_thr, saved_corrected = final_train(train, smote=False), None, corrected
        saved_pipe, saved_thr = final_train(train, smote=False)

print("\n" + "=" * 65)
print("요약")
cols = ["실험","방식","n_normal","n_impaired","auc","sens","spec","f1","thr"]
print(pd.DataFrame(summary_rows)[cols].to_string(index=False))

# ── B 모델 저장 ─────────────────────────────────────────────
if saved_pipe is not None:
    artifact = {
        "pipeline":  saved_pipe,
        "features":  FEATURES,
        "threshold": saved_thr,
        "threshold_strategy": "physionet_normal_correctable_only_youden",
        "model_mode": "axis_aligned_correctable_domains_only",
        "excluded_domains": ["Chapman_PD_OFF_RAW", "FoG_STAR_BACK_WALK"],
    }
    out_path = MODEL_DIR / "gait_filtered_domain_corrected_youden.joblib"
    joblib.dump(artifact, out_path)

    meta = {
        "features":  FEATURES,
        "threshold": saved_thr,
        "excluded":  ["Chapman_PD_OFF_RAW", "FoG_STAR_BACK_WALK"],
        "reason":    "no within-domain normal controls — domain correction impossible",
    }
    (MODEL_DIR / "gait_filtered_domain_corrected_youden_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[B 모델 저장] {out_path}")
