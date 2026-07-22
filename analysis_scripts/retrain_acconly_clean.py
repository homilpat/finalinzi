"""
Retrain the final acc-only 3-feature gait model with an expanded clinical OR label.

Label:
    impaired = TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19
               OR base_velocity < 1.0 OR s3_velocity < 1.0

Features:
    v_jerk_rms_median, v_jerk_rms_iqr, v_harmonic_ratio_iqr
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


SUBWIN_CSV = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
MODEL_DST = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat.joblib"
META_DST = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_metadata.json"

FEATS = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
N_SPLIT = 5
N_SEED = 100

SERVICE_THRESHOLD = 0.50
LABEL_RULE = (
    "TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19 "
    "OR base_velocity < 1.0 OR s3_velocity < 1.0"
)
SIGNAL_CORRECTION = {
    "alpha": 1.9705093832241642,
    "tau": 1.0,
    "description": "sensor-level v_bp_rms ratio: OUR vs PhysioNet raw reference",
    "our_v_bp_rms_median": 0.09838236440400697,
    "pn_v_bp_rms_median": 0.19386337220187475,
    "n_our_normals": 6,
}


def make_pipe() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
        ]
    )


def clinical_expanded_target(clinical: pd.DataFrame) -> pd.Series:
    for col in ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)"]:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    return (
        (clinical["TUG"] >= 12)
        | (clinical["FSST"] >= 15)
        | (clinical["BERG"] < 52)
        | (clinical["DGI"] <= 19)
        | (clinical["base(velocity)"] < 1.0)
        | (clinical["s3(velocity)"] < 1.0)
    ).astype(int)


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    best_j, best_thr = -np.inf, 0.5
    best_sens, best_spec = 0.0, 0.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j_score = sens + spec - 1
        if j_score > best_j:
            best_j, best_thr = j_score, float(threshold)
            best_sens, best_spec = float(sens), float(spec)
    return best_thr, best_sens, best_spec


print("[1] Load data and build expanded clinical OR label")
sub = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["clinical_expanded_target"] = clinical_expanded_target(clin)

criterion_counts = {
    "TUG_ge_12": int((clin["TUG"] >= 12).sum()),
    "FSST_ge_15": int((clin["FSST"] >= 15).sum()),
    "BERG_lt_52": int((clin["BERG"] < 52).sum()),
    "DGI_le_19": int((clin["DGI"] <= 19).sum()),
    "base_velocity_lt_1p0": int((clin["base(velocity)"] < 1.0).sum()),
    "s3_velocity_lt_1p0": int((clin["s3(velocity)"] < 1.0).sum()),
}

clin_lbl = clin[["subject_id", "clinical_expanded_target"]].drop_duplicates("subject_id")
sub = sub.merge(clin_lbl, on="subject_id", how="inner", suffixes=("_old", ""))
sub = sub.drop(columns=[c for c in sub.columns if c.endswith("_old")])

df = (
    sub.groupby("subject_id")[FEATS + ["clinical_expanded_target"]]
    .agg({f: "median" for f in FEATS} | {"clinical_expanded_target": "first"})
    .reset_index()
)
df = df.dropna(subset=FEATS).reset_index(drop=True)

X = df[FEATS].to_numpy(dtype=float)
y = df["clinical_expanded_target"].to_numpy(dtype=int)
groups = df["subject_id"].to_numpy()
n0, n1 = int((y == 0).sum()), int((y == 1).sum())
print(f"  subjects={len(y)} normal={n0} impaired={n1}")
print(f"  rule={LABEL_RULE}")
print(f"  criteria positives={criterion_counts}")

print("\n[2] Feature distribution by label")
feature_summary = {}
for feat in FEATS:
    normal = float(df.loc[y == 0, feat].median())
    impaired = float(df.loc[y == 1, feat].median())
    feature_summary[feat] = {
        "normal_median": normal,
        "impaired_median": impaired,
        "difference": impaired - normal,
        "impaired_to_normal_ratio": impaired / normal if normal else None,
    }
    print(
        f"  {feat:24s} normal={normal:.6f} impaired={impaired:.6f} "
        f"diff={impaired - normal:+.6f} ratio={impaired / normal:.3f}x"
    )

print("\n[3] Multicollinearity")
scaled = StandardScaler().fit_transform(df[FEATS])
vif = {
    feat: float(variance_inflation_factor(scaled, idx))
    for idx, feat in enumerate(FEATS)
}
for feat, value in vif.items():
    print(f"  {feat:24s} VIF={value:.3f}")

print(f"\n[4] OOF AUC ({N_SPLIT}-fold StratifiedGroupKFold, {N_SEED} seeds)")
auc_list = []
for seed in range(N_SEED):
    sgkf = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        fold_aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    if fold_aucs:
        auc_list.append(float(np.mean(fold_aucs)))

oof_auc_mean = float(np.mean(auc_list))
oof_auc_std = float(np.std(auc_list))
oof_auc_ci = [float(np.percentile(auc_list, 2.5)), float(np.percentile(auc_list, 97.5))]
print(f"  AUC={oof_auc_mean:.4f} +/- {oof_auc_std:.4f}  95CI [{oof_auc_ci[0]:.3f}, {oof_auc_ci[1]:.3f}]")

print("\n[5] Subject-level OOF metrics (seed=42)")
sgkf42 = StratifiedGroupKFold(N_SPLIT, shuffle=True, random_state=42)
oof_prob = np.zeros(len(y))
for tr, te in sgkf42.split(X, y, groups):
    pipe = make_pipe()
    pipe.fit(X[tr], y[tr])
    oof_prob[te] = pipe.predict_proba(X[te])[:, 1]

youden_thr, youden_sens, youden_spec = youden_threshold(y, oof_prob)
pred_thr = (oof_prob >= SERVICE_THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_thr, labels=[0, 1]).ravel()
oof_subject_auc = float(roc_auc_score(y, oof_prob))
sens = tp / (tp + fn) if tp + fn else 0.0
spec = tn / (tn + fp) if tn + fp else 0.0
print(f"  OOF AUC={oof_subject_auc:.4f}")
print(f"  Youden threshold={youden_thr:.3f} sens={youden_sens:.3f} spec={youden_spec:.3f}")
print(f"  service threshold={SERVICE_THRESHOLD:.3f} sens={sens:.3f} spec={spec:.3f}")
print(f"  confusion_matrix tn={tn} fp={fp} fn={fn} tp={tp}")

print("\n[6] Fit final model and save")
final_pipe = make_pipe()
final_pipe.fit(X, y)

artifact = {
    "pipeline": final_pipe,
    "features": FEATS,
    "threshold": float(SERVICE_THRESHOLD),
    "threshold_strategy": "fixed_screening_threshold_sensitivity_prioritized",
    "model_mode": "daily_3feat_acconly",
    "label_source": LABEL_RULE,
    "clinical_label_criteria": {
        "TUG": ">= 12 seconds",
        "FSST": ">= 15 seconds",
        "BERG": "< 52",
        "DGI": "<= 19",
        "base_velocity": "< 1.0 m/s",
        "s3_velocity": "< 1.0 m/s",
    },
    "criterion_positive_counts": criterion_counts,
    "train_n_subjects": int(len(y)),
    "train_n_normal": n0,
    "train_n_impaired": n1,
    "feature_summary_by_label": feature_summary,
    "vif": vif,
    "oof_subject_auc": round(oof_auc_mean, 4),
    "oof_subject_auc_std": round(oof_auc_std, 4),
    "oof_subject_auc_95ci": [round(oof_auc_ci[0], 4), round(oof_auc_ci[1], 4)],
    "seed42_oof_auc": round(oof_subject_auc, 4),
    "seed42_youden_threshold": round(youden_thr, 4),
    "seed42_youden_sens": round(youden_sens, 4),
    "seed42_youden_spec": round(youden_spec, 4),
    "oof_subject_sens": round(sens, 4),
    "oof_subject_spec": round(spec, 4),
    "confusion_matrix_seed42": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "signal_correction": SIGNAL_CORRECTION,
}
joblib.dump(artifact, MODEL_DST)

meta = {k: v for k, v in artifact.items() if k != "pipeline"}
META_DST.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"  saved model={MODEL_DST}")
print(f"  saved metadata={META_DST}")
