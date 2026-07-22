"""
Leakage-safe validation for the final OR4 acc-only gait model.

Validation design:
  - subject-level feature table only
  - label: TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19
  - StratifiedGroupKFold, group=subject_id
  - threshold selected on each train fold only, then applied to test fold
  - 100 repeated 5-fold runs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


SUBWIN_CSV = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
OUT_DIR = ROOT / "analysis_outputs" / "or4_acc_only_validation_100rep"

FEATURES = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
N_SPLITS = 5
N_REPEATS = 100
LABEL_RULE = "TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=0)),
        ]
    )


def make_label(clinical: pd.DataFrame) -> pd.Series:
    for column in ["TUG", "FSST", "BERG", "DGI"]:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")
    return (
        (clinical["TUG"] >= 12)
        | (clinical["FSST"] >= 15)
        | (clinical["BERG"] < 52)
        | (clinical["DGI"] <= 19)
    ).astype(int)


def choose_train_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    best_j, best_threshold = -np.inf, 0.5
    best_sensitivity, best_specificity = 0.0, 0.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        youden = sensitivity + specificity - 1
        if youden > best_j:
            best_j = youden
            best_threshold = float(threshold)
            best_sensitivity = float(sensitivity)
            best_specificity = float(specificity)
    return best_threshold, best_sensitivity, best_specificity


def summarize(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().to_numpy(dtype=float)
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "ci95_low": float(np.percentile(clean, 2.5)),
        "ci95_high": float(np.percentile(clean, 97.5)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(SUBWIN_CSV)
    clinical = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    clinical["target"] = make_label(clinical)
    labels = clinical[["subject_id", "target"]].drop_duplicates("subject_id")

    table = sub.merge(labels, on="subject_id", how="inner", suffixes=("_old", ""))
    table = table.drop(columns=[column for column in table.columns if column.endswith("_old")])
    table = (
        table.groupby("subject_id")[FEATURES + ["target"]]
        .agg({**{feature: "median" for feature in FEATURES}, "target": "first"})
        .reset_index()
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )

    x = table[FEATURES].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=int)
    groups = table["subject_id"].to_numpy()

    criterion_counts = {
        "TUG_ge_12": int((clinical["TUG"] >= 12).sum()),
        "FSST_ge_15": int((clinical["FSST"] >= 15).sum()),
        "BERG_lt_52": int((clinical["BERG"] < 52).sum()),
        "DGI_le_19": int((clinical["DGI"] <= 19).sum()),
    }

    print(f"Label rule: {LABEL_RULE}")
    print(f"Subjects: {len(y)} normal={int((y == 0).sum())} impaired={int((y == 1).sum())}")
    print(f"Criterion positives: {criterion_counts}")
    print("\nFeature medians by label")
    for feature in FEATURES:
        normal = float(table.loc[y == 0, feature].median())
        impaired = float(table.loc[y == 1, feature].median())
        print(
            f"  {feature:24s} normal={normal:.6f} impaired={impaired:.6f} "
            f"diff={impaired - normal:+.6f} ratio={impaired / normal:.3f}x"
        )

    scaled = StandardScaler().fit_transform(table[FEATURES])
    print("\nVIF")
    for idx, feature in enumerate(FEATURES):
        print(f"  {feature:24s} {variance_inflation_factor(scaled, idx):.3f}")

    fold_rows = []
    repeat_rows = []
    for seed in range(N_REPEATS):
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
        repeat_true, repeat_prob, repeat_pred = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
            train_subjects = set(groups[train_idx])
            test_subjects = set(groups[test_idx])
            overlap = train_subjects.intersection(test_subjects)
            if overlap:
                raise RuntimeError(f"Subject leakage detected: {sorted(overlap)[:5]}")

            model = make_pipeline()
            model.fit(x[train_idx], y[train_idx])

            train_prob = model.predict_proba(x[train_idx])[:, 1]
            test_prob = model.predict_proba(x[test_idx])[:, 1]
            threshold, train_sens, train_spec = choose_train_youden_threshold(y[train_idx], train_prob)
            test_pred = (test_prob >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y[test_idx], test_pred, labels=[0, 1]).ravel()
            test_sens = tp / (tp + fn) if tp + fn else np.nan
            test_spec = tn / (tn + fp) if tn + fp else np.nan
            test_auc = (
                roc_auc_score(y[test_idx], test_prob)
                if len(np.unique(y[test_idx])) == 2
                else np.nan
            )

            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "threshold": threshold,
                    "train_sens_at_threshold": train_sens,
                    "train_spec_at_threshold": train_spec,
                    "test_auc": test_auc,
                    "test_sens": test_sens,
                    "test_spec": test_spec,
                    "test_acc": accuracy_score(y[test_idx], test_pred),
                    "test_f1": f1_score(y[test_idx], test_pred, zero_division=0),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "n_test": int(len(test_idx)),
                    "n_pos_test": int((y[test_idx] == 1).sum()),
                    "n_neg_test": int((y[test_idx] == 0).sum()),
                }
            )

            repeat_true.extend(y[test_idx].tolist())
            repeat_prob.extend(test_prob.tolist())
            repeat_pred.extend(test_pred.tolist())

        repeat_true = np.array(repeat_true, dtype=int)
        repeat_prob = np.array(repeat_prob, dtype=float)
        repeat_pred = np.array(repeat_pred, dtype=int)
        tn, fp, fn, tp = confusion_matrix(repeat_true, repeat_pred, labels=[0, 1]).ravel()
        repeat_rows.append(
            {
                "seed": seed,
                "oof_auc": roc_auc_score(repeat_true, repeat_prob),
                "oof_sens": tp / (tp + fn) if tp + fn else np.nan,
                "oof_spec": tn / (tn + fp) if tn + fp else np.nan,
                "oof_acc": accuracy_score(repeat_true, repeat_pred),
                "oof_f1": f1_score(repeat_true, repeat_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    repeat_metrics = pd.DataFrame(repeat_rows)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    repeat_metrics.to_csv(OUT_DIR / "repeat_oof_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nPer-repeat OOF metrics")
    for metric in ["oof_auc", "oof_sens", "oof_spec", "oof_acc", "oof_f1"]:
        stats = summarize(repeat_metrics[metric])
        print(
            f"  {metric:10s} mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"95CI=[{stats['ci95_low']:.4f},{stats['ci95_high']:.4f}]"
        )

    print("\nFold-level metrics")
    for metric in ["threshold", "test_auc", "test_sens", "test_spec", "test_acc", "test_f1"]:
        stats = summarize(fold_metrics[metric])
        print(
            f"  {metric:10s} mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"95CI=[{stats['ci95_low']:.4f},{stats['ci95_high']:.4f}]"
        )

    print(f"\nSaved: {OUT_DIR / 'fold_metrics.csv'}")
    print(f"Saved: {OUT_DIR / 'repeat_oof_metrics.csv'}")


if __name__ == "__main__":
    main()
