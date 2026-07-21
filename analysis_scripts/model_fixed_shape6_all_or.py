from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "fixed_shape6_all_or_modeling"
PHYS_PATH = (
    ROOT
    / "analysis_outputs"
    / "physionet_labwalks_smartphone_shape_extractor_all_or"
    / "physionet_labwalks_shape_best10_all_or.csv"
)
SMARTPHONE_PATH = (
    ROOT
    / "analysis_outputs"
    / "waveform_shape_feature_analysis"
    / "waveform_shape_features_same_preprocessing.csv"
)

FEATURES = [
    "v_stride_regularity",
    "ap_stride_regularity",
    "v_stride_shape_cv_mean",
    "step_time_median",
    "stride_time_median",
    "ml_spec_entropy",
]


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) == 0:
        return 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-9]
    best_t = 0.5
    best_j = -np.inf
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(threshold)
    return best_t


def metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def vif_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=features)
    rows = []
    for feature in features:
        y = x[feature].to_numpy()
        others = [c for c in features if c != feature]
        mat = np.column_stack([np.ones(len(x)), x[others].to_numpy()])
        beta = np.linalg.lstsq(mat, y, rcond=None)[0]
        pred = mat @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rows.append({"feature": feature, "vif": 1.0 / max(1e-9, 1.0 - r2), "r2_with_others": r2})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def spearman_pairs(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            pair = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = stats.spearmanr(pair[a], pair[b]).correlation if len(pair) >= 5 else np.nan
            rows.append({"feature_a": a, "feature_b": b, "spearman_rho": corr, "abs_spearman_rho": abs(corr)})
    return pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False)


def repeated_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = df["target"].astype(int).to_numpy()
    prob_sum = np.zeros(len(df), dtype=float)
    count = np.zeros(len(df), dtype=int)
    fold_rows = []
    for repeat in range(100):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=410000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df[FEATURES], y)):
            model = make_model(420000 + repeat * 10 + fold)
            model.fit(df.iloc[train_idx][FEATURES], y[train_idx])
            train_prob = model.predict_proba(df.iloc[train_idx][FEATURES])[:, 1]
            threshold = youden_threshold(y[train_idx], train_prob)
            test_prob = model.predict_proba(df.iloc[test_idx][FEATURES])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            m = metrics(y[test_idx], test_prob, test_pred)
            m.update({"repeat": repeat, "fold": fold, "threshold": threshold})
            fold_rows.append(m)
            prob_sum[test_idx] += test_prob
            count[test_idx] += 1
    oof_prob = prob_sum / count
    oof_threshold = youden_threshold(y, oof_prob)
    oof_pred = (oof_prob >= oof_threshold).astype(int)
    oof_metrics = {"threshold": oof_threshold, **metrics(y, oof_prob, oof_pred)}
    oof = df[["subject_id", "target", *FEATURES]].copy()
    oof["probability_impaired"] = oof_prob
    oof["prediction"] = oof_pred
    return pd.DataFrame(fold_rows), oof, oof_metrics


def full_model_external(phys: pd.DataFrame, smartphone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    y = phys["target"].astype(int).to_numpy()
    model = make_model(510000)
    model.fit(phys[FEATURES], y)
    train_prob = model.predict_proba(phys[FEATURES])[:, 1]
    threshold = youden_threshold(y, train_prob)
    external = smartphone.dropna(subset=["target"]).copy()
    external["target"] = external["target"].astype(int)
    external["probability_impaired"] = model.predict_proba(external[FEATURES])[:, 1]
    external["prediction"] = (external["probability_impaired"] >= threshold).astype(int)
    keep = ["dataset", "label_group", "target", "source_id", "subject_id", "probability_impaired", "prediction", *FEATURES]
    external = external[keep]
    summary = (
        external.groupby(["dataset", "label_group", "target"], dropna=False)
        .agg(
            n=("source_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
            prob_min=("probability_impaired", "min"),
            prob_max=("probability_impaired", "max"),
        )
        .reset_index()
    )
    return external, summary, threshold


def direction_table(phys: pd.DataFrame, smartphone: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        item = {"feature": feature}
        for name, df in [("phys", phys), ("smartphone", smartphone)]:
            normal = df[df["target"].eq(0)][feature].replace([np.inf, -np.inf], np.nan).dropna()
            impaired = df[df["target"].eq(1)][feature].replace([np.inf, -np.inf], np.nan).dropna()
            nmed = float(normal.median())
            imed = float(impaired.median())
            iqr = float(normal.quantile(0.75) - normal.quantile(0.25))
            z = (imed - nmed) / (iqr / 1.349) if iqr > 1e-12 else np.nan
            item[f"{name}_normal_median"] = nmed
            item[f"{name}_impaired_median"] = imed
            item[f"{name}_robust_z"] = z
        item["same_direction"] = np.sign(item["phys_robust_z"]) == np.sign(item["smartphone_robust_z"])
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phys = pd.read_csv(PHYS_PATH)
    phys = phys[phys["target"].notna()].copy()
    phys["target"] = phys["target"].astype(int)
    smartphone = pd.read_csv(SMARTPHONE_PATH)
    smartphone = smartphone[smartphone["target"].notna()].copy()
    smartphone["target"] = smartphone["target"].astype(int)

    missing_phys = [c for c in FEATURES if c not in phys.columns]
    missing_smart = [c for c in FEATURES if c not in smartphone.columns]
    if missing_phys or missing_smart:
        raise RuntimeError(f"Missing fixed features. phys={missing_phys}, smartphone={missing_smart}")

    folds, oof, oof_metrics = repeated_cv(phys)
    external, external_summary, external_threshold = full_model_external(phys, smartphone)
    vif = vif_values(phys, FEATURES)
    corr = spearman_pairs(phys, FEATURES)
    directions = direction_table(phys, smartphone)

    folds.to_csv(OUT_DIR / "fixed_shape6_fold_metrics_x100.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT_DIR / "fixed_shape6_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([oof_metrics]).to_csv(OUT_DIR / "fixed_shape6_oof_metrics.csv", index=False, encoding="utf-8-sig")
    external.to_csv(OUT_DIR / "fixed_shape6_smartphone_external_predictions.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(OUT_DIR / "fixed_shape6_smartphone_external_summary.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUT_DIR / "fixed_shape6_vif.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT_DIR / "fixed_shape6_spearman_pairs.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(OUT_DIR / "fixed_shape6_direction_check.csv", index=False, encoding="utf-8-sig")

    print("fixed features", FEATURES)
    print("phys label counts", phys["target"].value_counts().sort_index().to_dict())
    print("\nOOF")
    print(pd.DataFrame([oof_metrics]).to_string(index=False))
    print("\nfold mean")
    print(folds[["auc", "accuracy", "sensitivity", "specificity", "f1"]].mean().to_string())
    print("\nVIF")
    print(vif.to_string(index=False))
    print("\nSpearman max")
    print(corr.head(10).to_string(index=False))
    print("\nDirections")
    print(directions.to_string(index=False))
    print("\nExternal threshold", external_threshold)
    print(external_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
