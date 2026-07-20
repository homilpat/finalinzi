from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
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
OUT_DIR = ROOT / "analysis_outputs" / "fixed_shape6_directional_score_modeling"

FEATURES = [
    "v_stride_regularity",
    "ap_stride_regularity",
    "v_stride_shape_cv_mean",
    "step_time_median",
    "stride_time_median",
    "ml_spec_entropy",
]

# +1: higher means impaired. -1: lower means impaired.
DIRECTION = {
    "v_stride_regularity": -1.0,
    "ap_stride_regularity": -1.0,
    "v_stride_shape_cv_mean": 1.0,
    "step_time_median": 1.0,
    "stride_time_median": 1.0,
    "ml_spec_entropy": -1.0,
}


def fit_directional_reference(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    normal = train[train["target"].eq(0)]
    ref = {}
    for feature in FEATURES:
        values = pd.to_numeric(normal[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        scale = iqr / 1.349 if iqr > 1e-12 else float(values.std(ddof=0) or 1.0)
        if scale <= 1e-12 or not np.isfinite(scale):
            scale = 1.0
        ref[feature] = (median, scale)
    return ref


def transform_directional(df: pd.DataFrame, ref: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        median, scale = ref[feature]
        value = pd.to_numeric(df[feature], errors="coerce")
        out[f"{feature}__risk_z"] = DIRECTION[feature] * (value - median) / scale
    out["fixed_shape6_directional_mean"] = out.mean(axis=1, skipna=True)
    out["fixed_shape6_directional_max"] = out.max(axis=1, skipna=True)
    out["fixed_shape6_directional_count_pos"] = (out[[f"{f}__risk_z" for f in FEATURES]] > 0).sum(axis=1)
    return out


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


def repeated_cv(phys: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = phys["target"].astype(int).to_numpy()
    prob_sum = np.zeros(len(phys), dtype=float)
    count = np.zeros(len(phys), dtype=int)
    fold_rows = []
    for repeat in range(100):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=610000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(phys[FEATURES], y)):
            train = phys.iloc[train_idx].copy()
            test = phys.iloc[test_idx].copy()
            ref = fit_directional_reference(train)
            x_train = transform_directional(train, ref)
            x_test = transform_directional(test, ref)
            model = make_model(620000 + repeat * 10 + fold)
            model.fit(x_train[model_cols], y[train_idx])
            train_prob = model.predict_proba(x_train[model_cols])[:, 1]
            threshold = youden_threshold(y[train_idx], train_prob)
            test_prob = model.predict_proba(x_test[model_cols])[:, 1]
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
    oof = phys[["subject_id", "target", *FEATURES]].copy()
    oof["probability_impaired"] = oof_prob
    oof["prediction"] = oof_pred
    return pd.DataFrame(fold_rows), oof, oof_metrics


def external_predictions(phys: pd.DataFrame, smartphone: pd.DataFrame, model_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    y = phys["target"].astype(int).to_numpy()
    ref = fit_directional_reference(phys)
    x_train = transform_directional(phys, ref)
    x_ext = transform_directional(smartphone, ref)
    model = make_model(710000)
    model.fit(x_train[model_cols], y)
    train_prob = model.predict_proba(x_train[model_cols])[:, 1]
    threshold = youden_threshold(y, train_prob)
    ext = smartphone[["dataset", "label_group", "target", "source_id", "subject_id", *FEATURES]].copy()
    ext = pd.concat([ext.reset_index(drop=True), x_ext.reset_index(drop=True)], axis=1)
    ext["probability_impaired"] = model.predict_proba(ext[model_cols])[:, 1]
    ext["prediction"] = (ext["probability_impaired"] >= threshold).astype(int)
    summary = (
        ext.groupby(["dataset", "label_group", "target"], dropna=False)
        .agg(
            n=("source_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
            directional_mean_median=("fixed_shape6_directional_mean", "median"),
            directional_count_pos_median=("fixed_shape6_directional_count_pos", "median"),
        )
        .reset_index()
    )
    return ext, summary, threshold


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phys = pd.read_csv(PHYS_PATH)
    phys = phys[phys["target"].notna()].copy()
    phys["target"] = phys["target"].astype(int)
    smartphone = pd.read_csv(SMARTPHONE_PATH)
    smartphone = smartphone[smartphone["target"].notna()].copy()
    smartphone["target"] = smartphone["target"].astype(int)

    risk_cols = [f"{feature}__risk_z" for feature in FEATURES]
    model_sets = {
        "risk_z_all6": risk_cols,
        "risk_z_summary3": [
            "fixed_shape6_directional_mean",
            "fixed_shape6_directional_max",
            "fixed_shape6_directional_count_pos",
        ],
        "risk_z_all6_plus_summary": risk_cols
        + [
            "fixed_shape6_directional_mean",
            "fixed_shape6_directional_max",
            "fixed_shape6_directional_count_pos",
        ],
    }

    rows = []
    outputs = {}
    for name, cols in model_sets.items():
        folds, oof, oof_metrics = repeated_cv(phys, cols)
        ext, summary, threshold = external_predictions(phys, smartphone, cols)
        rows.append({"model_set": name, **oof_metrics, "external_threshold": threshold})
        outputs[name] = (folds, oof, ext, summary)
        folds.to_csv(OUT_DIR / f"{name}_fold_metrics_x100.csv", index=False, encoding="utf-8-sig")
        oof.to_csv(OUT_DIR / f"{name}_oof_predictions.csv", index=False, encoding="utf-8-sig")
        ext.to_csv(OUT_DIR / f"{name}_smartphone_external_predictions.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT_DIR / f"{name}_smartphone_external_summary.csv", index=False, encoding="utf-8-sig")

    result = pd.DataFrame(rows).sort_values(["auc", "sensitivity", "specificity"], ascending=[False, False, False])
    result.to_csv(OUT_DIR / "directional_score_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("features", FEATURES)
    print("direction", DIRECTION)
    print("phys label counts", phys["target"].value_counts().sort_index().to_dict())
    print("\nMODEL COMPARISON")
    print(result.to_string(index=False))
    best = result.iloc[0]["model_set"]
    print("\nBEST EXTERNAL", best)
    print(outputs[best][3].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
