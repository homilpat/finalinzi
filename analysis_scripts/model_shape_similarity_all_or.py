from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
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
OUT_DIR = ROOT / "analysis_outputs" / "shape_similarity_all_or_modeling"

SIM_FEATURES = [
    "ml_spec_entropy",
    "ml_stride_regularity",
    "ml_stride_shape_cv_mean",
    "v_peak_timing_sd_pct",
    "step_time_median",
    "stride_time_median",
    "v_stride_shape_cv_mean",
    "ap_stride_regularity",
    "v_stride_regularity",
]


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


def normal_reference_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    normal = train[train["target"].eq(0)].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    x_test = scaler.transform(imputer.transform(test[features]))
    x_norm = scaler.transform(imputer.transform(normal[features]))

    template = np.mean(x_norm, axis=0)
    template_norm = np.linalg.norm(template) + 1e-12

    def cosine_to_template(x: np.ndarray) -> np.ndarray:
        denom = (np.linalg.norm(x, axis=1) * template_norm) + 1e-12
        return (x @ template) / denom

    lw = LedoitWolf().fit(x_norm)
    maha = lw.mahalanobis(x_test)

    out = pd.DataFrame(index=test.index)
    out["cosine_to_normal"] = cosine_to_template(x_test)
    out["cosine_distance"] = 1.0 - out["cosine_to_normal"]
    out["mahalanobis_to_normal"] = maha
    out["z_l2_to_normal"] = np.linalg.norm(x_test - template, axis=1)
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


def repeated_cv_similarity(df: pd.DataFrame, features: list[str], n_repeats: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["target"].astype(int).to_numpy()
    prob_sum = np.zeros(len(df), dtype=float)
    count = np.zeros(len(df), dtype=int)
    fold_rows = []
    score_rows = []

    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=310000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df[features], y)):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            train_scores = normal_reference_scores(train, train, features)
            test_scores = normal_reference_scores(train, test, features)
            score_cols = ["cosine_distance", "mahalanobis_to_normal", "z_l2_to_normal"]
            model = make_model(320000 + repeat * 10 + fold)
            model.fit(train_scores[score_cols], y[train_idx])
            train_prob = model.predict_proba(train_scores[score_cols])[:, 1]
            threshold = youden_threshold(y[train_idx], train_prob)
            test_prob = model.predict_proba(test_scores[score_cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            m = metrics(y[test_idx], test_prob, test_pred)
            m.update({"repeat": repeat, "fold": fold, "threshold": threshold})
            fold_rows.append(m)
            prob_sum[test_idx] += test_prob
            count[test_idx] += 1

            tmp = test[["subject_id", "target"]].copy()
            tmp["repeat"] = repeat
            tmp["fold"] = fold
            tmp["probability_impaired"] = test_prob
            tmp["prediction"] = test_pred
            score_rows.append(tmp)

    oof_prob = prob_sum / count
    oof_threshold = youden_threshold(y, oof_prob)
    oof_pred = (oof_prob >= oof_threshold).astype(int)
    oof = df[["subject_id", "target"]].copy()
    oof["probability_impaired"] = oof_prob
    oof["prediction"] = oof_pred
    oof.attrs["metrics"] = {"threshold": oof_threshold, **metrics(y, oof_prob, oof_pred)}
    return pd.DataFrame(fold_rows), oof


def external_similarity_predictions(phys: pd.DataFrame, smartphone: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, float]:
    score_cols = ["cosine_distance", "mahalanobis_to_normal", "z_l2_to_normal"]
    y = phys["target"].astype(int).to_numpy()
    train_scores = normal_reference_scores(phys, phys, features)
    model = make_model(880000)
    model.fit(train_scores[score_cols], y)
    train_prob = model.predict_proba(train_scores[score_cols])[:, 1]
    threshold = youden_threshold(y, train_prob)
    ext_scores = normal_reference_scores(phys, smartphone, features)
    out = smartphone[["dataset", "label_group", "target", "source_id", "subject_id"]].copy()
    out = pd.concat([out.reset_index(drop=True), ext_scores.reset_index(drop=True)], axis=1)
    out["probability_impaired"] = model.predict_proba(out[score_cols])[:, 1]
    out["prediction"] = (out["probability_impaired"] >= threshold).astype(int)
    return out, threshold


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phys = pd.read_csv(PHYS_PATH)
    phys = phys[phys["target"].notna()].copy()
    phys["target"] = phys["target"].astype(int)
    smartphone = pd.read_csv(SMARTPHONE_PATH)
    smartphone = smartphone[smartphone["target"].notna()].copy()
    smartphone["target"] = smartphone["target"].astype(int)
    features = [c for c in SIM_FEATURES if c in phys.columns and c in smartphone.columns]

    fold_metrics, oof = repeated_cv_similarity(phys, features, n_repeats=100)
    oof_metrics = {"threshold": oof.attrs["metrics"]["threshold"], **{k: v for k, v in oof.attrs["metrics"].items() if k != "threshold"}}
    fold_metrics.to_csv(OUT_DIR / "similarity_model_fold_metrics_x100.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT_DIR / "similarity_model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([oof_metrics]).to_csv(OUT_DIR / "similarity_model_oof_metrics.csv", index=False, encoding="utf-8-sig")

    external, threshold = external_similarity_predictions(phys, smartphone, features)
    external.to_csv(OUT_DIR / "similarity_model_smartphone_external_predictions.csv", index=False, encoding="utf-8-sig")
    summary = (
        external.groupby(["dataset", "label_group", "target"], dropna=False)
        .agg(
            n=("source_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
            cosine_distance_median=("cosine_distance", "median"),
            mahalanobis_median=("mahalanobis_to_normal", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "similarity_model_smartphone_external_summary.csv", index=False, encoding="utf-8-sig")

    print("features", features)
    print("phys label counts", phys["target"].value_counts().sort_index().to_dict())
    print("\nOOF")
    print(pd.DataFrame([oof_metrics]).to_string(index=False))
    print("\nfold mean")
    print(fold_metrics[["auc", "accuracy", "sensitivity", "specificity", "f1"]].mean().to_string())
    print("\nexternal threshold", threshold)
    print(summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
