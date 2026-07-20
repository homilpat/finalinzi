from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "physionet_waveform_template_similarity_all_or"
WAVEFORM_DIR = ROOT / "analysis_outputs" / "physionet_waveform_pca_t2_all_or"
X_PATH = WAVEFORM_DIR / "X_physionet_best10_vmlap_zscore_waveform.npy"
Y_PATH = WAVEFORM_DIR / "y_physionet_best10_all_or.npy"
META_PATH = WAVEFORM_DIR / "physionet_best10_waveform_meta.csv"
SAMPLE_X_PATH = WAVEFORM_DIR / "X_sample_best10_vmlap_zscore_waveform.npy"
SAMPLE_PRED_PATH = WAVEFORM_DIR / "waveform_pca_t2_sample_predictions.csv"


def class_templates(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "normal": np.nanmean(x_train[y_train == 0], axis=0),
        "impaired": np.nanmean(x_train[y_train == 1], axis=0),
        "normal_median": np.nanmedian(x_train[y_train == 0], axis=0),
        "impaired_median": np.nanmedian(x_train[y_train == 1], axis=0),
    }


def cosine(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(template), 1e-12)
    return (x @ template) / denom


def corr(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=1, keepdims=True)
    t0 = template - template.mean()
    denom = np.maximum(np.linalg.norm(x0, axis=1) * np.linalg.norm(t0), 1e-12)
    return (x0 @ t0) / denom


def euclidean(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - template, axis=1) / np.sqrt(x.shape[1])


def axis_scores(x: np.ndarray, template: np.ndarray, fn, prefix: str) -> pd.DataFrame:
    n = x.shape[1] // 3
    out = {}
    axes = {"v": (0, n), "ml": (n, 2 * n), "ap": (2 * n, 3 * n)}
    for axis, (lo, hi) in axes.items():
        out[f"{prefix}_{axis}"] = fn(x[:, lo:hi], template[lo:hi])
    return pd.DataFrame(out)


def transform_similarity(x: np.ndarray, templates: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ["normal", "impaired", "normal_median", "impaired_median"]:
        template = templates[name]
        part = pd.DataFrame(
            {
                f"{name}_cosine": cosine(x, template),
                f"{name}_corr": corr(x, template),
                f"{name}_euclidean": euclidean(x, template),
            }
        )
        part = pd.concat(
            [
                part,
                axis_scores(x, template, cosine, f"{name}_cosine_axis"),
                axis_scores(x, template, corr, f"{name}_corr_axis"),
                axis_scores(x, template, euclidean, f"{name}_euclidean_axis"),
            ],
            axis=1,
        )
        rows.append(part)
    out = pd.concat(rows, axis=1)
    out["cosine_margin_imp_minus_norm"] = out["impaired_cosine"] - out["normal_cosine"]
    out["corr_margin_imp_minus_norm"] = out["impaired_corr"] - out["normal_corr"]
    out["euclidean_margin_norm_minus_imp"] = out["normal_euclidean"] - out["impaired_euclidean"]
    out["median_cosine_margin_imp_minus_norm"] = out["impaired_median_cosine"] - out["normal_median_cosine"]
    out["median_corr_margin_imp_minus_norm"] = out["impaired_median_corr"] - out["normal_median_corr"]
    out["median_euclidean_margin_norm_minus_imp"] = (
        out["normal_median_euclidean"] - out["impaired_median_euclidean"]
    )
    for axis in ["v", "ml", "ap"]:
        out[f"{axis}_cosine_margin_imp_minus_norm"] = (
            out[f"impaired_cosine_axis_{axis}"] - out[f"normal_cosine_axis_{axis}"]
        )
        out[f"{axis}_corr_margin_imp_minus_norm"] = (
            out[f"impaired_corr_axis_{axis}"] - out[f"normal_corr_axis_{axis}"]
        )
        out[f"{axis}_euclidean_margin_norm_minus_imp"] = (
            out[f"normal_euclidean_axis_{axis}"] - out[f"impaired_euclidean_axis_{axis}"]
        )
    return out


def threshold_youden(y: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) <= 1:
        return float(vals[0]) if len(vals) else 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-9]
    best_t = 0.5
    best_j = -np.inf
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best_j:
            best_j = sens + spec - 1
            best_t = float(t)
    return best_t


def calc_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )


MODEL_SETS = {
    "whole_template_margins": [
        "cosine_margin_imp_minus_norm",
        "corr_margin_imp_minus_norm",
        "euclidean_margin_norm_minus_imp",
    ],
    "whole_median_template_margins": [
        "median_cosine_margin_imp_minus_norm",
        "median_corr_margin_imp_minus_norm",
        "median_euclidean_margin_norm_minus_imp",
    ],
    "axis_template_margins": [
        "v_cosine_margin_imp_minus_norm",
        "ml_cosine_margin_imp_minus_norm",
        "ap_cosine_margin_imp_minus_norm",
        "v_corr_margin_imp_minus_norm",
        "ml_corr_margin_imp_minus_norm",
        "ap_corr_margin_imp_minus_norm",
        "v_euclidean_margin_norm_minus_imp",
        "ml_euclidean_margin_norm_minus_imp",
        "ap_euclidean_margin_norm_minus_imp",
    ],
    "all_template_similarity": None,
}


def run_repeated(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1210000)
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        templates = class_templates(x[train_idx], y[train_idx])
        train_scores = transform_similarity(x[train_idx], templates)
        test_scores = transform_similarity(x[test_idx], templates)
        sets = dict(MODEL_SETS)
        sets["all_template_similarity"] = train_scores.columns.tolist()
        for model_name, cols in sets.items():
            clf = logistic(1220000 + repeat)
            clf.fit(train_scores[cols], y[train_idx])
            train_prob = clf.predict_proba(train_scores[cols])[:, 1]
            threshold = threshold_youden(y[train_idx], train_prob)
            test_prob = clf.predict_proba(test_scores[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(y[test_idx], test_prob, test_pred)
            row.update({"model_set": model_name, "repeat": repeat, "threshold": threshold})
            metrics_rows.append(row)
            pred = meta.iloc[test_idx][["group_id", "dataset", "source_id", "subject_id", "target"]].copy()
            pred["model_set"] = model_name
            pred["repeat"] = repeat
            pred["probability_impaired"] = test_prob
            pred["prediction"] = test_pred
            pred_rows.append(pred)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def predict_samples(x: np.ndarray, y: np.ndarray, sample_x: np.ndarray, sample_meta: pd.DataFrame) -> pd.DataFrame:
    templates = class_templates(x, y)
    train_scores = transform_similarity(x, templates)
    sample_scores = transform_similarity(sample_x, templates)
    sets = dict(MODEL_SETS)
    sets["all_template_similarity"] = train_scores.columns.tolist()
    rows = []
    for model_name, cols in sets.items():
        clf = logistic(1230000 + len(model_name))
        clf.fit(train_scores[cols], y)
        train_prob = clf.predict_proba(train_scores[cols])[:, 1]
        threshold = threshold_youden(y, train_prob)
        sample_prob = clf.predict_proba(sample_scores[cols])[:, 1]
        for idx, row in sample_meta.iterrows():
            out = row.to_dict()
            out["model_set"] = model_name
            out["probability_impaired"] = float(sample_prob[idx])
            out["threshold"] = threshold
            out["prediction"] = int(sample_prob[idx] >= threshold)
            for col in [
                "cosine_margin_imp_minus_norm",
                "corr_margin_imp_minus_norm",
                "euclidean_margin_norm_minus_imp",
                "v_corr_margin_imp_minus_norm",
                "ml_corr_margin_imp_minus_norm",
                "ap_corr_margin_imp_minus_norm",
            ]:
                out[col] = sample_scores.loc[idx, col]
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.load(X_PATH)
    y = np.load(Y_PATH).astype(int)
    meta = pd.read_csv(META_PATH)
    metrics, preds = run_repeated(x, y, meta)
    metrics.to_csv(OUT_DIR / "waveform_template_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "waveform_template_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics.groupby("model_set")
        .agg(
            n_repeats=("repeat", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
        )
        .reset_index()
        .sort_values(["auc_mean", "sensitivity_mean", "specificity_mean"], ascending=[False, False, False])
    )
    summary.to_csv(OUT_DIR / "waveform_template_metrics_summary.csv", index=False, encoding="utf-8-sig")

    sample_x = np.load(SAMPLE_X_PATH)
    sample_meta = pd.read_csv(SAMPLE_PRED_PATH).drop_duplicates("sample_file")[
        ["sample_file", "start_sec", "quality_score"]
    ].reset_index(drop=True)
    sample_preds = predict_samples(x, y, sample_x, sample_meta)
    sample_preds.to_csv(OUT_DIR / "waveform_template_sample_predictions.csv", index=False, encoding="utf-8-sig")
    print("metrics summary")
    print(summary.to_string(index=False))
    print("\nsample predictions")
    print(sample_preds.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
