from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "combined_centroid_similarity_feature_sets"
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
SAMPLE_PATH = (
    ROOT
    / "analysis_outputs"
    / "combined_fixed_shape6_sample_predictions"
    / "sample_best10_fixed_shape6_features.csv"
)

FEATURE_SETS = {
    "A_local_stable4": [
        "ml_spec_entropy",
        "ml_spec_peak_freq",
        "v_spec_peak_freq",
        "v_peak_timing_sd_pct",
    ],
    "B_direction_shape6": [
        "v_stride_regularity",
        "ap_stride_regularity",
        "v_stride_shape_cv_mean",
        "step_time_median",
        "stride_time_median",
        "ml_spec_entropy",
    ],
    "C_union9": [
        "ml_spec_entropy",
        "ml_spec_peak_freq",
        "v_spec_peak_freq",
        "v_peak_timing_sd_pct",
        "v_stride_regularity",
        "ap_stride_regularity",
        "v_stride_shape_cv_mean",
        "step_time_median",
        "stride_time_median",
    ],
}


def make_subject_table(features: list[str]) -> pd.DataFrame:
    phys = pd.read_csv(PHYS_PATH)
    phys = phys[phys["target"].notna()].copy()
    phys["target"] = phys["target"].astype(int)

    sp = pd.read_csv(SMARTPHONE_PATH)
    sp = sp[sp["target"].notna()].copy()
    sp["target"] = sp["target"].astype(int)

    common_cols = ["dataset", "label_group", "target", "source_id", "subject_id", *features]
    df = pd.concat([phys[common_cols], sp[common_cols]], ignore_index=True, sort=False)
    df["subject_id"] = df["subject_id"].fillna(df["source_id"]).astype(str)
    df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)

    rows = []
    for group_id, part in df.groupby("group_id", sort=True):
        targets = part["target"].dropna().astype(int).unique()
        if len(targets) != 1:
            continue
        row = {
            "group_id": group_id,
            "dataset": part["dataset"].iloc[0],
            "label_group": part["label_group"].iloc[0],
            "target": int(targets[0]),
            "n_windows": len(part),
        }
        for feature in features:
            row[feature] = pd.to_numeric(part[feature], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["target"]).reset_index(drop=True)


def robust_scale_params(train: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    med = train[features].apply(pd.to_numeric, errors="coerce").median()
    q1 = train[features].apply(pd.to_numeric, errors="coerce").quantile(0.25)
    q3 = train[features].apply(pd.to_numeric, errors="coerce").quantile(0.75)
    scale = (q3 - q1) / 1.349
    std = train[features].apply(pd.to_numeric, errors="coerce").std(ddof=0)
    scale = scale.mask(~np.isfinite(scale) | (scale <= 1e-12), std)
    scale = scale.mask(~np.isfinite(scale) | (scale <= 1e-12), 1.0)
    return med, scale


def fold_direction(train: pd.DataFrame, features: list[str]) -> pd.Series:
    normal_med = train.loc[train["target"].eq(0), features].median()
    impaired_med = train.loc[train["target"].eq(1), features].median()
    direction = np.sign(impaired_med - normal_med)
    direction = direction.replace(0, 1.0).fillna(1.0)
    return direction.astype(float)


def orient(df: pd.DataFrame, features: list[str], med: pd.Series, scale: pd.Series, direction: pd.Series) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    return ((x - med) / scale) * direction


def cosine_similarity_matrix(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1)
    c_norm = np.linalg.norm(centroid)
    denom = np.maximum(x_norm * c_norm, 1e-12)
    return (x @ centroid) / denom


def mahalanobis_distance(x: np.ndarray, center: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = x - center
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.0))


def build_similarity_features(
    train_oriented: pd.DataFrame,
    train_y: np.ndarray,
    df_oriented: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_oriented)
    x = imputer.transform(df_oriented)

    normal = x_train[train_y == 0]
    impaired = x_train[train_y == 1]
    normal_center = np.median(normal, axis=0)
    impaired_center = np.median(impaired, axis=0)

    out = pd.DataFrame(index=df_oriented.index)
    out[f"{prefix}cosine_to_normal"] = cosine_similarity_matrix(x, normal_center)
    out[f"{prefix}cosine_to_impaired"] = cosine_similarity_matrix(x, impaired_center)
    out[f"{prefix}cosine_margin_imp_minus_norm"] = (
        out[f"{prefix}cosine_to_impaired"] - out[f"{prefix}cosine_to_normal"]
    )
    out[f"{prefix}euclidean_to_normal"] = np.linalg.norm(x - normal_center, axis=1)
    out[f"{prefix}euclidean_to_impaired"] = np.linalg.norm(x - impaired_center, axis=1)
    out[f"{prefix}euclidean_margin_norm_minus_imp"] = (
        out[f"{prefix}euclidean_to_normal"] - out[f"{prefix}euclidean_to_impaired"]
    )

    if len(normal) > 2 and len(impaired) > 2:
        normal_lw = LedoitWolf().fit(normal)
        impaired_lw = LedoitWolf().fit(impaired)
        out[f"{prefix}mahalanobis_to_normal"] = mahalanobis_distance(
            x, normal_lw.location_, normal_lw.precision_
        )
        out[f"{prefix}mahalanobis_to_impaired"] = mahalanobis_distance(
            x, impaired_lw.location_, impaired_lw.precision_
        )
        out[f"{prefix}mahalanobis_margin_norm_minus_imp"] = (
            out[f"{prefix}mahalanobis_to_normal"] - out[f"{prefix}mahalanobis_to_impaired"]
        )
    else:
        out[f"{prefix}mahalanobis_to_normal"] = np.nan
        out[f"{prefix}mahalanobis_to_impaired"] = np.nan
        out[f"{prefix}mahalanobis_margin_norm_minus_imp"] = np.nan
    return out


def threshold_youden(y: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) == 0:
        return 0.5
    if len(vals) == 1:
        return float(vals[0])
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


def run_feature_set(name: str, features: list[str], table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=930000)
    metrics_rows = []
    pred_rows = []

    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        train_y = y[train_idx]
        test_y = y[test_idx]

        med, scale = robust_scale_params(train, features)
        direction = fold_direction(train, features)
        x_train_oriented = orient(train, features, med, scale, direction)
        x_test_oriented = orient(test, features, med, scale, direction)

        sim_train = build_similarity_features(x_train_oriented, train_y, x_train_oriented, "")
        sim_test = build_similarity_features(x_train_oriented, train_y, x_test_oriented, "")

        model_sets = {
            "centroid_margin_logistic": sim_train.columns.tolist(),
            "cosine_margin_only": ["cosine_margin_imp_minus_norm"],
            "euclidean_margin_only": ["euclidean_margin_norm_minus_imp"],
            "mahalanobis_margin_only": ["mahalanobis_margin_norm_minus_imp"],
        }
        for model_name, cols in model_sets.items():
            clf = logistic(940000 + repeat)
            clf.fit(sim_train[cols], train_y)
            train_prob = clf.predict_proba(sim_train[cols])[:, 1]
            threshold = threshold_youden(train_y, train_prob)
            test_prob = clf.predict_proba(sim_test[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(test_y, test_prob, test_pred)
            row.update(
                {
                    "feature_set": name,
                    "similarity_model": model_name,
                    "repeat": repeat,
                    "threshold": threshold,
                }
            )
            metrics_rows.append(row)

            pred = test[["group_id", "dataset", "label_group", "target"]].copy()
            pred["feature_set"] = name
            pred["similarity_model"] = model_name
            pred["repeat"] = repeat
            pred["probability_impaired"] = test_prob
            pred["prediction"] = test_pred
            pred_rows.append(pred)

    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def fit_full_and_predict_samples(feature_set: str, features: list[str], table: pd.DataFrame) -> pd.DataFrame:
    samples = pd.read_csv(SAMPLE_PATH)
    y = table["target"].astype(int).to_numpy()
    med, scale = robust_scale_params(table, features)
    direction = fold_direction(table, features)

    train_oriented = orient(table, features, med, scale, direction)
    sample_oriented = orient(samples, features, med, scale, direction)
    sim_train = build_similarity_features(train_oriented, y, train_oriented, "")
    sim_sample = build_similarity_features(train_oriented, y, sample_oriented, "")

    rows = []
    model_sets = {
        "centroid_margin_logistic": sim_train.columns.tolist(),
        "cosine_margin_only": ["cosine_margin_imp_minus_norm"],
        "euclidean_margin_only": ["euclidean_margin_norm_minus_imp"],
        "mahalanobis_margin_only": ["mahalanobis_margin_norm_minus_imp"],
    }
    for model_name, cols in model_sets.items():
        clf = logistic(950000 + len(feature_set) + len(model_name))
        clf.fit(sim_train[cols], y)
        train_prob = clf.predict_proba(sim_train[cols])[:, 1]
        threshold = threshold_youden(y, train_prob)
        sample_prob = clf.predict_proba(sim_sample[cols])[:, 1]
        for idx, sample in samples.iterrows():
            row = {
                "sample_file": sample["sample_file"],
                "feature_set": feature_set,
                "similarity_model": model_name,
                "probability_impaired": float(sample_prob[idx]),
                "threshold": threshold,
                "prediction": int(sample_prob[idx] >= threshold),
                "best10_start_sec": sample.get("start_sec", np.nan),
                "quality_score": sample.get("quality_score", np.nan),
            }
            for feature in features:
                row[feature] = sample.get(feature, np.nan)
            for col in sim_sample.columns:
                row[col] = sim_sample.loc[idx, col]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_features = list(dict.fromkeys(feature for features in FEATURE_SETS.values() for feature in features))
    table = make_subject_table(all_features)
    table.to_csv(OUT_DIR / "combined_subject_table_centroid_features.csv", index=False, encoding="utf-8-sig")

    metrics_all = []
    preds_all = []
    sample_all = []
    for feature_set, features in FEATURE_SETS.items():
        metrics, preds = run_feature_set(feature_set, features, table.dropna(subset=["target"]).copy())
        metrics_all.append(metrics)
        preds_all.append(preds)
        sample_all.append(fit_full_and_predict_samples(feature_set, features, table))

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    preds_df = pd.concat(preds_all, ignore_index=True)
    sample_df = pd.concat(sample_all, ignore_index=True)

    metrics_df.to_csv(OUT_DIR / "centroid_similarity_8020_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "centroid_similarity_8020_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(OUT_DIR / "centroid_similarity_sample_predictions.csv", index=False, encoding="utf-8-sig")

    summary = (
        metrics_df.groupby(["feature_set", "similarity_model"])
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
    summary.to_csv(OUT_DIR / "centroid_similarity_8020_metrics_summary.csv", index=False, encoding="utf-8-sig")

    dataset_summary = (
        preds_df.groupby(["feature_set", "similarity_model", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("group_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(
        OUT_DIR / "centroid_similarity_dataset_prediction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("subject table counts")
    print(pd.crosstab(table["dataset"], table["target"]).to_string())
    print("\nmetrics summary")
    print(summary.head(20).to_string(index=False))
    print("\nsample predictions")
    print(
        sample_df[
            [
                "sample_file",
                "feature_set",
                "similarity_model",
                "probability_impaired",
                "threshold",
                "prediction",
                "best10_start_sec",
                "quality_score",
            ]
        ].to_string(index=False)
    )
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
