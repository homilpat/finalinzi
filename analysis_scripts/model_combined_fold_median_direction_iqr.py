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
OUT_DIR = ROOT / "analysis_outputs" / "combined_fold_median_direction_iqr"
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


def fit_fold_params(train: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    normal = train[train["target"].eq(0)]
    impaired = train[train["target"].eq(1)]
    normal_median = normal[features].apply(pd.to_numeric, errors="coerce").median()
    impaired_median = impaired[features].apply(pd.to_numeric, errors="coerce").median()
    direction = np.sign(impaired_median - normal_median).replace(0, 1.0).fillna(1.0).astype(float)

    q1 = normal[features].apply(pd.to_numeric, errors="coerce").quantile(0.25)
    q3 = normal[features].apply(pd.to_numeric, errors="coerce").quantile(0.75)
    scale = (q3 - q1) / 1.349
    normal_std = normal[features].apply(pd.to_numeric, errors="coerce").std(ddof=0)
    scale = scale.mask(~np.isfinite(scale) | (scale <= 1e-12), normal_std)
    scale = scale.mask(~np.isfinite(scale) | (scale <= 1e-12), 1.0)
    return normal_median, scale, direction


def transform_risk_z(
    df: pd.DataFrame,
    features: list[str],
    normal_median: pd.Series,
    scale: pd.Series,
    direction: pd.Series,
) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    z = ((x - normal_median) / scale) * direction
    z.columns = [f"{col}__risk_z" for col in z.columns]
    z["risk_z_mean"] = z.mean(axis=1, skipna=True)
    z["risk_z_max"] = z[[c for c in z.columns if c.endswith("__risk_z")]].max(axis=1, skipna=True)
    z["risk_z_count_pos"] = (z[[c for c in z.columns if c.endswith("__risk_z")]] > 0).sum(axis=1)
    return z


def model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )


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


def run_feature_set(name: str, features: list[str], table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=970000)
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        normal_median, scale, direction = fit_fold_params(train, features)
        x_train = transform_risk_z(train, features, normal_median, scale, direction)
        x_test = transform_risk_z(test, features, normal_median, scale, direction)
        model_sets = {
            "risk_z_features": [f"{feature}__risk_z" for feature in features],
            "risk_z_summary3": ["risk_z_mean", "risk_z_max", "risk_z_count_pos"],
            "risk_z_features_plus_summary": [f"{feature}__risk_z" for feature in features]
            + ["risk_z_mean", "risk_z_max", "risk_z_count_pos"],
        }
        for model_name, cols in model_sets.items():
            clf = model(980000 + repeat)
            clf.fit(x_train[cols], y[train_idx])
            train_prob = clf.predict_proba(x_train[cols])[:, 1]
            threshold = threshold_youden(y[train_idx], train_prob)
            test_prob = clf.predict_proba(x_test[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(y[test_idx], test_prob, test_pred)
            row.update({"feature_set": name, "model_set": model_name, "repeat": repeat, "threshold": threshold})
            metrics_rows.append(row)
            pred = test[["group_id", "dataset", "label_group", "target"]].copy()
            pred["feature_set"] = name
            pred["model_set"] = model_name
            pred["repeat"] = repeat
            pred["probability_impaired"] = test_prob
            pred["prediction"] = test_pred
            pred_rows.append(pred)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def predict_samples(feature_set: str, features: list[str], table: pd.DataFrame) -> pd.DataFrame:
    samples = pd.read_csv(SAMPLE_PATH)
    y = table["target"].astype(int).to_numpy()
    normal_median, scale, direction = fit_fold_params(table, features)
    x_train = transform_risk_z(table, features, normal_median, scale, direction)
    x_sample = transform_risk_z(samples, features, normal_median, scale, direction)
    rows = []
    model_sets = {
        "risk_z_features": [f"{feature}__risk_z" for feature in features],
        "risk_z_summary3": ["risk_z_mean", "risk_z_max", "risk_z_count_pos"],
        "risk_z_features_plus_summary": [f"{feature}__risk_z" for feature in features]
        + ["risk_z_mean", "risk_z_max", "risk_z_count_pos"],
    }
    for model_name, cols in model_sets.items():
        clf = model(990000 + len(feature_set) + len(model_name))
        clf.fit(x_train[cols], y)
        train_prob = clf.predict_proba(x_train[cols])[:, 1]
        threshold = threshold_youden(y, train_prob)
        sample_prob = clf.predict_proba(x_sample[cols])[:, 1]
        for idx, sample in samples.iterrows():
            rows.append(
                {
                    "sample_file": sample["sample_file"],
                    "feature_set": feature_set,
                    "model_set": model_name,
                    "probability_impaired": float(sample_prob[idx]),
                    "threshold": threshold,
                    "prediction": int(sample_prob[idx] >= threshold),
                    "best10_start_sec": sample.get("start_sec", np.nan),
                    "quality_score": sample.get("quality_score", np.nan),
                    "risk_z_mean": x_sample.loc[idx, "risk_z_mean"],
                    "risk_z_max": x_sample.loc[idx, "risk_z_max"],
                    "risk_z_count_pos": x_sample.loc[idx, "risk_z_count_pos"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_features = list(dict.fromkeys(feature for features in FEATURE_SETS.values() for feature in features))
    table = make_subject_table(all_features)
    table.to_csv(OUT_DIR / "combined_subject_table.csv", index=False, encoding="utf-8-sig")
    metrics_all = []
    preds_all = []
    samples_all = []
    for feature_set, features in FEATURE_SETS.items():
        metrics, preds = run_feature_set(feature_set, features, table)
        metrics_all.append(metrics)
        preds_all.append(preds)
        samples_all.append(predict_samples(feature_set, features, table))
    metrics_df = pd.concat(metrics_all, ignore_index=True)
    preds_df = pd.concat(preds_all, ignore_index=True)
    samples_df = pd.concat(samples_all, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "fold_median_iqr_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "fold_median_iqr_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    samples_df.to_csv(OUT_DIR / "fold_median_iqr_sample_predictions.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics_df.groupby(["feature_set", "model_set"])
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
    summary.to_csv(OUT_DIR / "fold_median_iqr_metrics_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary = (
        preds_df.groupby(["feature_set", "model_set", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("group_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUT_DIR / "fold_median_iqr_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print("subject table counts")
    print(pd.crosstab(table["dataset"], table["target"]).to_string())
    print("\nmetrics summary")
    print(summary.to_string(index=False))
    print("\nsample predictions")
    print(samples_df.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
