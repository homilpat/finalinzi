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
OUT_DIR = ROOT / "analysis_outputs" / "combined_fixed_shape6_directional_8020"
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

DIRECTION = {
    "v_stride_regularity": -1.0,
    "ap_stride_regularity": -1.0,
    "v_stride_shape_cv_mean": 1.0,
    "step_time_median": 1.0,
    "stride_time_median": 1.0,
    "ml_spec_entropy": -1.0,
}


def make_subject_table() -> pd.DataFrame:
    phys = pd.read_csv(PHYS_PATH)
    phys = phys[phys["target"].notna()].copy()
    phys["target"] = phys["target"].astype(int)
    sp = pd.read_csv(SMARTPHONE_PATH)
    sp = sp[sp["target"].notna()].copy()
    sp["target"] = sp["target"].astype(int)
    common_cols = ["dataset", "label_group", "target", "source_id", "subject_id", *FEATURES]
    df = pd.concat([phys[common_cols], sp[common_cols]], ignore_index=True, sort=False)
    df["subject_id"] = df["subject_id"].fillna(df["source_id"]).astype(str)
    df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)

    rows = []
    for group_id, part in df.groupby("group_id", sort=True):
        target_values = part["target"].dropna().astype(int).unique()
        if len(target_values) != 1:
            continue
        row = {
            "group_id": group_id,
            "dataset": part["dataset"].iloc[0],
            "label_group": part["label_group"].iloc[0],
            "target": int(target_values[0]),
            "n_windows": len(part),
        }
        for feature in FEATURES:
            row[feature] = pd.to_numeric(part[feature], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["target"]).reset_index(drop=True)


def fit_ref(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    normal = train[train["target"].eq(0)]
    ref = {}
    for feature in FEATURES:
        values = pd.to_numeric(normal[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        scale = iqr / 1.349 if iqr > 1e-12 else float(values.std(ddof=0) or 1.0)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        ref[feature] = (median, scale)
    return ref


def transform(df: pd.DataFrame, ref: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        median, scale = ref[feature]
        out[f"{feature}__risk_z"] = DIRECTION[feature] * (
            pd.to_numeric(df[feature], errors="coerce") - median
        ) / scale
    risk_cols = [f"{feature}__risk_z" for feature in FEATURES]
    out["fixed_shape6_directional_mean"] = out[risk_cols].mean(axis=1, skipna=True)
    out["fixed_shape6_directional_max"] = out[risk_cols].max(axis=1, skipna=True)
    out["fixed_shape6_directional_count_pos"] = (out[risk_cols] > 0).sum(axis=1)
    return out


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


def run_8020(table: pd.DataFrame, model_cols: list[str], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=810000)
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(table, y)):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        ref = fit_ref(train)
        x_train = transform(train, ref)
        x_test = transform(test, ref)
        clf = model(820000 + repeat)
        clf.fit(x_train[model_cols], y[train_idx])
        train_prob = clf.predict_proba(x_train[model_cols])[:, 1]
        threshold = threshold_youden(y[train_idx], train_prob)
        test_prob = clf.predict_proba(x_test[model_cols])[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        m = calc_metrics(y[test_idx], test_prob, test_pred)
        m.update({"model_set": name, "repeat": repeat, "threshold": threshold})
        metrics_rows.append(m)
        pred = test[["group_id", "dataset", "label_group", "target"]].copy()
        pred["model_set"] = name
        pred["repeat"] = repeat
        pred["probability_impaired"] = test_prob
        pred["prediction"] = test_pred
        pred_rows.append(pred)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = make_subject_table()
    table.to_csv(OUT_DIR / "combined_subject_table_fixed_shape6.csv", index=False, encoding="utf-8-sig")
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
    all_metrics = []
    all_preds = []
    for name, cols in model_sets.items():
        metrics_df, preds = run_8020(table, cols, name)
        all_metrics.append(metrics_df)
        all_preds.append(preds)
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    preds_all = pd.concat(all_preds, ignore_index=True)
    metrics_all.to_csv(OUT_DIR / "combined_8020_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds_all.to_csv(OUT_DIR / "combined_8020_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics_all.groupby("model_set")
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
    summary.to_csv(OUT_DIR / "combined_8020_metrics_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary = (
        preds_all.groupby(["model_set", "dataset", "target"], dropna=False)
        .agg(
            n_predictions=("group_id", "count"),
            pred_impaired_rate=("prediction", "mean"),
            prob_median=("probability_impaired", "median"),
            prob_mean=("probability_impaired", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUT_DIR / "combined_8020_dataset_prediction_summary.csv", index=False, encoding="utf-8-sig")
    print("subject table counts")
    print(pd.crosstab(table["dataset"], table["target"]).to_string())
    print("\nmetrics summary")
    print(summary.to_string(index=False))
    print("\ndataset prediction summary")
    print(dataset_summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
