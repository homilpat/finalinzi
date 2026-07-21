from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
GAIT_CODE = Path.home() / "Desktop" / "파이널 보행 프로젝트" / "75h_processing_butterworth"
if str(GAIT_CODE) not in sys.path:
    sys.path.insert(0, str(GAIT_CODE))
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from extract_physionet_labwalks_shape_features_all_or import PARAMS  # noqa: E402
from run_strict_preprocessing_from_physionet import (  # noqa: E402
    DEFAULT_DATA_DIR,
    butterworth_bandpass,
    butterworth_lowpass,
    load_physical_record,
)


OUT_DIR = ROOT / "analysis_outputs" / "physionet_waveform_pca_t2_all_or"
BEST10_PATH = (
    ROOT
    / "analysis_outputs"
    / "physionet_labwalks_smartphone_shape_extractor_all_or"
    / "physionet_labwalks_shape_best10_all_or.csv"
)
SAMPLE_DIR = ROOT / "보행SAMPLE"
WINDOW_SEC = 10.0
FS_OUT = 100.0
N_POINTS = int(WINDOW_SEC * FS_OUT)


def zscore_axis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std = np.where(std <= 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def flatten_waveform(acc_vmlap: np.ndarray, fs: float) -> np.ndarray:
    v = butterworth_bandpass(acc_vmlap[:, 0], fs, low=0.6, high=3.0, order=4)
    ml = butterworth_bandpass(acc_vmlap[:, 1], fs, low=0.6, high=3.0, order=4)
    ap = butterworth_bandpass(acc_vmlap[:, 2], fs, low=0.6, high=3.0, order=4)
    arr = np.column_stack([v, ml, ap])
    arr = zscore_axis(arr)
    if len(arr) != N_POINTS:
        old_t = np.linspace(0.0, WINDOW_SEC, len(arr), endpoint=False)
        new_t = np.linspace(0.0, WINDOW_SEC, N_POINTS, endpoint=False)
        arr = np.column_stack([np.interp(new_t, old_t, arr[:, i]) for i in range(arr.shape[1])])
    return arr.reshape(-1).astype(np.float32)


def build_physionet_waveforms() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    best = pd.read_csv(BEST10_PATH)
    best = best[best["target"].notna()].copy()
    rows = []
    x_rows = []
    y_rows = []
    for _, row in best.iterrows():
        source_id = str(row["source_id"])
        data, fs, _ = load_physical_record(DEFAULT_DATA_DIR / source_id, channels=(0, 1, 2))
        data = butterworth_lowpass(data, fs, cutoff=20.0, order=4)
        start = int(round(float(row["start_sec"]) * fs))
        end = start + int(round(WINDOW_SEC * fs))
        segment = data[start:end]
        if segment.shape[0] != int(round(WINDOW_SEC * fs)):
            continue
        x_rows.append(flatten_waveform(segment, fs))
        y_rows.append(int(row["target"]))
        rows.append(
            {
                "dataset": "PhysioNet_LabWalks",
                "source_id": source_id,
                "subject_id": str(row["subject_id"]),
                "group_id": f"PhysioNet_LabWalks::{row['subject_id']}",
                "target": int(row["target"]),
                "start_sec": float(row["start_sec"]),
                "quality_score": float(row["quality_score"]),
            }
        )
    return np.vstack(x_rows), np.asarray(y_rows, dtype=int), pd.DataFrame(rows)


def read_sample_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def resample_sample_100hz(df: pd.DataFrame) -> pd.DataFrame:
    t_raw = pd.to_numeric(df["Timestamp_ns"], errors="coerce")
    t = (t_raw - t_raw.iloc[0]) / 1e9
    valid = t.notna()
    df = df.loc[valid].copy()
    t = t.loc[valid].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    df = df.iloc[order].reset_index(drop=True)
    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]
    df = df.iloc[unique_idx].reset_index(drop=True)
    grid = np.arange(0.0, float(t[-1]) + 1e-9, 1.0 / FS_OUT)
    out = pd.DataFrame({"time_sec": grid})
    for col in df.columns:
        if col == "Timestamp_ns":
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(values) & np.isfinite(t)
        if mask.sum() < 2:
            continue
        f = interp1d(t[mask], values[mask], kind="linear", bounds_error=False, fill_value="extrapolate")
        out[col] = f(grid)
    return out


def sample_acc_vmlap(rs: pd.DataFrame) -> np.ndarray:
    if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(rs.columns):
        return np.column_stack(
            [
                rs["Acc_Vertical_g"].to_numpy(float),
                rs["Acc_ML_g"].to_numpy(float),
                rs["Acc_AP_g"].to_numpy(float),
            ]
        )
    return np.column_stack(
        [
            rs["Acc_Z"].to_numpy(float),
            rs["Acc_X"].to_numpy(float),
            rs["Acc_Y"].to_numpy(float),
        ]
    )


def build_sample_waveforms() -> tuple[np.ndarray, pd.DataFrame]:
    # Use the same best10 start positions already selected by the fixed extractor.
    sample_best = pd.read_csv(
        ROOT
        / "analysis_outputs"
        / "combined_fixed_shape6_sample_predictions"
        / "sample_best10_fixed_shape6_features.csv"
    )
    rows = []
    x_rows = []
    for _, row in sample_best.iterrows():
        path = SAMPLE_DIR / str(row["sample_file"])
        if not path.exists():
            continue
        rs = resample_sample_100hz(read_sample_csv(path))
        acc = sample_acc_vmlap(rs)
        start = int(round(float(row["start_sec"]) * FS_OUT))
        end = start + N_POINTS
        segment = acc[start:end]
        if segment.shape[0] != N_POINTS:
            continue
        x_rows.append(flatten_waveform(segment, FS_OUT))
        rows.append(
            {
                "sample_file": row["sample_file"],
                "start_sec": float(row["start_sec"]),
                "quality_score": float(row["quality_score"]),
            }
        )
    return np.vstack(x_rows), pd.DataFrame(rows)


def fit_pca_space(x: np.ndarray, max_components: int = 12, variance: float = 0.90) -> dict[str, object]:
    n_components = min(max_components, max(1, x.shape[0] - 1), x.shape[1])
    pca_full = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca_full.fit(x)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    keep = int(np.searchsorted(cum, variance) + 1)
    keep = min(max(1, keep), n_components)
    pca = PCA(n_components=keep, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(x)
    var = np.maximum(pca.explained_variance_, 1e-8)
    return {"pca": pca, "var": var, "n_components": keep}


def pca_t2_q(x: np.ndarray, space: dict[str, object], prefix: str) -> pd.DataFrame:
    pca: PCA = space["pca"]
    var = space["var"]
    scores = pca.transform(x)
    recon = pca.inverse_transform(scores)
    t2 = np.sum((scores**2) / var, axis=1)
    q = np.mean((x - recon) ** 2, axis=1)
    return pd.DataFrame(
        {
            f"{prefix}_t2": t2,
            f"{prefix}_q": q,
            f"{prefix}_t2_log": np.log1p(t2),
            f"{prefix}_q_log": np.log1p(q),
            f"{prefix}_n_components": int(space["n_components"]),
        }
    )


def transform_waveform_scores(x: np.ndarray, normal_space: dict, impaired_space: dict) -> pd.DataFrame:
    normal = pca_t2_q(x, normal_space, "normal_pca")
    impaired = pca_t2_q(x, impaired_space, "impaired_pca")
    out = pd.concat([normal, impaired], axis=1)
    out["t2_margin_log_normal_minus_impaired"] = out["normal_pca_t2_log"] - out["impaired_pca_t2_log"]
    out["q_margin_log_normal_minus_impaired"] = out["normal_pca_q_log"] - out["impaired_pca_q_log"]
    out["total_margin"] = out["t2_margin_log_normal_minus_impaired"] + out["q_margin_log_normal_minus_impaired"]
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


def run_repeated_8020(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1110000)
    model_sets = {
        "pca_t2_q_all": [
            "normal_pca_t2_log",
            "normal_pca_q_log",
            "impaired_pca_t2_log",
            "impaired_pca_q_log",
            "t2_margin_log_normal_minus_impaired",
            "q_margin_log_normal_minus_impaired",
            "total_margin",
        ],
        "pca_margin_only": ["t2_margin_log_normal_minus_impaired", "q_margin_log_normal_minus_impaired", "total_margin"],
        "normal_anomaly_only": ["normal_pca_t2_log", "normal_pca_q_log"],
    }
    metrics_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        normal_space = fit_pca_space(x_train[y_train == 0])
        impaired_space = fit_pca_space(x_train[y_train == 1])
        train_scores = transform_waveform_scores(x_train, normal_space, impaired_space)
        test_scores = transform_waveform_scores(x_test, normal_space, impaired_space)
        for model_name, cols in model_sets.items():
            clf = logistic(1120000 + repeat)
            clf.fit(train_scores[cols], y_train)
            train_prob = clf.predict_proba(train_scores[cols])[:, 1]
            threshold = threshold_youden(y_train, train_prob)
            test_prob = clf.predict_proba(test_scores[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = calc_metrics(y_test, test_prob, test_pred)
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
    normal_space = fit_pca_space(x[y == 0])
    impaired_space = fit_pca_space(x[y == 1])
    train_scores = transform_waveform_scores(x, normal_space, impaired_space)
    sample_scores = transform_waveform_scores(sample_x, normal_space, impaired_space)
    model_sets = {
        "pca_t2_q_all": [
            "normal_pca_t2_log",
            "normal_pca_q_log",
            "impaired_pca_t2_log",
            "impaired_pca_q_log",
            "t2_margin_log_normal_minus_impaired",
            "q_margin_log_normal_minus_impaired",
            "total_margin",
        ],
        "pca_margin_only": ["t2_margin_log_normal_minus_impaired", "q_margin_log_normal_minus_impaired", "total_margin"],
        "normal_anomaly_only": ["normal_pca_t2_log", "normal_pca_q_log"],
    }
    rows = []
    for model_name, cols in model_sets.items():
        clf = logistic(1130000 + len(model_name))
        clf.fit(train_scores[cols], y)
        prob_train = clf.predict_proba(train_scores[cols])[:, 1]
        threshold = threshold_youden(y, prob_train)
        prob_sample = clf.predict_proba(sample_scores[cols])[:, 1]
        for idx, row in sample_meta.iterrows():
            out = row.to_dict()
            out["model_set"] = model_name
            out["probability_impaired"] = float(prob_sample[idx])
            out["threshold"] = threshold
            out["prediction"] = int(prob_sample[idx] >= threshold)
            for col in sample_scores.columns:
                out[col] = sample_scores.loc[idx, col]
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, meta = build_physionet_waveforms()
    np.save(OUT_DIR / "X_physionet_best10_vmlap_zscore_waveform.npy", x)
    np.save(OUT_DIR / "y_physionet_best10_all_or.npy", y)
    meta.to_csv(OUT_DIR / "physionet_best10_waveform_meta.csv", index=False, encoding="utf-8-sig")

    metrics, preds = run_repeated_8020(x, y, meta)
    metrics.to_csv(OUT_DIR / "waveform_pca_t2_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "waveform_pca_t2_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
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
    summary.to_csv(OUT_DIR / "waveform_pca_t2_metrics_summary.csv", index=False, encoding="utf-8-sig")

    sample_x, sample_meta = build_sample_waveforms()
    np.save(OUT_DIR / "X_sample_best10_vmlap_zscore_waveform.npy", sample_x)
    sample_preds = predict_samples(x, y, sample_x, sample_meta)
    sample_preds.to_csv(OUT_DIR / "waveform_pca_t2_sample_predictions.csv", index=False, encoding="utf-8-sig")

    print("physionet counts")
    print(pd.Series(y).value_counts().sort_index().to_string())
    print("\nmetrics summary")
    print(summary.to_string(index=False))
    print("\nsample predictions")
    print(
        sample_preds[
            [
                "sample_file",
                "model_set",
                "probability_impaired",
                "threshold",
                "prediction",
                "quality_score",
                "normal_pca_t2_log",
                "normal_pca_q_log",
                "impaired_pca_t2_log",
                "impaired_pca_q_log",
                "total_margin",
            ]
        ].to_string(index=False)
    )
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
