from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import hilbert
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

from run_strict_preprocessing_from_physionet import (  # noqa: E402
    DEFAULT_DATA_DIR,
    butterworth_bandpass,
    butterworth_lowpass,
    load_physical_record,
    unbiased_acf,
)


OUT_DIR = ROOT / "analysis_outputs" / "waveform_representation_model_comparison_all_or"
BEST10_PATH = (
    ROOT
    / "analysis_outputs"
    / "physionet_labwalks_smartphone_shape_extractor_all_or"
    / "physionet_labwalks_shape_best10_all_or.csv"
)
SAMPLE_DIR = ROOT / "보행SAMPLE"
SAMPLE_BEST10_PATH = (
    ROOT
    / "analysis_outputs"
    / "combined_fixed_shape6_sample_predictions"
    / "sample_best10_fixed_shape6_features.csv"
)
FS_OUT = 100.0
WINDOW_SEC = 10.0
N_POINTS = int(FS_OUT * WINDOW_SEC)


def resample_to_100hz(arr: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - FS_OUT) < 1e-6 and len(arr) == N_POINTS:
        return arr.astype(np.float32)
    old_t = np.arange(len(arr), dtype=float) / fs
    new_t = np.arange(N_POINTS, dtype=float) / FS_OUT
    return np.column_stack([np.interp(new_t, old_t, arr[:, i]) for i in range(arr.shape[1])]).astype(np.float32)


def preprocess_segment(acc_vmlap: np.ndarray, fs: float) -> np.ndarray:
    band = np.column_stack(
        [
            butterworth_bandpass(acc_vmlap[:, 0], fs, low=0.6, high=3.0, order=4),
            butterworth_bandpass(acc_vmlap[:, 1], fs, low=0.6, high=3.0, order=4),
            butterworth_bandpass(acc_vmlap[:, 2], fs, low=0.6, high=3.0, order=4),
        ]
    )
    return resample_to_100hz(band, fs)


def window_normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if mode == "amplitude_keep":
        return x.astype(np.float32)
    centered = x - np.nanmean(x, axis=0, keepdims=True)
    if mode == "center_only":
        return centered.astype(np.float32)
    if mode == "rms":
        scale = np.sqrt(np.nanmean(centered**2, axis=0, keepdims=True))
    elif mode == "zscore":
        scale = np.nanstd(centered, axis=0, keepdims=True)
    else:
        raise ValueError(mode)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    return (centered / scale).astype(np.float32)


def raw_waveform_vector(arr: np.ndarray, mode: str) -> np.ndarray:
    return window_normalize(arr, mode).reshape(-1).astype(np.float32)


def acf_curve_vector(arr: np.ndarray, max_lag_sec: float = 3.0) -> np.ndarray:
    max_lag = int(max_lag_sec * FS_OUT)
    parts = []
    for axis in range(3):
        acf = unbiased_acf(arr[:, axis])[: max_lag + 1]
        parts.append(acf)
    return np.concatenate(parts).astype(np.float32)


def spectrum_vector(arr: np.ndarray) -> np.ndarray:
    parts = []
    freqs = np.fft.rfftfreq(len(arr), d=1 / FS_OUT)
    mask = (freqs >= 0.6) & (freqs <= 3.0)
    for axis in range(3):
        x = arr[:, axis] - np.mean(arr[:, axis])
        psd = np.abs(np.fft.rfft(x)) ** 2
        band = psd[mask]
        total = float(np.sum(band))
        if total <= 1e-12:
            parts.append(np.zeros(mask.sum(), dtype=np.float32))
        else:
            parts.append((band / total).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def envelope_vector(arr: np.ndarray, mode: str = "rms") -> np.ndarray:
    env = np.abs(hilbert(arr, axis=0))
    # Downsample envelope to 10 Hz-equivalent 100 points per axis.
    idx = np.linspace(0, len(env) - 1, 100)
    down = np.column_stack([np.interp(idx, np.arange(len(env)), env[:, i]) for i in range(3)])
    return window_normalize(down, mode).reshape(-1).astype(np.float32)


def stride_waveform_vector(arr: np.ndarray, stride_sec: float | None) -> np.ndarray:
    if stride_sec is None or not np.isfinite(stride_sec) or stride_sec <= 0:
        return np.full(600, np.nan, dtype=np.float32)
    stride_n = int(round(stride_sec * FS_OUT))
    if stride_n < int(0.75 * FS_OUT) or stride_n > int(1.75 * FS_OUT):
        return np.full(600, np.nan, dtype=np.float32)
    waves = []
    grid = np.linspace(0, stride_n - 1, 100)
    for start in range(0, len(arr) - stride_n + 1, stride_n):
        seg = arr[start : start + stride_n]
        if len(seg) != stride_n:
            continue
        z = window_normalize(seg, "zscore")
        wave = np.column_stack([np.interp(grid, np.arange(stride_n), z[:, i]) for i in range(3)])
        waves.append(wave)
    if len(waves) < 3:
        return np.full(600, np.nan, dtype=np.float32)
    stack = np.stack(waves, axis=0)
    mean_wave = np.nanmean(stack, axis=0)
    sd_wave = np.nanstd(stack, axis=0)
    return np.concatenate([mean_wave.reshape(-1), sd_wave.reshape(-1)]).astype(np.float32)


def build_representations(arr: np.ndarray, stride_sec: float | None) -> dict[str, np.ndarray]:
    return {
        "raw_amp_keep": raw_waveform_vector(arr, "amplitude_keep"),
        "raw_center_only": raw_waveform_vector(arr, "center_only"),
        "raw_rms_norm": raw_waveform_vector(arr, "rms"),
        "raw_zscore": raw_waveform_vector(arr, "zscore"),
        "acf_curve": acf_curve_vector(arr),
        "spectrum_shape": spectrum_vector(arr),
        "envelope_rms_norm": envelope_vector(arr, "rms"),
        "stride_mean_sd_shape": stride_waveform_vector(arr, stride_sec),
    }


def build_physionet() -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    best = pd.read_csv(BEST10_PATH)
    best = best[best["target"].notna()].copy()
    reps: dict[str, list[np.ndarray]] = {}
    meta_rows = []
    y = []
    for _, row in best.iterrows():
        data, fs, _ = load_physical_record(DEFAULT_DATA_DIR / str(row["source_id"]), channels=(0, 1, 2))
        data = butterworth_lowpass(data, fs, cutoff=20.0, order=4)
        start = int(round(float(row["start_sec"]) * fs))
        end = start + int(round(WINDOW_SEC * fs))
        segment = data[start:end]
        if segment.shape[0] != int(round(WINDOW_SEC * fs)):
            continue
        arr = preprocess_segment(segment, fs)
        stride_sec = row.get("stride_time_median", row.get("stride_duration", np.nan))
        rep = build_representations(arr, float(stride_sec) if pd.notna(stride_sec) else None)
        for key, vec in rep.items():
            reps.setdefault(key, []).append(vec)
        y.append(int(row["target"]))
        meta_rows.append(
            {
                "dataset": "PhysioNet_LabWalks",
                "source_id": row["source_id"],
                "subject_id": row["subject_id"],
                "group_id": f"PhysioNet_LabWalks::{row['subject_id']}",
                "target": int(row["target"]),
                "quality_score": row["quality_score"],
            }
        )
    return {key: np.vstack(vals) for key, vals in reps.items()}, np.asarray(y, dtype=int), pd.DataFrame(meta_rows)


def read_sample_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def resample_sample_csv(df: pd.DataFrame) -> pd.DataFrame:
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


def build_samples() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sample_best = pd.read_csv(SAMPLE_BEST10_PATH)
    reps: dict[str, list[np.ndarray]] = {}
    meta_rows = []
    for _, row in sample_best.iterrows():
        path = SAMPLE_DIR / str(row["sample_file"])
        if not path.exists():
            continue
        rs = resample_sample_csv(read_sample_csv(path))
        acc = sample_acc_vmlap(rs)
        start = int(round(float(row["start_sec"]) * FS_OUT))
        end = start + N_POINTS
        segment = acc[start:end]
        if segment.shape[0] != N_POINTS:
            continue
        arr = preprocess_segment(segment, FS_OUT)
        stride_sec = row.get("stride_time_median", row.get("stride_duration", np.nan))
        rep = build_representations(arr, float(stride_sec) if pd.notna(stride_sec) else None)
        for key, vec in rep.items():
            reps.setdefault(key, []).append(vec)
        meta_rows.append(
            {
                "sample_file": row["sample_file"],
                "start_sec": row["start_sec"],
                "quality_score": row["quality_score"],
            }
        )
    return {key: np.vstack(vals) for key, vals in reps.items()}, pd.DataFrame(meta_rows)


def fit_pca_space(x: np.ndarray, max_components: int = 10, variance: float = 0.90) -> dict[str, object]:
    x = SimpleImputer(strategy="median").fit_transform(x)
    n_components = min(max_components, max(1, x.shape[0] - 1), x.shape[1])
    pca0 = PCA(n_components=n_components, svd_solver="randomized", random_state=42).fit(x)
    keep = int(np.searchsorted(np.cumsum(pca0.explained_variance_ratio_), variance) + 1)
    keep = min(max(1, keep), n_components)
    pca = PCA(n_components=keep, svd_solver="randomized", random_state=42).fit(x)
    return {"pca": pca, "var": np.maximum(pca.explained_variance_, 1e-8), "imputer": SimpleImputer(strategy="median").fit(x)}


def pca_scores(x: np.ndarray, normal_space: dict, impaired_space: dict) -> pd.DataFrame:
    def calc(space: dict, prefix: str) -> pd.DataFrame:
        xi = space["imputer"].transform(x)
        pca: PCA = space["pca"]
        z = pca.transform(xi)
        recon = pca.inverse_transform(z)
        t2 = np.sum((z**2) / space["var"], axis=1)
        q = np.mean((xi - recon) ** 2, axis=1)
        return pd.DataFrame({f"{prefix}_t2_log": np.log1p(t2), f"{prefix}_q_log": np.log1p(q)})

    out = pd.concat([calc(normal_space, "normal"), calc(impaired_space, "impaired")], axis=1)
    out["t2_margin"] = out["normal_t2_log"] - out["impaired_t2_log"]
    out["q_margin"] = out["normal_q_log"] - out["impaired_q_log"]
    out["total_margin"] = out["t2_margin"] + out["q_margin"]
    return out


def template_scores(x: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    imp = SimpleImputer(strategy="median").fit(x_train)
    xt = imp.transform(x_train)
    xe = imp.transform(x)
    normal = np.nanmean(xt[y_train == 0], axis=0)
    impaired = np.nanmean(xt[y_train == 1], axis=0)
    def cos(a, b):
        return (a @ b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b), 1e-12)
    def dist(a, b):
        return np.linalg.norm(a - b, axis=1) / np.sqrt(a.shape[1])
    return pd.DataFrame(
        {
            "cosine_margin": cos(xe, impaired) - cos(xe, normal),
            "euclidean_margin": dist(xe, normal) - dist(xe, impaired),
        }
    )


def logistic(seed: int) -> Pipeline:
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


def metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
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


def run_representation(name: str, x: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1310000)
    metric_rows = []
    pred_rows = []
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        normal_space = fit_pca_space(x_train[y_train == 0])
        impaired_space = fit_pca_space(x_train[y_train == 1])
        pca_train = pca_scores(x_train, normal_space, impaired_space)
        pca_test = pca_scores(x_test, normal_space, impaired_space)
        tmpl_train = template_scores(x_train, x_train, y_train)
        tmpl_test = template_scores(x_test, x_train, y_train)
        model_inputs = {
            "pca_t2_q": pca_train.columns.tolist(),
            "pca_margin": ["t2_margin", "q_margin", "total_margin"],
            "template_margin": [f"tmpl_{c}" for c in tmpl_train.columns],
            "pca_plus_template": pca_train.columns.tolist() + [f"tmpl_{c}" for c in tmpl_train.columns],
        }
        train_all = pd.concat([pca_train, tmpl_train.add_prefix("tmpl_")], axis=1)
        test_all = pd.concat([pca_test, tmpl_test.add_prefix("tmpl_")], axis=1)
        for model_name, cols in model_inputs.items():
            clf = logistic(1320000 + repeat)
            clf.fit(train_all[cols], y_train)
            train_prob = clf.predict_proba(train_all[cols])[:, 1]
            threshold = threshold_youden(y_train, train_prob)
            test_prob = clf.predict_proba(test_all[cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            row = metrics(y_test, test_prob, test_pred)
            row.update({"representation": name, "model_set": model_name, "repeat": repeat, "threshold": threshold})
            metric_rows.append(row)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "representation": name,
                        "model_set": model_name,
                        "repeat": repeat,
                        "target": y_test,
                        "probability_impaired": test_prob,
                        "prediction": test_pred,
                    }
                )
            )
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def predict_samples_for_rep(name: str, x: np.ndarray, y: np.ndarray, sx: np.ndarray, sample_meta: pd.DataFrame) -> pd.DataFrame:
    normal_space = fit_pca_space(x[y == 0])
    impaired_space = fit_pca_space(x[y == 1])
    pca_train = pca_scores(x, normal_space, impaired_space)
    pca_sample = pca_scores(sx, normal_space, impaired_space)
    tmpl_train = template_scores(x, x, y)
    tmpl_sample = template_scores(sx, x, y)
    train_all = pd.concat([pca_train, tmpl_train.add_prefix("tmpl_")], axis=1)
    sample_all = pd.concat([pca_sample, tmpl_sample.add_prefix("tmpl_")], axis=1)
    model_inputs = {
        "pca_t2_q": pca_train.columns.tolist(),
        "pca_margin": ["t2_margin", "q_margin", "total_margin"],
        "template_margin": [f"tmpl_{c}" for c in tmpl_train.columns],
        "pca_plus_template": pca_train.columns.tolist() + [f"tmpl_{c}" for c in tmpl_train.columns],
    }
    rows = []
    for model_name, cols in model_inputs.items():
        clf = logistic(1330000 + len(name) + len(model_name))
        clf.fit(train_all[cols], y)
        train_prob = clf.predict_proba(train_all[cols])[:, 1]
        threshold = threshold_youden(y, train_prob)
        sample_prob = clf.predict_proba(sample_all[cols])[:, 1]
        for idx, row in sample_meta.iterrows():
            out = row.to_dict()
            out.update(
                {
                    "representation": name,
                    "model_set": model_name,
                    "probability_impaired": float(sample_prob[idx]),
                    "threshold": threshold,
                    "prediction": int(sample_prob[idx] >= threshold),
                    "total_margin": sample_all.loc[idx, "total_margin"],
                    "tmpl_cosine_margin": sample_all.loc[idx, "tmpl_cosine_margin"],
                    "tmpl_euclidean_margin": sample_all.loc[idx, "tmpl_euclidean_margin"],
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps, y, meta = build_physionet()
    sample_reps, sample_meta = build_samples()
    meta.to_csv(OUT_DIR / "physionet_waveform_representation_meta.csv", index=False, encoding="utf-8-sig")
    sample_meta.to_csv(OUT_DIR / "sample_waveform_representation_meta.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    all_preds = []
    all_samples = []
    for name, x in reps.items():
        print("running", name, x.shape)
        metric_df, pred_df = run_representation(name, x, y)
        all_metrics.append(metric_df)
        all_preds.append(pred_df)
        all_samples.append(predict_samples_for_rep(name, x, y, sample_reps[name], sample_meta))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    samples_df = pd.concat(all_samples, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "waveform_representation_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(OUT_DIR / "waveform_representation_predictions_by_repeat.csv", index=False, encoding="utf-8-sig")
    samples_df.to_csv(OUT_DIR / "waveform_representation_sample_predictions.csv", index=False, encoding="utf-8-sig")

    summary = (
        metrics_df.groupby(["representation", "model_set"])
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
    summary.to_csv(OUT_DIR / "waveform_representation_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print("\nsummary")
    print(summary.to_string(index=False))
    print("\nsamples")
    print(
        samples_df[
            [
                "sample_file",
                "representation",
                "model_set",
                "probability_impaired",
                "threshold",
                "prediction",
                "total_margin",
                "tmpl_cosine_margin",
                "tmpl_euclidean_margin",
            ]
        ].to_string(index=False)
    )
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
