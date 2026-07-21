from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from extract_physionet_labwalks_shape_features_all_or import extract_shape_features  # noqa: E402
from model_combined_fixed_shape6_directional_8020 import (  # noqa: E402
    FEATURES,
    fit_ref,
    make_subject_table,
    transform,
)


OUT_DIR = ROOT / "analysis_outputs" / "combined_fixed_shape6_sample_predictions"
SAMPLE_DIR = ROOT / "보행SAMPLE"


def read_sample_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def resample_100hz(df: pd.DataFrame) -> pd.DataFrame:
    t = (pd.to_numeric(df["Timestamp_ns"], errors="coerce") - pd.to_numeric(df["Timestamp_ns"], errors="coerce").iloc[0]) / 1e9
    valid = t.notna()
    df = df.loc[valid].copy()
    t = t.loc[valid].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    df = df.iloc[order].reset_index(drop=True)
    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]
    df = df.iloc[unique_idx].reset_index(drop=True)
    if len(t) < 20:
        raise ValueError("too few rows")
    end = float(t[-1])
    grid = np.arange(0.0, end + 1e-9, 0.01)
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


def sample_acc_gyro(resampled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"Acc_Vertical_g", "Acc_ML_g", "Acc_AP_g"}.issubset(resampled.columns):
        acc = np.column_stack(
            [
                resampled["Acc_Vertical_g"].to_numpy(float),
                resampled["Acc_ML_g"].to_numpy(float),
                resampled["Acc_AP_g"].to_numpy(float),
            ]
        )
    else:
        # Older calibrated sample mapping from previous axis check:
        # V=Acc_Z, ML=Acc_X, AP=Acc_Y.
        acc = np.column_stack(
            [
                resampled["Acc_Z"].to_numpy(float),
                resampled["Acc_X"].to_numpy(float),
                resampled["Acc_Y"].to_numpy(float),
            ]
        )
    if "Gyro_Roll_deg_s" in resampled.columns:
        roll = resampled["Gyro_Roll_deg_s"].to_numpy(float)
    elif "Gyro_Clean_Z" in resampled.columns:
        roll = np.rad2deg(resampled["Gyro_Clean_Z"].to_numpy(float))
    else:
        roll = np.zeros(len(resampled))
    gyro = np.column_stack([np.zeros(len(resampled)), np.zeros(len(resampled)), roll])
    return acc, gyro


def extract_sample_best10(path: Path) -> dict:
    raw = read_sample_csv(path)
    rs = resample_100hz(raw)
    acc, gyro = sample_acc_gyro(rs)
    fs = 100.0
    win = int(round(10.0 * fs))
    step = int(round(1.0 * fs))
    rows = []
    for start in range(0, len(rs) - win + 1, step):
        end = start + win
        feats, quality = extract_shape_features(acc[start:end], gyro[start:end], fs)
        row = {
            "sample_file": path.name,
            "start_sec": start / fs,
            "end_sec": end / fs,
            "quality_score": quality,
            **feats,
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No valid window for {path}")
    windows = pd.DataFrame(rows)
    windows.to_csv(OUT_DIR / f"{path.stem}_shape_windows.csv", index=False, encoding="utf-8-sig")
    best = windows.sort_values("quality_score", ascending=False).iloc[0].to_dict()
    return best


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
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens + spec - 1 > best_j:
            best_j = sens + spec - 1
            best_t = float(t)
    return best_t


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = make_subject_table()
    y = table["target"].astype(int).to_numpy()
    ref = fit_ref(table)
    x_train = transform(table, ref)
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
    fitted = {}
    thresholds = {}
    for name, cols in model_sets.items():
        clf = model(910000 + len(name))
        clf.fit(x_train[cols], y)
        prob = clf.predict_proba(x_train[cols])[:, 1]
        thresholds[name] = threshold_youden(y, prob)
        fitted[name] = (clf, cols)

    sample_rows = []
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        best = extract_sample_best10(path)
        sample_rows.append(best)
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(OUT_DIR / "sample_best10_fixed_shape6_features.csv", index=False, encoding="utf-8-sig")
    x_sample = transform(samples, ref)

    pred_rows = []
    for idx, sample in samples.iterrows():
        for name, (clf, cols) in fitted.items():
            prob = float(clf.predict_proba(x_sample.loc[[idx], cols])[:, 1][0])
            pred_rows.append(
                {
                    "sample_file": sample["sample_file"],
                    "model_set": name,
                    "probability_impaired": prob,
                    "threshold": thresholds[name],
                    "prediction": int(prob >= thresholds[name]),
                    "best10_start_sec": sample["start_sec"],
                    "quality_score": sample["quality_score"],
                    **{feature: sample.get(feature, np.nan) for feature in FEATURES},
                    **{col: x_sample.loc[idx, col] for col in x_sample.columns},
                }
            )
    preds = pd.DataFrame(pred_rows)
    preds.to_csv(OUT_DIR / "sample_predictions_combined_fixed_shape6.csv", index=False, encoding="utf-8-sig")
    print(preds[["sample_file", "model_set", "probability_impaired", "threshold", "prediction", "best10_start_sec", "quality_score", "fixed_shape6_directional_mean", "fixed_shape6_directional_count_pos"]].to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
