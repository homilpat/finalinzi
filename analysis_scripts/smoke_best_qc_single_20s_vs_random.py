"""Smoke-test a label-blind best-QC 20s segment against random selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from compare_60s_freewalk_models_100rep import (
    inner_oof_ml,
    threshold_for_min_sensitivity,
)


OUT_DIR = ROOT / "analysis_outputs" / "smoke_best_qc_single_20s"
CACHE_PATH = (
    ROOT
    / "analysis_outputs"
    / "single_20s_segment_model_comparison_100rep"
    / "single_20s_segment_feature_cache.npz"
)
REPEATS = 20
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]


def v2_path() -> Path:
    matches = list(
        ROOT.parent.glob(
            "**/strict_preprocessed_accgyro_v2/"
            "gait_features_strict_20s_accgyro_v2.csv"
        )
    )
    if not matches:
        raise FileNotFoundError("Strict 20s feature table was not found.")
    return matches[0]


def make_lr() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=1000, random_state=0),
            ),
        ]
    )


def load_segments() -> pd.DataFrame:
    cached = np.load(CACHE_PATH, allow_pickle=True)
    meta = pd.DataFrame(cached["meta"].tolist()).reset_index(names="cache_index")
    x_agg = cached["x_agg"].astype(np.float32)
    for index, feature in enumerate(FEATURES):
        meta[feature] = x_agg[:, index]

    quality_columns = [
        "subject_id",
        "start_sec",
        "step_duration",
        "stride_duration",
        "v_peak_power_ratio",
        "gyro_turn_intensity",
        "yaw_band_abs_p95",
    ]
    quality = pd.read_csv(v2_path(), usecols=quality_columns)
    quality["start_sec"] = pd.to_numeric(quality["start_sec"], errors="coerce")
    quality = quality.drop_duplicates(["subject_id", "start_sec"])
    merged = meta.merge(
        quality,
        on=["subject_id", "start_sec"],
        how="left",
        validate="many_to_one",
    )
    return merged


def choose_best_qc(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, part in segments.groupby("subject_id", sort=True):
        valid = part[
            part["step_duration"].between(0.35, 0.80)
            & part["stride_duration"].between(0.70, 1.70)
            & np.isfinite(part["v_peak_power_ratio"])
            & np.isfinite(part["gyro_turn_intensity"])
            & np.isfinite(part["yaw_band_abs_p95"])
        ].copy()
        if valid.empty:
            valid = part.dropna(
                subset=["gyro_turn_intensity", "yaw_band_abs_p95"]
            ).copy()
        if valid.empty:
            valid = part.copy()

        clarity_cutoff = valid["v_peak_power_ratio"].median()
        clear = valid[valid["v_peak_power_ratio"] >= clarity_cutoff]
        if not clear.empty:
            valid = clear
        chosen = valid.sort_values(
            ["gyro_turn_intensity", "yaw_band_abs_p95", "start_sec"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def choose_all_median(segments: pd.DataFrame) -> pd.DataFrame:
    labels = (
        segments[["subject_id", "target"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    aggregated = segments.groupby("subject_id")[FEATURES].median()
    return aggregated.join(labels).reset_index()


def metrics(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "sensitivity": float(recall_score(y, prediction)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_table(
    table: pd.DataFrame, strategy: str, repeat: int
) -> dict:
    table = table.sort_values("subject_id").reset_index(drop=True)
    x = table[FEATURES].to_numpy(np.float32)
    y = table["target"].to_numpy(int)
    outer = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=810000 + repeat
    )
    oof_probability = np.full(len(y), np.nan)
    oof_prediction = np.zeros(len(y), dtype=int)
    thresholds = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y)):
        model = make_lr()
        seed = 900000 + repeat * 100 + fold
        inner_probability = inner_oof_ml(
            model, x[train_idx], y[train_idx], seed
        )
        threshold = threshold_for_min_sensitivity(
            y[train_idx], inner_probability
        )
        model.fit(x[train_idx], y[train_idx])
        probability = model.predict_proba(x[test_idx])[:, 1]
        oof_probability[test_idx] = probability
        oof_prediction[test_idx] = (
            probability >= threshold
        ).astype(int)
        thresholds.append(threshold)
    return {
        "strategy": strategy,
        "repeat": repeat,
        "n_subjects": len(table),
        "threshold_median": float(np.median(thresholds)),
        **metrics(y, oof_probability, oof_prediction),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segments = load_segments()
    best = choose_best_qc(segments)
    all_median = choose_all_median(segments)
    best.to_csv(
        OUT_DIR / "best_qc_selected_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(20260724)
    rows = []
    for repeat in range(REPEATS):
        random_rows = []
        for _, part in segments.groupby("subject_id", sort=True):
            random_rows.append(
                part.iloc[int(rng.integers(0, len(part)))]
            )
        random_table = pd.DataFrame(random_rows)
        rows.append(evaluate_table(random_table, "random_single", repeat))
        rows.append(evaluate_table(best, "best_qc_single", repeat))
        rows.append(
            evaluate_table(all_median, "all_segments_median", repeat)
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "metrics_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        results.groupby("strategy")
        .agg(
            repeats=("repeat", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            threshold_median=("threshold_median", "median"),
        )
        .reset_index()
        .sort_values("auc_mean", ascending=False)
    )
    summary.to_csv(
        OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "repeats": REPEATS,
        "outer_cv": "subject-level StratifiedKFold(5)",
        "threshold": "inner 3-fold OOF, sensitivity >= 0.80",
        "selection_is_label_blind": True,
        "best_qc_rule": (
            "physiologic step/stride duration; subject-level upper half of "
            "v_peak_power_ratio; minimum gyro_turn_intensity then "
            "yaw_band_abs_p95"
        ),
        "features_not_used_for_selection": FEATURES,
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
