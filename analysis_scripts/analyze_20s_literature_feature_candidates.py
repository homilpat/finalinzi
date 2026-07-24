"""Leakage-aware screening of literature-grounded gait features for one 20s test."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from audit_gyro_turn_freewalk40 import event_mask, read_interval
from compare_normal_gait_pattern_features import sample_entropy, shape_cv
from gait_axis_aligned_core import (
    TARGET_FS_HZ,
    acf,
    bandpass,
    peak_in_range,
    spec_entropy,
    window_features,
)


SOURCE = ROOT / "analysis_outputs" / "smoke_top4_subwindows_vs_single19s" / "chosen_segments.csv"
OUT_DIR = ROOT / "analysis_outputs" / "literature_20s_feature_candidate_analysis"
FS = int(TARGET_FS_HZ)
WIN = 10 * FS
STEP = 2 * FS
TARGET_SENSITIVITY = 0.80
SEED = 20260724

BASELINE = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "ap_spec_entropy_median",
]

ACC_WINDOW_FEATURES = [
    "v_stride_regularity",
    "v_step_regularity",
    "v_step_stride_asymmetry",
    "v_stride_width_sec",
    "v_stride_freq_hz",
    "v_spec_peak_ratio",
    "v_spec_entropy",
    "v_sample_entropy",
    "v_peak_timing_sd_pct",
    "v_jerk_rms",
    "ml_stride_regularity",
    "ml_stride_width_sec",
    "ml_spec_peak_ratio",
    "ml_spec_entropy",
    "ml_sample_entropy",
    "ml_peak_timing_sd_pct",
    "ml_jerk_rms",
    "ap_stride_regularity",
    "ap_step_regularity",
    "ap_step_stride_asymmetry",
    "ap_stride_width_sec",
    "ap_spec_peak_ratio",
    "ap_spec_entropy",
    "ap_sample_entropy",
    "ap_peak_timing_sd_pct",
    "ap_jerk_rms",
]

GYRO_WINDOW_FEATURES = [
    "gyro_pitch_stride_regularity",
    "gyro_pitch_spec_entropy",
    "gyro_pitch_sample_entropy",
    "gyro_pitch_rms",
    "gyro_roll_stride_regularity",
    "gyro_roll_spec_entropy",
    "gyro_roll_sample_entropy",
    "gyro_roll_rms",
]


def signal_features(signal: np.ndarray, prefix: str) -> dict:
    filtered = bandpass(np.asarray(signal, dtype=float), float(FS))
    corr = acf(filtered)
    stride_sec, stride_peak, stride_width = peak_in_range(corr, float(FS), 0.80, 1.70)
    if np.isfinite(stride_sec):
        half = stride_sec / 2.0
        _, step_peak, _ = peak_in_range(corr, float(FS), half * 0.6, half * 1.4)
    else:
        step_peak = np.nan
    peak_ratio, entropy = spec_entropy(filtered, float(FS))
    shape_variation, peak_timing = shape_cv(filtered, float(FS), stride_sec)
    downsampled = filtered[np.linspace(0, len(filtered) - 1, 250).round().astype(int)]
    sampen = sample_entropy(downsampled)
    jerk = float(np.sqrt(np.nanmean(np.diff(filtered) ** 2)) * FS)
    return {
        f"{prefix}_stride_regularity": stride_peak,
        f"{prefix}_step_regularity": step_peak,
        f"{prefix}_step_stride_asymmetry": (
            float(abs(step_peak - stride_peak)) if np.isfinite(step_peak) and np.isfinite(stride_peak) else np.nan
        ),
        f"{prefix}_stride_width_sec": stride_width,
        f"{prefix}_stride_freq_hz": 1.0 / stride_sec if np.isfinite(stride_sec) and stride_sec > 0 else np.nan,
        f"{prefix}_spec_peak_ratio": peak_ratio,
        f"{prefix}_spec_entropy": entropy,
        f"{prefix}_sample_entropy": sampen,
        f"{prefix}_stride_shape_cv": shape_variation,
        f"{prefix}_peak_timing_sd_pct": peak_timing,
        f"{prefix}_jerk_rms": jerk,
        f"{prefix}_rms": float(np.sqrt(np.nanmean(filtered**2))),
    }


def gyro_features(signal: np.ndarray, prefix: str) -> dict:
    values = signal_features(signal, prefix)
    return {
        f"{prefix}_stride_regularity": values[f"{prefix}_stride_regularity"],
        f"{prefix}_spec_entropy": values[f"{prefix}_spec_entropy"],
        f"{prefix}_sample_entropy": values[f"{prefix}_sample_entropy"],
        f"{prefix}_rms": values[f"{prefix}_rms"],
    }


def aggregate(values: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for column in columns:
        finite = pd.to_numeric(values[column], errors="coerce").dropna()
        out[f"{column}_median"] = float(finite.median()) if len(finite) else np.nan
        out[f"{column}_iqr"] = (
            float(finite.quantile(0.75) - finite.quantile(0.25)) if len(finite) else np.nan
        )
    return out


def segment_features(acc: np.ndarray, gyro: np.ndarray) -> dict:
    rows = []
    for start in range(0, len(acc) - WIN + 1, STEP):
        acc_window = acc[start : start + WIN]
        gyro_window = gyro[start : start + WIN]
        core = window_features(acc_window)
        v = signal_features(acc_window[:, 0], "v")
        row = {
            "v_stride_regularity": core["v_acf_stride_peak"],
            "v_step_regularity": core["v_acf_step_peak"],
            "v_step_stride_asymmetry": abs(core["v_acf_step_peak"] - core["v_acf_stride_peak"]),
            "v_stride_width_sec": core["v_acf_stride_peak_width_sec"],
            "v_stride_freq_hz": core["v_stride_freq_hz"],
            "v_spec_peak_ratio": v["v_spec_peak_ratio"],
            "v_spec_entropy": v["v_spec_entropy"],
            "v_sample_entropy": v["v_sample_entropy"],
            "v_peak_timing_sd_pct": v["v_peak_timing_sd_pct"],
            "v_jerk_rms": core["v_jerk_rms"],
        }
        ml = signal_features(acc_window[:, 1], "ml")
        ap = signal_features(acc_window[:, 2], "ap")
        row.update({key: ml.get(key, np.nan) for key in ACC_WINDOW_FEATURES if key.startswith("ml_")})
        row.update(
            {
                "ap_stride_regularity": core["ap_acf_stride_peak"],
                "ap_step_regularity": core["ap_acf_step_peak"],
                "ap_step_stride_asymmetry": abs(core["ap_acf_step_peak"] - core["ap_acf_stride_peak"]),
                "ap_stride_width_sec": core["ap_acf_stride_peak_width_sec"],
                "ap_spec_peak_ratio": ap["ap_spec_peak_ratio"],
                "ap_spec_entropy": core["ap_spec_entropy"],
                "ap_sample_entropy": ap["ap_sample_entropy"],
                "ap_peak_timing_sd_pct": ap["ap_peak_timing_sd_pct"],
                "ap_jerk_rms": ap["ap_jerk_rms"],
            }
        )
        row.update(gyro_features(gyro_window[:, 1], "gyro_pitch"))
        row.update(gyro_features(gyro_window[:, 2], "gyro_roll"))
        rows.append(row)

    windows = pd.DataFrame(rows)
    yaw = gyro[:, 0] - np.nanmedian(gyro[:, 0])
    _, turn_count = event_mask(yaw)
    return {
        **aggregate(windows, ACC_WINDOW_FEATURES + GYRO_WINDOW_FEATURES),
        "yaw_turn_event_count_qc": int(turn_count),
        "n_windows": int(len(windows)),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def feature_quality(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    y = table["target"].to_numpy(int)
    for feature in features:
        values = pd.to_numeric(table[feature], errors="coerce")
        valid = values.notna()
        auc = (
            roc_auc_score(y[valid], values[valid])
            if valid.sum() and len(np.unique(y[valid])) == 2
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "missing_rate": float(1.0 - valid.mean()),
                "normal_median": float(values[y == 0].median()),
                "impaired_median": float(values[y == 1].median()),
                "univariate_auc": float(auc),
                "auc_separation": float(abs(auc - 0.5)) if np.isfinite(auc) else np.nan,
                "cliffs_delta_normal_minus_impaired": cliffs_delta(
                    values[y == 0].to_numpy(float), values[y == 1].to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["auc_separation", "missing_rate"], ascending=[False, True]
    )


def vif_values(x: pd.DataFrame) -> pd.Series:
    if x.shape[1] <= 1:
        return pd.Series(1.0, index=x.columns)
    rows = {}
    for column in x.columns:
        others = [name for name in x.columns if name != column]
        score = LinearRegression().fit(x[others], x[column]).score(x[others], x[column])
        rows[column] = 1.0 / max(1.0 - float(score), 1e-12)
    return pd.Series(rows)


class CollinearityAwareSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_features: int = 4,
        missing_limit: float = 0.25,
        corr_limit: float = 0.85,
        vif_limit: float = 5.0,
    ):
        self.max_features = max_features
        self.missing_limit = missing_limit
        self.corr_limit = corr_limit
        self.vif_limit = vif_limit

    def fit(self, x, y):
        frame = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
        frame.columns = list(getattr(x, "columns", range(frame.shape[1])))
        medians = frame.median()
        usable = [
            column
            for column in frame.columns
            if frame[column].isna().mean() <= self.missing_limit
            and frame[column].nunique(dropna=True) > 1
        ]
        imputed = frame[usable].fillna(medians[usable])
        scores = {}
        y = np.asarray(y, dtype=int)
        for column in usable:
            try:
                auc = roc_auc_score(y, imputed[column])
                scores[column] = abs(float(auc) - 0.5)
            except ValueError:
                scores[column] = 0.0
        ranked = sorted(usable, key=lambda column: (-scores[column], str(column)))

        kept = []
        for column in ranked:
            if any(abs(imputed[column].corr(imputed[other], method="spearman")) >= self.corr_limit for other in kept):
                continue
            kept.append(column)
            if len(kept) >= self.max_features:
                break

        while len(kept) > 1:
            vifs = vif_values(imputed[kept])
            if float(vifs.max()) <= self.vif_limit:
                break
            kept.remove(vifs.idxmax())

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.selected_features_ = kept
        self.selected_indices_ = [frame.columns.get_loc(column) for column in kept]
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            return x.loc[:, self.selected_features_]
        return np.asarray(x)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def threshold_for_sensitivity(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(probability)
    best_threshold = float(candidates.min() - 1e-9)
    best_specificity = -1.0
    for threshold in candidates:
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= TARGET_SENSITIVITY and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = specificity
    return best_threshold


def model_pipeline(select: bool) -> Pipeline:
    steps = []
    if select:
        steps.append(("select", CollinearityAwareSelector()))
    steps.extend(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    return Pipeline(steps)


def evaluate_nested(table: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].to_numpy(int)
    repeat_rows = []
    selection_rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    splits = list(outer.split(np.zeros((len(y), 1)), y))

    for name, features in feature_sets.items():
        selectable = name != "baseline_top3"
        repeat_probabilities = np.full((10, len(y)), np.nan)
        repeat_predictions = np.full((10, len(y)), np.nan)
        for split_index, (train_idx, test_idx) in enumerate(splits):
            repeat = split_index // 5
            fold = split_index % 5
            x_train = table.iloc[train_idx][features]
            x_test = table.iloc[test_idx][features]
            y_train = y[train_idx]

            pipeline = model_pipeline(selectable)
            if selectable:
                search = GridSearchCV(
                    pipeline,
                    {
                        "select__max_features": [2, 3, 4, 5],
                        "model__C": [0.1, 0.5, 1.0],
                    },
                    scoring="roc_auc",
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + split_index),
                    n_jobs=1,
                    refit=True,
                )
                search.fit(x_train, y_train)
                fitted = search.best_estimator_
                selected = fitted.named_steps["select"].selected_features_
                best_k = int(search.best_params_["select__max_features"])
                best_c = float(search.best_params_["model__C"])
            else:
                fitted = pipeline.fit(x_train, y_train)
                selected = features
                best_k = len(features)
                best_c = 0.5

            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1000 + split_index)
            train_probability = cross_val_predict(
                clone(fitted), x_train, y_train, cv=inner, method="predict_proba", n_jobs=1
            )[:, 1]
            threshold = threshold_for_sensitivity(y_train, train_probability)
            test_probability = fitted.predict_proba(x_test)[:, 1]
            repeat_probabilities[repeat, test_idx] = test_probability
            repeat_predictions[repeat, test_idx] = (test_probability >= threshold).astype(int)
            selection_rows.append(
                {
                    "representation": name,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_features": "|".join(map(str, selected)),
                    "selected_count": len(selected),
                    "tuned_max_features": best_k,
                    "tuned_C": best_c,
                    "max_vif_train": float(
                        vif_values(
                            pd.DataFrame(
                                SimpleImputer(strategy="median").fit_transform(x_train[selected]),
                                columns=selected,
                            )
                        ).max()
                    ),
                }
            )

        for repeat in range(10):
            probability = repeat_probabilities[repeat]
            prediction = repeat_predictions[repeat].astype(int)
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            repeat_rows.append(
                {
                    "representation": name,
                    "repeat": repeat,
                    "auc": roc_auc_score(y, probability),
                    "sensitivity": recall_score(y, prediction),
                    "specificity": tn / (tn + fp),
                    "precision": precision_score(y, prediction, zero_division=0),
                    "accuracy": accuracy_score(y, prediction),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
    return pd.DataFrame(repeat_rows), pd.DataFrame(selection_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = pd.read_csv(SOURCE)
    rows = []
    for _, item in chosen.iterrows():
        acc, gyro = read_interval(
            str(item["subject_id"]), float(item["start_sec"]), duration_sec=20.0
        )
        rows.append(
            {
                "subject_id": str(item["subject_id"]),
                "target": int(item["target"]),
                "segment_id": str(item["segment_id"]),
                **segment_features(acc, gyro),
            }
        )
    table = pd.DataFrame(rows)

    acc_features = [
        f"{feature}_{stat}"
        for feature in ACC_WINDOW_FEATURES
        for stat in ("median", "iqr")
    ]
    gyro_features_all = [
        f"{feature}_{stat}"
        for feature in GYRO_WINDOW_FEATURES
        for stat in ("median", "iqr")
    ]
    all_features = acc_features + gyro_features_all
    quality = feature_quality(table, all_features)

    corr = table[all_features].corr(method="spearman")
    corr_rows = []
    for index, first in enumerate(all_features):
        for second in all_features[index + 1 :]:
            value = corr.loc[first, second]
            corr_rows.append(
                {
                    "feature_a": first,
                    "feature_b": second,
                    "spearman": value,
                    "abs_spearman": abs(value),
                }
            )
    corr_table = pd.DataFrame(corr_rows).sort_values("abs_spearman", ascending=False)

    imputed_all = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(table[all_features]),
        columns=all_features,
    )
    full_vif = vif_values(imputed_all).rename("vif").reset_index()
    full_vif.columns = ["feature", "vif"]
    full_vif = full_vif.sort_values("vif", ascending=False)

    feature_sets = {
        "baseline_top3": BASELINE,
        "acc_literature_nested": acc_features,
        "acc_plus_gyro_nested": all_features,
    }
    repeat_metrics, selections = evaluate_nested(table, feature_sets)
    summary = (
        repeat_metrics.groupby("representation")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
        .sort_values("auc_mean", ascending=False)
    )

    counts = Counter()
    for value in selections.loc[
        selections["representation"].eq("acc_literature_nested"), "selected_features"
    ]:
        counts.update(value.split("|"))
    selection_frequency = pd.DataFrame(
        [
            {"feature": feature, "selected_fold_count": count, "selection_rate": count / 50.0}
            for feature, count in counts.most_common()
        ]
    )

    table.to_csv(OUT_DIR / "feature_table.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT_DIR / "feature_quality.csv", index=False, encoding="utf-8-sig")
    corr_table.to_csv(OUT_DIR / "spearman_pairs.csv", index=False, encoding="utf-8-sig")
    full_vif.to_csv(OUT_DIR / "full_pool_vif.csv", index=False, encoding="utf-8-sig")
    repeat_metrics.to_csv(OUT_DIR / "nested_cv_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT_DIR / "nested_cv_fold_selections.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "nested_cv_summary.csv", index=False, encoding="utf-8-sig")
    selection_frequency.to_csv(
        OUT_DIR / "acc_selection_frequency.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "n_subjects": int(len(table)),
        "normal": int((table["target"] == 0).sum()),
        "impaired": int((table["target"] == 1).sum()),
        "window_seconds": 10,
        "window_step_seconds": 2,
        "windows_per_measurement": int(table["n_windows"].median()),
        "outer_cv": "RepeatedStratifiedKFold(5 folds x 10 repeats)",
        "inner_cv": "3-fold GridSearchCV; feature ranking, correlation pruning, and VIF pruning fit on training folds only",
        "correlation_limit": 0.85,
        "vif_limit": 5.0,
        "target_sensitivity": TARGET_SENSITIVITY,
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, part in repeat_metrics.groupby("representation"):
        axes[0].scatter([name] * len(part), part["auc"], alpha=0.65, s=30)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Outer-repeat AUC")
    axes[0].set_title("Leakage-aware nested CV")
    top = quality.head(15).sort_values("auc_separation")
    axes[1].barh(top["feature"], top["auc_separation"])
    axes[1].set_xlabel("|univariate AUC - 0.5|")
    axes[1].set_title("Exploratory effect size")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "candidate_analysis.png", dpi=180)
    plt.close(fig)

    print("\nNested CV summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTop exploratory features")
    print(
        quality.head(15)[
            ["feature", "missing_rate", "univariate_auc", "auc_separation"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nACC selection frequency")
    print(selection_frequency.head(15).to_string(index=False))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
