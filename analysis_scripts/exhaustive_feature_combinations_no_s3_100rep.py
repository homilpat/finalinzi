"""Exhaustive no-s3 LR screen for all 1-4 feature combinations.

This is a development screen of fixed combinations, not a nested estimate of
the performance of selecting the best combination. Every fixed combination is
evaluated on the same subject-level StratifiedKFold splits.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "analysis_outputs"
    / "nested_feature_selection_final10_100rep"
    / "subject_candidate_table.csv"
)
NO_S3_OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "no_s3_exhaustive_feature_combinations_100rep"
)
S3_TARGET_SOURCE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)

FEATURES = [
    "v_acf_step_stride_symmetry_median",
    "v_acf_step_stride_symmetry_iqr",
    "ap_acf_step_stride_symmetry_median",
    "ap_acf_step_stride_symmetry_iqr",
    "v_stride_freq_hz_median",
    "v_stride_freq_hz_iqr",
    "ap_spec_entropy_median",
    "ap_spec_entropy_iqr",
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
]
N_REPEATS = 100
N_SPLITS = 5
OUTER_SEED = 810000


def combination_diagnostics(
    table: pd.DataFrame, feature_names: list[str]
) -> tuple[float, float]:
    values = table[feature_names].to_numpy(float)
    if len(feature_names) == 1:
        return 0.0, 1.0
    correlation = (
        table[feature_names].corr(method="spearman").to_numpy(float)
    )
    upper = np.triu_indices(len(feature_names), k=1)
    max_abs_correlation = float(np.max(np.abs(correlation[upper])))
    vif_values = []
    for index in range(len(feature_names)):
        other = [j for j in range(len(feature_names)) if j != index]
        r_squared = LinearRegression().fit(
            values[:, other], values[:, index]
        ).score(values[:, other], values[:, index])
        vif_values.append(
            float("inf") if r_squared >= 1 else 1.0 / (1.0 - r_squared)
        )
    return max_abs_correlation, float(max(vif_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-mode",
        choices=["no-s3", "s3"],
        default="no-s3",
        help="Clinical target definition to use for the fixed feature screen.",
    )
    args = parser.parse_args()
    out_dir = (
        NO_S3_OUT_DIR
        if args.label_mode == "no-s3"
        else ROOT
        / "analysis_outputs"
        / "s3_exhaustive_feature_combinations_100rep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SOURCE)
    if args.label_mode == "s3":
        s3_targets = pd.read_csv(
            S3_TARGET_SOURCE, usecols=["subject_id", "target"]
        )
        table = (
            table.drop(columns=["target"])
            .merge(
                s3_targets,
                on="subject_id",
                how="inner",
                validate="one_to_one",
            )
        )
    required = FEATURES + ["subject_id", "target"]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    table = table.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    if len(table) != 71:
        raise ValueError(f"Expected 71 subjects, found {len(table)}")

    x = table[FEATURES].to_numpy(float)
    y = table["target"].to_numpy(int)
    combo_specs = [
        (size, combo)
        for size in range(1, 5)
        for combo in combinations(range(len(FEATURES)), size)
    ]
    auc_by_combo = np.full(
        (len(combo_specs), N_REPEATS), np.nan, dtype=float
    )

    for repeat in range(N_REPEATS):
        if repeat % 10 == 0:
            print(f"repeat {repeat}/{N_REPEATS}", flush=True)
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=OUTER_SEED + repeat,
        )
        scaled_folds = []
        for train_idx, test_idx in splitter.split(x, y):
            scaler = RobustScaler()
            scaled_folds.append(
                (
                    train_idx,
                    test_idx,
                    scaler.fit_transform(x[train_idx]),
                    scaler.transform(x[test_idx]),
                )
            )

        for combo_index, (_, combo) in enumerate(combo_specs):
            oof = np.full(len(y), np.nan)
            columns = list(combo)
            for train_idx, test_idx, x_train, x_test in scaled_folds:
                model = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                )
                model.fit(x_train[:, columns], y[train_idx])
                oof[test_idx] = model.predict_proba(
                    x_test[:, columns]
                )[:, 1]
            auc_by_combo[combo_index, repeat] = roc_auc_score(y, oof)

    rows = []
    repeat_rows = []
    for combo_index, (size, combo) in enumerate(combo_specs):
        values = auc_by_combo[combo_index]
        feature_names = [FEATURES[index] for index in combo]
        combo_id = "|".join(feature_names)
        max_correlation, max_vif = combination_diagnostics(
            table, feature_names
        )
        rows.append(
            {
                "rank": 0,
                "n_features": size,
                "features": combo_id,
                "auc_mean": float(np.mean(values)),
                "auc_std": float(np.std(values, ddof=1)),
                "auc_ci_lo": float(np.quantile(values, 0.025)),
                "auc_ci_hi": float(np.quantile(values, 0.975)),
                "auc_median": float(np.median(values)),
                "max_abs_spearman": max_correlation,
                "max_vif": max_vif,
                "passes_corr_vif": (
                    max_correlation <= 0.85 and max_vif <= 5.0
                ),
                "n_repeats": N_REPEATS,
            }
        )
        repeat_rows.extend(
            {
                "features": combo_id,
                "n_features": size,
                "repeat": repeat,
                "auc": float(value),
            }
            for repeat, value in enumerate(values)
        )

    summary = pd.DataFrame(rows).sort_values(
        ["auc_mean", "n_features", "features"],
        ascending=[False, True, True],
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["admissible_rank"] = pd.NA
    admissible_indices = summary.index[summary["passes_corr_vif"]]
    summary.loc[admissible_indices, "admissible_rank"] = np.arange(
        1, len(admissible_indices) + 1
    )
    summary.to_csv(
        out_dir / "all_1_to_4_feature_combinations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.loc[summary["n_features"].between(2, 4)].to_csv(
        out_dir / "all_2_to_4_feature_combinations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(repeat_rows).to_csv(
        out_dir / "repeat_auc_all_combinations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for size in range(1, 5):
        summary.loc[summary["n_features"].eq(size)].to_csv(
            out_dir / f"size_{size}_combinations.csv",
            index=False,
            encoding="utf-8-sig",
        )

    final_three = {
        "v_jerk_rms_median",
        "v_jerk_rms_iqr",
        "v_acf_step_stride_symmetry_iqr",
    }
    selected = summary.loc[
        summary["features"].map(
            lambda value: set(value.split("|")) == final_three
        )
    ].iloc[0]
    metadata = {
        "status": "development_screen_not_nested_selection_performance",
        "subjects": int(len(table)),
        "normal_subjects": int((y == 0).sum()),
        "impaired_subjects": int((y == 1).sum()),
        "candidate_features": FEATURES,
        "combination_sizes": [1, 2, 3, 4],
        "total_combinations": int(len(summary)),
        "two_to_four_combinations": int(
            summary["n_features"].between(2, 4).sum()
        ),
        "evaluation": "StratifiedKFold(5) x 100 identical subject-level splits",
        "preprocessing": "RobustScaler fit on each training fold only",
        "model": "LogisticRegression(C=1, L2, lbfgs)",
        "label": (
            "TUG>=12 OR FSST>=15 OR BERG<52 OR DGI<=19 "
            "OR base_velocity<1.0"
            + (" OR s3_velocity<1.0" if args.label_mode == "s3" else "")
        ),
        "final_three_screen_auc_mean": float(selected["auc_mean"]),
        "warning": (
            "Ranking all combinations and reporting the winner on the same CV "
            "is optimistic. Use the separate nested feature-selection audit "
            "for selection-process performance."
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nTop 20 combinations")
    print(summary.head(20).to_string(index=False))
    print("\nFinal deployed three-feature combination")
    print(selected.to_string())


if __name__ == "__main__":
    main()
