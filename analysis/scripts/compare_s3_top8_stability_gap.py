"""Compare the top eight s3 feature combinations for stability and AUC gap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
RANKING = (
    ROOT
    / "analysis_outputs"
    / "s3_exhaustive_feature_combinations_100rep"
    / "all_1_to_4_feature_combinations.csv"
)
FEATURE_TABLE = (
    ROOT
    / "analysis_outputs"
    / "nested_feature_selection_final10_100rep"
    / "subject_candidate_table.csv"
)
S3_TARGET_TABLE = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "s3_top8_stability_train_test_gap_100rep"
)
CURRENT = {
    "v_acf_step_stride_symmetry_iqr",
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
}
N_REPEATS = 100
N_SPLITS = 5
OUTER_SEED = 810000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(RANKING).sort_values("rank").head(8).copy()
    ranking["feature_list"] = ranking["features"].str.split("|")

    values = pd.read_csv(FEATURE_TABLE).drop(columns=["target"])
    targets = pd.read_csv(
        S3_TARGET_TABLE, usecols=["subject_id", "target"]
    )
    table = values.merge(
        targets, on="subject_id", how="inner", validate="one_to_one"
    )
    y = table["target"].to_numpy(int)

    repeat_rows = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=OUTER_SEED + repeat,
        )
        splits = list(splitter.split(np.zeros(len(y)), y))
        for _, spec in ranking.iterrows():
            features = list(spec["feature_list"])
            x = table[features].to_numpy(float)
            oof = np.full(len(y), np.nan)
            train_aucs = []
            test_aucs = []
            for train_index, test_index in splits:
                scaler = RobustScaler()
                x_train = scaler.fit_transform(x[train_index])
                x_test = scaler.transform(x[test_index])
                model = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                )
                model.fit(x_train, y[train_index])
                train_probability = model.predict_proba(x_train)[:, 1]
                test_probability = model.predict_proba(x_test)[:, 1]
                oof[test_index] = test_probability
                train_aucs.append(
                    roc_auc_score(y[train_index], train_probability)
                )
                test_aucs.append(
                    roc_auc_score(y[test_index], test_probability)
                )
            train_auc = float(np.mean(train_aucs))
            fold_test_auc = float(np.mean(test_aucs))
            repeat_rows.append(
                {
                    "screen_rank": int(spec["rank"]),
                    "features": spec["features"],
                    "n_features": int(spec["n_features"]),
                    "is_current": set(features) == CURRENT,
                    "repeat": repeat,
                    "oof_auc": float(roc_auc_score(y, oof)),
                    "train_auc": train_auc,
                    "fold_test_auc": fold_test_auc,
                    "train_test_gap": train_auc - fold_test_auc,
                }
            )

    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(
        OUT_DIR / "repeat_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        repeats.groupby(
            ["screen_rank", "features", "n_features", "is_current"],
            as_index=False,
        )
        .agg(
            oof_auc_mean=("oof_auc", "mean"),
            oof_auc_std=("oof_auc", "std"),
            oof_auc_min=("oof_auc", "min"),
            oof_auc_max=("oof_auc", "max"),
            train_auc_mean=("train_auc", "mean"),
            fold_test_auc_mean=("fold_test_auc", "mean"),
            train_test_gap_mean=("train_test_gap", "mean"),
            train_test_gap_std=("train_test_gap", "std"),
        )
        .sort_values("oof_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary["auc_rank"] = np.arange(1, len(summary) + 1)
    summary["stability_rank"] = (
        summary["oof_auc_std"].rank(method="min").astype(int)
    )
    summary["gap_rank"] = (
        summary["train_test_gap_mean"].abs().rank(method="min").astype(int)
    )
    summary.to_csv(
        OUT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "status": "diagnostic_comparison_not_a_new_selection_procedure",
        "subjects": int(len(table)),
        "normal": int((y == 0).sum()),
        "impaired": int((y == 1).sum()),
        "candidate_source": "top eight from the fixed 385-combination screen",
        "splits": "StratifiedKFold(5) x 100; same splits for every candidate",
        "warning": (
            "Candidates were chosen using the same data. Train-test gap and "
            "repeat SD are diagnostics, not unbiased model-selection evidence."
        ),
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Written: {OUT_DIR}")


if __name__ == "__main__":
    main()
