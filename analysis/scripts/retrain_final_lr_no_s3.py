"""Retrain the deployed three-feature LR with the revised no-s3 label."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
)
SUBJECT_TABLE = SOURCE_DIR / "subject_feature_table.csv"
METRICS_SUMMARY = SOURCE_DIR / "metrics_summary.csv"
MODEL_DST = ROOT / "backend" / "models" / "gait_daily_clinical_3feat.joblib"
META_DST = (
    ROOT
    / "giukhaji"
    / "models"
    / "gait_daily_clinical_3feat_metadata.json"
)

LEGACY_TO_CANONICAL = {
    "v_harmonic_ratio_iqr": "v_acf_step_stride_symmetry_iqr",
}
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_acf_step_stride_symmetry_iqr",
]
LABEL_RULE = (
    "TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19 "
    "OR base_velocity < 1.0"
)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )


def main() -> None:
    table = pd.read_csv(SUBJECT_TABLE).rename(columns=LEGACY_TO_CANONICAL)
    missing = sorted(set(FEATURES + ["target"]) - set(table.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    table = table.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    y = table["target"].to_numpy(int)
    if len(table) != 71 or (y == 0).sum() != 37 or (y == 1).sum() != 34:
        raise ValueError(
            "Unexpected no-s3 cohort: "
            f"n={len(table)} normal={(y == 0).sum()} impaired={(y == 1).sum()}"
        )

    comparison = pd.read_csv(METRICS_SUMMARY)
    lr_metrics = comparison.loc[comparison["model"].eq("LR")].iloc[0]
    threshold = float(lr_metrics["threshold_median"])

    model = make_model()
    model.fit(table[FEATURES], y)
    train_probability = model.predict_proba(table[FEATURES])[:, 1]
    if not np.isfinite(train_probability).all():
        raise ValueError("Final model produced non-finite probabilities.")

    old_metadata = {}
    if META_DST.exists():
        old_metadata = json.loads(META_DST.read_text(encoding="utf-8"))

    feature_summary = {}
    for feature in FEATURES:
        normal = table.loc[y == 0, feature]
        impaired = table.loc[y == 1, feature]
        feature_summary[feature] = {
            "normal_median": float(normal.median()),
            "impaired_median": float(impaired.median()),
            "difference": float(impaired.median() - normal.median()),
        }

    validation = {
        "outer_cv": "subject-level StratifiedKFold(5) x 100 repeats",
        "inner_threshold_cv": (
            "3-fold OOF within each outer-training fold; "
            "sensitivity >= 0.80 then maximum specificity"
        ),
        "auc_mean": float(lr_metrics["test_auc_mean"]),
        "auc_std": float(lr_metrics["test_auc_std"]),
        "auc_95ci": [
            float(lr_metrics["auc_ci_lo"]),
            float(lr_metrics["auc_ci_hi"]),
        ],
        "sensitivity_mean": float(lr_metrics["sensitivity_mean"]),
        "specificity_mean": float(lr_metrics["specificity_mean"]),
        "precision_mean": float(lr_metrics["precision_mean"]),
        "accuracy_mean": float(lr_metrics["accuracy_mean"]),
        "f1_mean": float(lr_metrics["f1_mean"]),
        "threshold_median": threshold,
        "aggregation_condition": (
            "median of all available 20-second segment-level final3 rows "
            "per subject"
        ),
    }
    artifact = {
        "pipeline": model,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_strategy": (
            "median of training-only nested thresholds; "
            "sensitivity_ge_0.80_max_specificity"
        ),
        "model_mode": "daily_3feat_acconly_no_s3",
        "label_source": LABEL_RULE,
        "excluded_label_rule": (
            "s3_velocity < 1.0 m/s is not used as a standalone criterion"
        ),
        "clinical_label_criteria": {
            "TUG": ">= 12 seconds",
            "FSST": ">= 15 seconds",
            "BERG": "< 52",
            "DGI": "<= 19",
            "base_velocity": "< 1.0 m/s",
        },
        "train_n_subjects": int(len(table)),
        "train_n_normal": int((y == 0).sum()),
        "train_n_impaired": int((y == 1).sum()),
        "feature_summary_by_label": feature_summary,
        "feature_aliases": LEGACY_TO_CANONICAL,
        "feature_definition_note": (
            "The third feature is the IQR across subwindows of the vertical "
            "acceleration ACF step-peak/stride-peak ratio (Ad1/Ad2), not a "
            "Fourier harmonic ratio."
        ),
        "validation": validation,
        "training_source": SUBJECT_TABLE.relative_to(ROOT).as_posix(),
        "signal_correction": old_metadata.get("signal_correction", {}),
    }

    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_DST)
    metadata = {key: value for key, value in artifact.items() if key != "pipeline"}
    META_DST.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"saved={MODEL_DST}\n"
        f"subjects={len(table)} normal={(y == 0).sum()} impaired={(y == 1).sum()}\n"
        f"threshold={threshold:.6f}\n"
        f"nested_auc={validation['auc_mean']:.4f} "
        f"sensitivity={validation['sensitivity_mean']:.4f} "
        f"specificity={validation['specificity_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
