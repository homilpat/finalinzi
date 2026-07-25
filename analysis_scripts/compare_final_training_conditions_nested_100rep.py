"""Compare models under the deployed LR model's original training conditions.

Each subject contributes all available 20-second segment summaries. Classical
models receive the subject-level median of the deployed three features. CNN1D
and LSTM receive the sequence of the same segment-level three features.
Evaluation uses nested subject-level stratified CV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts"))

import compare_60s_freewalk_models_100rep as engine


OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
)
CACHE_PATH = OUT_DIR / "subject_all_segments_final3_cache.npz"
ASSIGNMENT_PATH = OUT_DIR / "subject_repeat_assignments.csv"
TABLE_PATH = (
    ROOT
    / "analysis_outputs"
    / "daily_subwindow_median_iqr"
    / "subwindow_median_iqr_table.csv"
)
FEATURES = [
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
    "v_harmonic_ratio_iqr",
]
MAX_SEGMENTS = 100
ENGINE_MAKE_MODEL = engine.make_model


def clinical_path() -> Path:
    matches = list(
        ROOT.parent.glob(
            "**/clinical_motor_label_modeling/subject_features_with_clinical.csv"
        )
    )
    if not matches:
        raise FileNotFoundError("Clinical label table was not found.")
    return matches[0]


def clinical_target(clinical: pd.DataFrame) -> pd.Series:
    columns = [
        "TUG",
        "FSST",
        "BERG",
        "DGI",
        "base(velocity)",
        "s3(velocity)",
    ]
    for column in columns:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")
    return (
        (clinical["TUG"] >= 12)
        | (clinical["FSST"] >= 15)
        | (clinical["BERG"] < 52)
        | (clinical["DGI"] <= 19)
        | (clinical["base(velocity)"] < 1.0)
        | (clinical["s3(velocity)"] < 1.0)
    ).astype(int)


def build_subject_cache() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_path = OUT_DIR / "subject_feature_table.csv"
    sequence_index_path = OUT_DIR / "subject_sequence_index.csv"
    if (
        CACHE_PATH.exists()
        and ASSIGNMENT_PATH.exists()
        and subject_path.exists()
        and sequence_index_path.exists()
    ):
        assignments = pd.read_csv(ASSIGNMENT_PATH)
        cached = np.load(CACHE_PATH)
        return assignments, cached["x_agg"], cached["x_seq"]

    segments = pd.read_csv(TABLE_PATH)
    clinical = pd.read_csv(clinical_path(), encoding="utf-8-sig")
    clinical["target"] = clinical_target(clinical)
    labels = clinical[["subject_id", "target"]].drop_duplicates("subject_id")
    segments = segments.drop(
        columns=[
            column
            for column in segments.columns
            if column in {"target", "label", "clinical_target"}
            or column.startswith("target_")
        ],
        errors="ignore",
    )
    segments = segments.merge(labels, on="subject_id", how="inner")
    segments = segments.dropna(subset=FEATURES).copy()
    segments["_source_order"] = np.arange(len(segments))
    segments = segments.sort_values(
        ["subject_id", "_source_order"], kind="stable"
    ).reset_index(drop=True)

    subject_rows = []
    sequence_rows = []
    sequence_index_rows = []
    for cache_index, (subject_id, part) in enumerate(
        segments.groupby("subject_id", sort=True)
    ):
        values = part[FEATURES].to_numpy(np.float32)
        values = values[:MAX_SEGMENTS]
        sequence = np.full(
            (MAX_SEGMENTS, len(FEATURES)), np.nan, dtype=np.float32
        )
        sequence[: len(values)] = values
        sequence_rows.append(sequence)
        subject_rows.append(
            {
                "cache_index": cache_index,
                "subject_id": str(subject_id),
                "target": int(part["target"].iloc[0]),
                "n_segments": int(len(values)),
                **{
                    feature: float(np.nanmedian(values[:, index]))
                    for index, feature in enumerate(FEATURES)
                },
            }
        )
        sequence_index_rows.append(
            {
                "cache_index": cache_index,
                "subject_id": str(subject_id),
                "n_segments": int(len(values)),
            }
        )

    subjects = pd.DataFrame(subject_rows)
    x_agg = subjects[FEATURES].to_numpy(np.float32)
    x_seq = np.stack(sequence_rows).astype(np.float32)
    np.savez_compressed(CACHE_PATH, x_agg=x_agg, x_seq=x_seq)
    subjects.to_csv(subject_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sequence_index_rows).to_csv(
        sequence_index_path, index=False, encoding="utf-8-sig"
    )

    assignments = pd.concat(
        [
            subjects[["cache_index", "subject_id", "target"]].assign(
                repeat=repeat,
                segment_id=lambda frame: frame["subject_id"]
                + "__all_segments",
                start_sec=np.nan,
            )
            for repeat in range(engine.N_REPEATS)
        ],
        ignore_index=True,
    )
    assignments.to_csv(
        ASSIGNMENT_PATH, index=False, encoding="utf-8-sig"
    )
    return assignments, x_agg, x_seq


def configure_engine() -> None:
    engine.OUT_DIR = OUT_DIR
    engine.CACHE_PATH = CACHE_PATH
    engine.ASSIGNMENT_PATH = ASSIGNMENT_PATH
    engine.MODEL_RESULT_DIR = OUT_DIR / "model_results"
    engine.ROC_DIR = OUT_DIR / "roc_curves"
    engine.CM_DIR = OUT_DIR / "confusion_matrices"
    engine.CANDIDATE_DIR = OUT_DIR / "candidate_models"
    engine.PLOT_TITLE = (
        "All daily segments aggregated by subject: pooled ROC over 100 repeats"
    )
    engine.FEATURES = FEATURES
    engine.SEQUENCE_FEATURES = FEATURES
    engine.make_model = make_comparison_model
    for directory in [
        OUT_DIR,
        engine.MODEL_RESULT_DIR,
        engine.ROC_DIR,
        engine.CM_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def make_comparison_model(name: str, seed: int) -> Pipeline:
    if name != "LR":
        return ENGINE_MAKE_MODEL(name, seed)
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


def save_subject_consensus_plots(predictions: pd.DataFrame) -> None:
    """Collapse repeated OOF predictions so each subject appears once."""
    rows = []
    metric_rows = []
    all_roc_fig, all_roc_ax = plt.subplots(figsize=(8, 7))

    for model_name, part in predictions.groupby("model", sort=True):
        subject = (
            part.groupby("subject_id", as_index=False)
            .agg(
                target=("target", "first"),
                probability_mean=("probability", "mean"),
                probability_median=("probability", "median"),
                threshold_mean=("threshold", "mean"),
                threshold_median=("threshold", "median"),
                positive_vote_rate=("prediction", "mean"),
                n_oof_predictions=("prediction", "size"),
            )
        )
        # Each repeat used its own training-only threshold. Majority voting
        # preserves those decisions without inventing a post-hoc threshold.
        subject["prediction"] = (
            subject["positive_vote_rate"] >= 0.5
        ).astype(int)
        subject["model"] = model_name
        rows.append(subject)

        y = subject["target"].to_numpy(int)
        probability = subject["probability_mean"].to_numpy(float)
        prediction = subject["prediction"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()
        auc = roc_auc_score(y, probability)
        metric_rows.append(
            {
                "model": model_name,
                "n_subjects": len(subject),
                "auc_subject_mean_oof_probability": auc,
                "sensitivity_majority_vote": recall_score(
                    y, prediction, zero_division=0
                ),
                "specificity_majority_vote": (
                    tn / (tn + fp) if tn + fp else np.nan
                ),
                "precision_majority_vote": precision_score(
                    y, prediction, zero_division=0
                ),
                "accuracy_majority_vote": accuracy_score(y, prediction),
                "f1_majority_vote": f1_score(
                    y, prediction, zero_division=0
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

        fpr, tpr, _ = roc_curve(y, probability)
        all_roc_ax.plot(
            fpr, tpr, lw=1.8, label=f"{model_name} AUC={auc:.3f}"
        )
        roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
        roc_ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        roc_ax.plot([0, 1], [0, 1], "k--", lw=1)
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title(f"{model_name}: subject-level consensus ROC (n=71)")
        roc_ax.grid(alpha=0.25)
        roc_ax.legend(loc="lower right")
        roc_fig.tight_layout()
        roc_fig.savefig(
            engine.ROC_DIR / f"roc_{model_name}_subject_consensus.png",
            dpi=180,
        )
        plt.close(roc_fig)

        matrix = np.asarray([[tn, fp], [fn, tp]])
        cm_fig, cm_ax = plt.subplots(figsize=(4.5, 4))
        cm_ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                cm_ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
        cm_ax.set_xticks([0, 1], ["Pred normal", "Pred impaired"])
        cm_ax.set_yticks([0, 1], ["True normal", "True impaired"])
        cm_ax.set_title(f"{model_name}: subject consensus (n=71)")
        cm_fig.tight_layout()
        cm_fig.savefig(
            engine.CM_DIR
            / f"confusion_matrix_{model_name}_subject_consensus.png",
            dpi=180,
        )
        plt.close(cm_fig)

    all_roc_ax.plot([0, 1], [0, 1], "k--", lw=1)
    all_roc_ax.set_xlabel("False Positive Rate")
    all_roc_ax.set_ylabel("True Positive Rate")
    all_roc_ax.set_title(
        "Subject-level mean OOF probability over 100 repeats (n=71)"
    )
    all_roc_ax.grid(alpha=0.25)
    all_roc_ax.legend(loc="lower right", fontsize=8)
    all_roc_fig.tight_layout()
    all_roc_fig.savefig(
        engine.ROC_DIR / "roc_all_models_subject_consensus.png", dpi=180
    )
    plt.close(all_roc_fig)

    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / "subject_consensus_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_rows).sort_values(
        "auc_subject_mean_oof_probability", ascending=False
    ).to_csv(
        OUT_DIR / "subject_consensus_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def finalize() -> pd.DataFrame:
    metrics, predictions = engine.collect_results()
    if metrics.empty:
        raise RuntimeError("No completed model results were found.")
    summary = engine.summarize(metrics)
    metrics.to_csv(
        OUT_DIR / "metrics_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUT_DIR / "predictions_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT_DIR / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    engine.save_plots(predictions)
    save_subject_consensus_plots(predictions)
    metadata = {
        "status": "experimental_comparison_deployed_model_unchanged",
        "features": FEATURES,
        "subjects": int(predictions["subject_id"].nunique()),
        "normal_subjects": int(
            predictions[["subject_id", "target"]]
            .drop_duplicates()["target"]
            .eq(0)
            .sum()
        ),
        "impaired_subjects": int(
            predictions[["subject_id", "target"]]
            .drop_duplicates()["target"]
            .eq(1)
            .sum()
        ),
        "source_segments": int(pd.read_csv(TABLE_PATH).shape[0]),
        "aggregation_for_ml": (
            "median of all available 20s segment-level final3 rows per subject"
        ),
        "sequence_for_dl": (
            "up to 100 chronological 20s segment-level final3 rows per subject"
        ),
        "outer_cv": "subject-level StratifiedKFold(5) x 100 repeats",
        "inner_threshold_cv": (
            "3-fold OOF within each outer-training fold; sensitivity >= 0.80 "
            "then maximum specificity"
        ),
        "preprocessing": "fit within each training fold only",
        "reporting_plots": (
            "subject consensus: one row per subject; ROC uses mean OOF "
            "probability over 100 repeats and confusion matrix uses majority "
            "vote of training-only-threshold predictions"
        ),
        "deployed_model_modified": False,
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(engine.ALL_MODELS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute requested models even when cached results exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(engine.ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    configure_engine()
    assignments, x_agg, x_seq = build_subject_cache()
    print(
        f"subjects={assignments['subject_id'].nunique()} "
        f"rows={len(pd.read_csv(TABLE_PATH))} device={engine.DEVICE}",
        flush=True,
    )
    for model_name in requested:
        metric_path = engine.MODEL_RESULT_DIR / f"{model_name}_metrics.csv"
        prediction_path = (
            engine.MODEL_RESULT_DIR / f"{model_name}_predictions.csv"
        )
        if metric_path.exists() and prediction_path.exists() and not args.force:
            print(f"[reuse] {model_name}", flush=True)
            continue
        engine.evaluate_model(
            model_name, assignments, x_agg, x_seq
        )
        print(finalize().to_string(index=False), flush=True)

    print("\nCompleted summary")
    print(finalize().to_string(index=False), flush=True)
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
