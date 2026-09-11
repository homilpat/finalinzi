from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_clinical_domain_feature_group_modeling import make_binary_targets
from run_clinical_motor_label_modeling import DEFAULT_CLINICAL_XLSX, load_clinical, make_motor_score
from run_labwalks_service20_ml_modeling import TARGET, decision_scores, sample_one_window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CSV = (
    PROJECT_ROOT
    / "physionet_AWS"
    / "strict_preprocessing_runs"
    / "labwalks_service_window_features"
    / "service10"
    / "labwalks_service10_amp_spec_features.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "physionet_AWS"
    / "strict_preprocessing_runs"
    / "service10_domain_representative_model_compare"
)

THRESHOLD_STRATEGY = "train_sens80_maxspec"


DOMAIN_FEATURES: dict[str, list[str]] = {
    "vertical_vitality": [
        "v_amp_pool_median",
        "v_amp_pool_p90",
        "v_amp_pool_iqr",
    ],
    "lateral_stability": [
        "ml_amp_pool_iqr",
        "ml_amp_pool_p90",
        "ml_amp_pool_soft_rms",
    ],
    "rhythm_regularity": [
        "base_step_duration",
        "base_stride_duration",
        "base_v_stride_regularity",
        "base_ml_stride_regularity",
        "v_amp_pool_step_regularity",
        "v_amp_pool_stride_regularity",
    ],
    "trunk_rotation": [
        "roll_amp_pool_iqr",
        "roll_amp_pool_p90",
        "roll_spec_entropy",
        "roll_spec_peak_freq",
    ],
    "movement_complexity": [
        "v_spec_entropy",
        "ml_spec_entropy",
        "roll_spec_entropy",
        "v_spec_peak_freq",
    ],
}


FIXED_DOMAIN_FEATURE = {
    "vertical_vitality": "v_amp_pool_median",
    "lateral_stability": "ml_amp_pool_iqr",
    "rhythm_regularity": "base_v_stride_regularity",
    "trunk_rotation": "roll_amp_pool_iqr",
    "movement_complexity": "v_spec_entropy",
}


@dataclass(frozen=True)
class Strategy:
    name: str
    domains: tuple[str, ...]
    selection_mode: str  # fixed | train_best


STRATEGIES = [
    Strategy("single_vertical_median", ("vertical_vitality",), "fixed"),
    Strategy("domain3_fixed", ("vertical_vitality", "lateral_stability", "rhythm_regularity"), "fixed"),
    Strategy("domain4_fixed", ("vertical_vitality", "lateral_stability", "rhythm_regularity", "trunk_rotation"), "fixed"),
    Strategy(
        "domain5_fixed",
        ("vertical_vitality", "lateral_stability", "rhythm_regularity", "trunk_rotation", "movement_complexity"),
        "fixed",
    ),
    Strategy("domain3_trainbest", ("vertical_vitality", "lateral_stability", "rhythm_regularity"), "train_best"),
    Strategy("domain4_trainbest", ("vertical_vitality", "lateral_stability", "rhythm_regularity", "trunk_rotation"), "train_best"),
    Strategy(
        "domain5_trainbest",
        ("vertical_vitality", "lateral_stability", "rhythm_regularity", "trunk_rotation", "movement_complexity"),
        "train_best",
    ),
]


def clinical_labels(clinical_xlsx: Path) -> pd.DataFrame:
    clinical = load_clinical(clinical_xlsx)
    clinical["motor_impairment_score"] = make_motor_score(clinical)
    binary = make_binary_targets(clinical).drop(columns=["subject_id"])
    return pd.concat([clinical[["subject_id", "DGI", "TUG", "motor_impairment_score"]], binary], axis=1)


def prepare_data(feature_csv: Path, clinical_xlsx: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = pd.read_csv(feature_csv)
    features["subject_id"] = features["subject_id"].astype(str)
    labels = clinical_labels(clinical_xlsx)
    merged = features.merge(labels, on="subject_id", how="inner")
    merged = merged[merged[TARGET].notna()].copy()
    merged["target"] = merged[TARGET].astype(int)
    subject_table = merged[["subject_id", "target"]].drop_duplicates().sort_values("subject_id")
    return merged, subject_table["subject_id"].to_numpy(), subject_table["target"].to_numpy()


def make_pipeline(c_value: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def feature_target_score(train_df: pd.DataFrame, feature: str, rank_mode: str) -> float:
    x = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return -np.inf
    if rank_mode == "binary_auc":
        y = train_df["target"].to_numpy()
        if len(np.unique(y[mask])) < 2:
            return -np.inf
        try:
            score = roc_auc_score(y[mask], x[mask])
        except ValueError:
            return -np.inf
        return abs(float(score) - 0.5)
    if rank_mode == "motor_score_spearman":
        y = pd.to_numeric(train_df["motor_impairment_score"], errors="coerce").to_numpy(dtype=float)
        valid = mask & np.isfinite(y)
        if valid.sum() < 4:
            return -np.inf
        corr = pd.Series(x[valid]).corr(pd.Series(y[valid]), method="spearman")
        return -np.inf if pd.isna(corr) else abs(float(corr))
    if rank_mode == "combined":
        return max(feature_target_score(train_df, feature, "binary_auc"), feature_target_score(train_df, feature, "motor_score_spearman"))
    raise ValueError(f"Unknown rank_mode: {rank_mode}")


def abs_spearman(df: pd.DataFrame, a: str, b: str) -> float:
    corr = df[[a, b]].corr(method="spearman").iloc[0, 1]
    return 1.0 if pd.isna(corr) else abs(float(corr))


def select_domain_features(train_df: pd.DataFrame, strategy: Strategy, rank_mode: str, corr_limit: float) -> list[str]:
    selected: list[str] = []
    for domain in strategy.domains:
        candidates = [f for f in DOMAIN_FEATURES[domain] if f in train_df.columns]
        if strategy.selection_mode == "fixed":
            feature = FIXED_DOMAIN_FEATURE[domain]
            if feature not in train_df.columns:
                raise ValueError(f"Missing fixed feature {feature}")
        else:
            scored = sorted(
                [(f, feature_target_score(train_df, f, rank_mode)) for f in candidates],
                key=lambda item: item[1],
                reverse=True,
            )
            feature = None
            for candidate, score in scored:
                if not np.isfinite(score):
                    continue
                if all(abs_spearman(train_df, candidate, existing) < corr_limit for existing in selected):
                    feature = candidate
                    break
            if feature is None:
                # Keep one representative per domain even if cross-domain correlation is high;
                # this is an explanation-oriented model comparison.
                feature = scored[0][0]
        selected.append(feature)
    return selected


def threshold_from_train(y_train: np.ndarray, train_prob: np.ndarray) -> float:
    candidates = np.unique(train_prob)
    best_threshold = float(candidates[0])
    best_spec = -np.inf
    best_sens = -np.inf
    for threshold in candidates:
        pred = (train_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if sensitivity >= 0.80 and (specificity > best_spec or (specificity == best_spec and sensitivity > best_sens)):
            best_threshold = float(threshold)
            best_spec = specificity
            best_sens = sensitivity
    return best_threshold


def clf_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def eval_fold(
    merged: pd.DataFrame,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategy: Strategy,
    scheme: str,
    repeat: int,
    fold: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    train_subjects = set(subject_ids[train_idx])
    test_subjects = set(subject_ids[test_idx])
    seed_base = repeat * 1000 + fold
    train_one = sample_one_window(merged, train_subjects, seed_base, "quality")
    test_one = sample_one_window(merged, test_subjects, seed_base + 500000, "quality")
    selected = select_domain_features(train_one, strategy, args.rank_mode, args.corr_limit)

    X_train = train_one[selected].replace([np.inf, -np.inf], np.nan)
    X_test = test_one[selected].replace([np.inf, -np.inf], np.nan)
    y_train = train_one["target"].to_numpy()
    y_test = test_one["target"].to_numpy()

    model = make_pipeline(c_value=args.C, random_state=20260709 + seed_base)
    model.fit(X_train, y_train)
    train_prob = decision_scores(model, X_train)
    test_prob = decision_scores(model, X_test)
    threshold = threshold_from_train(y_train, train_prob)

    metric_rows = []
    prediction_rows = []
    for split, frame, y, prob in [("train", train_one, y_train, train_prob), ("test", test_one, y_test, test_prob)]:
        row = clf_metrics(y, prob, threshold)
        row.update(
            {
                "strategy": strategy.name,
                "scheme": scheme,
                "repeat": repeat,
                "fold": fold,
                "split": split,
                "threshold_strategy": THRESHOLD_STRATEGY,
                "threshold": threshold,
                "features_used": "|".join(selected),
                "n_features": len(selected),
                "C": args.C,
                "rank_mode": args.rank_mode,
                "corr_limit": args.corr_limit,
            }
        )
        metric_rows.append(row)
        if split == "test":
            pred = (prob >= threshold).astype(int)
            for sid, true_value, p_value, pred_value in zip(frame["subject_id"], y, prob, pred):
                prediction_rows.append(
                    {
                        "strategy": strategy.name,
                        "scheme": scheme,
                        "repeat": repeat,
                        "fold": fold,
                        "subject_id": sid,
                        "target": int(true_value),
                        "probability": float(p_value),
                        "threshold": float(threshold),
                        "prediction": int(pred_value),
                        "features_used": "|".join(selected),
                    }
                )
    return metric_rows, prediction_rows, {
        "strategy": strategy.name,
        "scheme": scheme,
        "repeat": repeat,
        "fold": fold,
        "features_used": "|".join(selected),
        "n_features": len(selected),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "accuracy", "recall_sensitivity", "specificity", "f1"]
    summary = (
        metrics.groupby(["strategy", "scheme", "split"], dropna=False)[metric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    counts = metrics.groupby(["strategy", "scheme", "split"], dropna=False)[["n_positive", "n_negative"]].mean().reset_index()
    return summary.merge(counts, on=["strategy", "scheme", "split"], how="left")


def pooled_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, scheme), df in predictions.groupby(["strategy", "scheme"], dropna=False):
        pooled = (
            df.groupby("subject_id", as_index=False)
            .agg(
                target=("target", "first"),
                probability=("probability", "mean"),
                threshold=("threshold", "mean"),
                n_predictions=("probability", "size"),
            )
        )
        y = pooled["target"].to_numpy()
        prob = pooled["probability"].to_numpy()
        pred = (prob >= pooled["threshold"].to_numpy()).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "strategy": strategy,
                "scheme": scheme,
                "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
                "accuracy": accuracy_score(y, pred),
                "recall_sensitivity": recall_score(y, pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "f1": f1_score(y, pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "n_positive": int(np.sum(y == 1)),
                "n_negative": int(np.sum(y == 0)),
                "pooled_decision": "mean_prob_vs_mean_threshold",
                "n_subjects": int(len(pooled)),
                "mean_predictions_per_subject": float(pooled["n_predictions"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_repeated(merged: pd.DataFrame, subject_ids: np.ndarray, y_subject: np.ndarray, args: argparse.Namespace):
    metric_rows = []
    prediction_rows = []
    selection_rows = []
    for strategy in STRATEGIES:
        print(f"strategy={strategy.name}")
        for repeat in range(args.n_repeats):
            cv5 = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=420000 + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv5.split(subject_ids, y_subject)):
                m, p, s = eval_fold(merged, subject_ids, train_idx, test_idx, strategy, "A_5fold_x100", repeat, fold, args)
                metric_rows.extend(m)
                prediction_rows.extend(p)
                selection_rows.append(s)
            cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=430000 + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv3.split(subject_ids, y_subject)):
                m, p, s = eval_fold(merged, subject_ids, train_idx, test_idx, strategy, "B_3fold_x100", repeat, fold, args)
                metric_rows.extend(m)
                prediction_rows.extend(p)
                selection_rows.append(s)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=440000 + repeat)
            train_idx, test_idx = next(sss.split(subject_ids, y_subject))
            m, p, s = eval_fold(merged, subject_ids, train_idx, test_idx, strategy, "C_repeated_80_20_x100", repeat, 0, args)
            metric_rows.extend(m)
            prediction_rows.extend(p)
            selection_rows.append(s)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def run_loso(merged: pd.DataFrame, subject_ids: np.ndarray, args: argparse.Namespace):
    metric_rows = []
    prediction_rows = []
    selection_rows = []
    loo = LeaveOneOut()
    for strategy in STRATEGIES:
        print(f"LOSO strategy={strategy.name}")
        for fold, (train_idx, test_idx) in enumerate(loo.split(subject_ids)):
            m, p, s = eval_fold(merged, subject_ids, train_idx, test_idx, strategy, "E_LOSO_pooled", 0, fold, args)
            metric_rows.extend(m)
            prediction_rows.extend(p)
            selection_rows.append(s)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL_XLSX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.05)
    parser.add_argument("--corr-limit", type=float, default=0.80)
    parser.add_argument("--rank-mode", choices=["binary_auc", "motor_score_spearman", "combined"], default="binary_auc")
    parser.add_argument("--skip-loso", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged, subject_ids, y_subject = prepare_data(args.feature_csv, args.clinical_xlsx)
    metrics, preds, selections = run_repeated(merged, subject_ids, y_subject, args)
    if not args.skip_loso:
        loso_m, loso_p, loso_s = run_loso(merged, subject_ids, args)
        metrics = pd.concat([metrics, loso_m], ignore_index=True)
        preds = pd.concat([preds, loso_p], ignore_index=True)
        selections = pd.concat([selections, loso_s], ignore_index=True)

    summary = summarize_metrics(metrics)
    pooled = pooled_subject_metrics(preds)
    feature_counts = (
        selections.groupby(["strategy", "features_used"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["strategy", "count"], ascending=[True, False])
    )

    metrics.to_csv(args.out_dir / "domain_model_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(args.out_dir / "domain_model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(args.out_dir / "domain_model_selected_features_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "domain_model_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(args.out_dir / "domain_model_pooled_subject_metrics.csv", index=False, encoding="utf-8-sig")
    feature_counts.to_csv(args.out_dir / "domain_model_feature_selection_counts.csv", index=False, encoding="utf-8-sig")

    config = {
        "target": TARGET,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "rank_mode": args.rank_mode,
        "corr_limit": args.corr_limit,
        "domain_features": DOMAIN_FEATURES,
        "fixed_domain_feature": FIXED_DOMAIN_FEATURE,
        "strategies": [s.__dict__ for s in STRATEGIES],
        "n_repeats": args.n_repeats,
        "n_splits": args.n_splits,
        "C": args.C,
        "n_subjects": int(len(subject_ids)),
        "class_counts": {"0": int(np.sum(y_subject == 0)), "1": int(np.sum(y_subject == 1))},
        "leakage_control": [
            "Subjects are split before window sampling.",
            "Train-best domain feature selection uses train fold only.",
            "Cross-domain correlation check uses train fold only.",
            "Imputer, scaler, LogisticRegression, and threshold are fit on train fold only.",
            "Test fold is used only for transform and evaluation.",
        ],
    }
    (args.out_dir / "domain_model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPooled subject-level comparison:")
    print(
        pooled[
            [
                "strategy",
                "scheme",
                "roc_auc",
                "recall_sensitivity",
                "specificity",
                "accuracy",
                "f1",
                "tp",
                "fn",
                "tn",
                "fp",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
