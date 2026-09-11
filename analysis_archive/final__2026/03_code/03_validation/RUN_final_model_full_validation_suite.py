from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_clinical_domain_feature_group_modeling import make_binary_targets
from run_clinical_motor_label_modeling import DEFAULT_CLINICAL_XLSX, load_clinical, make_motor_score
from run_labwalks_service20_ml_modeling import TARGET, decision_scores, feature_sets, sample_one_window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CSV = (
    PROJECT_ROOT
    / "physionet_AWS"
    / "strict_preprocessing_runs"
    / "labwalks_service_window_features"
    / "service10"
    / "labwalks_service10_amp_spec_features.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "physionet_AWS" / "strict_preprocessing_runs" / "final_service10_motor_model_full_validation"


THRESHOLD_STRATEGIES = [
    "sens75_train_positive_p25",
    "sens80_train_positive_p20",
    "sens85_train_positive_p15",
    "spec80_train_negative_p80",
    "youden_train",
    "train_sens75_maxspec",
    "train_sens80_maxspec",
]


def clinical_labels(clinical_xlsx: Path) -> pd.DataFrame:
    clinical = load_clinical(clinical_xlsx)
    clinical["motor_impairment_score"] = make_motor_score(clinical)
    binary = make_binary_targets(clinical).drop(columns=["subject_id"])
    return pd.concat([clinical[["subject_id", "DGI", "TUG"]], binary], axis=1)


def make_pipeline(k: int, c_value: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
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


def threshold_candidates(y_train: np.ndarray, train_prob: np.ndarray) -> dict[str, float]:
    negative_prob = train_prob[y_train == 0]
    positive_prob = train_prob[y_train == 1]
    thresholds: dict[str, float] = {
        "sens75_train_positive_p25": float(np.quantile(positive_prob, 0.25)),
        "sens80_train_positive_p20": float(np.quantile(positive_prob, 0.20)),
        "sens85_train_positive_p15": float(np.quantile(positive_prob, 0.15)),
        "spec80_train_negative_p80": float(np.quantile(negative_prob, 0.80)),
    }

    candidates = np.unique(train_prob)
    best_youden = float(candidates[0])
    best_youden_score = -np.inf
    best_sens75 = float(candidates[0])
    best_sens75_spec = -np.inf
    best_sens80 = float(candidates[0])
    best_sens80_spec = -np.inf

    for threshold in candidates:
        pred = (train_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        youden = sensitivity + specificity - 1.0
        if youden > best_youden_score:
            best_youden_score = youden
            best_youden = float(threshold)
        if sensitivity >= 0.75 and specificity > best_sens75_spec:
            best_sens75_spec = specificity
            best_sens75 = float(threshold)
        if sensitivity >= 0.80 and specificity > best_sens80_spec:
            best_sens80_spec = specificity
            best_sens80 = float(threshold)

    thresholds["youden_train"] = best_youden
    thresholds["train_sens75_maxspec"] = best_sens75
    thresholds["train_sens80_maxspec"] = best_sens80
    return thresholds


def clf_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
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


def prepare_data(feature_csv: Path, clinical_xlsx: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    features = pd.read_csv(feature_csv)
    features["subject_id"] = features["subject_id"].astype(str)
    labels = clinical_labels(clinical_xlsx)
    merged = features.merge(labels, on="subject_id", how="inner")
    merged = merged[merged[TARGET].notna()].copy()
    merged["target"] = merged[TARGET].astype(int)

    id_cols = {"record", "subject_id", "group", "segment_idx", "start_sec", "end_sec", "window_sec"}
    numeric_cols = [
        c
        for c in merged.columns
        if c not in id_cols
        and c not in {"DGI", "TUG", TARGET, "target"}
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    feature_cols = feature_sets(numeric_cols)["all_combined"]
    subject_table = merged[["subject_id", "target"]].drop_duplicates().reset_index(drop=True)
    return merged, subject_table["subject_id"].to_numpy(), subject_table["target"].to_numpy(), feature_cols, subject_table


def fit_predict_fold(
    merged: pd.DataFrame,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    repeat: int,
    fold: int,
    scheme: str,
    k: int,
    c_value: float,
) -> tuple[list[dict], list[dict]]:
    train_subjects = set(subject_ids[train_idx])
    test_subjects = set(subject_ids[test_idx])
    train_one = sample_one_window(merged, train_subjects, repeat * 100 + fold, "quality")
    test_one = sample_one_window(merged, test_subjects, repeat * 100 + fold + 10000, "quality")
    X_train = train_one[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_one[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_one["target"].to_numpy()
    y_test = test_one["target"].to_numpy()

    model = make_pipeline(k, c_value, random_state=82000 + repeat * 100 + fold)
    model.fit(X_train, y_train)
    train_prob = decision_scores(model, X_train)
    test_prob = decision_scores(model, X_test)
    thresholds = threshold_candidates(y_train, train_prob)

    metric_rows = []
    prediction_rows = []
    for strategy, threshold in thresholds.items():
        for split, subject_frame, y, prob in [
            ("train", train_one, y_train, train_prob),
            ("test", test_one, y_test, test_prob),
        ]:
            pred = (prob >= threshold).astype(int)
            row = clf_metrics(y, prob, pred)
            row.update(
                {
                    "scheme": scheme,
                    "repeat": repeat,
                    "fold": fold,
                    "split": split,
                    "threshold_strategy": strategy,
                    "threshold": float(threshold),
                    "k": k,
                    "C": c_value,
                }
            )
            metric_rows.append(row)
            if split == "test":
                for sid, true_value, p_value, pred_value in zip(subject_frame["subject_id"], y, prob, pred):
                    prediction_rows.append(
                        {
                            "scheme": scheme,
                            "repeat": repeat,
                            "fold": fold,
                            "subject_id": sid,
                            "target": int(true_value),
                            "threshold_strategy": strategy,
                            "probability": float(p_value),
                            "threshold": float(threshold),
                            "prediction": int(pred_value),
                        }
                    )
    return metric_rows, prediction_rows


def summarize_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["scheme", "threshold_strategy", "split"], dropna=False)[
            ["roc_auc", "accuracy", "recall_sensitivity", "specificity", "f1", "n_positive", "n_negative"]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in summary.columns]
    return summary


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_ci_table(metrics: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260709)
    rows = []
    for (scheme, strategy, split), group in metrics.groupby(["scheme", "threshold_strategy", "split"], dropna=False):
        for metric in ["roc_auc", "accuracy", "recall_sensitivity", "specificity", "f1"]:
            lo, hi = bootstrap_ci(group[metric].to_numpy(dtype=float), rng, n_boot=n_boot)
            rows.append(
                {
                    "scheme": scheme,
                    "threshold_strategy": strategy,
                    "split": split,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_folds": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def pooled_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, strategy), group in predictions.groupby(["scheme", "threshold_strategy"], dropna=False):
        pooled = (
            group.groupby(["subject_id", "target"], as_index=False)
            .agg(
                mean_probability=("probability", "mean"),
                mean_threshold=("threshold", "mean"),
                positive_vote_rate=("prediction", "mean"),
                n_predictions=("prediction", "size"),
            )
        )
        for decision_name, pred in [
            ("majority_vote", (pooled["positive_vote_rate"] >= 0.5).astype(int).to_numpy()),
            ("mean_prob_vs_mean_threshold", (pooled["mean_probability"] >= pooled["mean_threshold"]).astype(int).to_numpy()),
        ]:
            prob = pooled["mean_probability"].to_numpy()
            y = pooled["target"].to_numpy()
            row = clf_metrics(y, prob, pred)
            row.update(
                {
                    "scheme": scheme,
                    "threshold_strategy": strategy,
                    "pooled_decision": decision_name,
                    "n_subjects": int(len(pooled)),
                    "mean_predictions_per_subject": float(pooled["n_predictions"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def subject_bootstrap_ci_for_pooled(predictions: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260710)
    rows = []
    for (scheme, strategy), group in predictions.groupby(["scheme", "threshold_strategy"], dropna=False):
        pooled = (
            group.groupby(["subject_id", "target"], as_index=False)
            .agg(
                mean_probability=("probability", "mean"),
                mean_threshold=("threshold", "mean"),
                positive_vote_rate=("prediction", "mean"),
            )
        )
        for decision_name, pred_base in [
            ("majority_vote", (pooled["positive_vote_rate"] >= 0.5).astype(int).to_numpy()),
            ("mean_prob_vs_mean_threshold", (pooled["mean_probability"] >= pooled["mean_threshold"]).astype(int).to_numpy()),
        ]:
            y_base = pooled["target"].to_numpy()
            prob_base = pooled["mean_probability"].to_numpy()
            boot_values = {m: [] for m in ["roc_auc", "accuracy", "recall_sensitivity", "specificity", "f1"]}
            for _ in range(n_boot):
                idx = rng.choice(np.arange(len(pooled)), size=len(pooled), replace=True)
                if len(np.unique(y_base[idx])) < 2:
                    continue
                m = clf_metrics(y_base[idx], prob_base[idx], pred_base[idx])
                for metric in boot_values:
                    boot_values[metric].append(m[metric])
            for metric, values in boot_values.items():
                arr = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "scheme": scheme,
                        "threshold_strategy": strategy,
                        "pooled_decision": decision_name,
                        "metric": metric,
                        "mean": float(np.mean(arr)) if len(arr) else np.nan,
                        "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
                        "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
                        "n_boot_valid": int(len(arr)),
                    }
                )
    return pd.DataFrame(rows)


def run_repeated_cv(
    merged: pd.DataFrame,
    subject_ids: np.ndarray,
    subject_y: np.ndarray,
    subject_table: pd.DataFrame,
    feature_cols: list[str],
    n_repeats: int,
    k: int,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    for repeat in range(n_repeats):
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=91000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv5.split(subject_table, subject_y)):
            m, p = fit_predict_fold(merged, subject_ids, train_idx, test_idx, feature_cols, repeat, fold, "A_5fold_x100", k, c_value)
            metrics.extend(m)
            predictions.extend(p)

        cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=92000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv3.split(subject_table, subject_y)):
            m, p = fit_predict_fold(merged, subject_ids, train_idx, test_idx, feature_cols, repeat, fold, "B_3fold_x100", k, c_value)
            metrics.extend(m)
            predictions.extend(p)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=93000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(sss.split(subject_table, subject_y)):
            m, p = fit_predict_fold(merged, subject_ids, train_idx, test_idx, feature_cols, repeat, fold, "C_repeated_80_20_x100", k, c_value)
            metrics.extend(m)
            predictions.extend(p)
        print(f"completed repeated validation repeat={repeat}")
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def run_loso(
    merged: pd.DataFrame,
    subject_ids: np.ndarray,
    subject_y: np.ndarray,
    subject_table: pd.DataFrame,
    feature_cols: list[str],
    k: int,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    loo = LeaveOneOut()
    for fold, (train_idx, test_idx) in enumerate(loo.split(subject_table, subject_y)):
        m, p = fit_predict_fold(merged, subject_ids, train_idx, test_idx, feature_cols, repeat=0, fold=fold, scheme="E_LOSO_pooled", k=k, c_value=c_value)
        # LOSO fold-level sensitivity/specificity is not meaningful for one-person tests,
        # but keeping rows makes the prediction audit traceable.
        metrics.extend(m)
        predictions.extend(p)
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL_XLSX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged, subject_ids, subject_y, feature_cols, subject_table = prepare_data(args.feature_csv, args.clinical_xlsx)

    repeated_metrics, repeated_predictions = run_repeated_cv(
        merged,
        subject_ids,
        subject_y,
        subject_table,
        feature_cols,
        n_repeats=args.n_repeats,
        k=args.k,
        c_value=args.C,
    )
    loso_metrics, loso_predictions = run_loso(merged, subject_ids, subject_y, subject_table, feature_cols, args.k, args.C)

    all_metrics = pd.concat([repeated_metrics, loso_metrics], ignore_index=True)
    all_predictions = pd.concat([repeated_predictions, loso_predictions], ignore_index=True)
    all_metrics.to_csv(args.out_dir / "A_to_E_fold_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(args.out_dir / "A_to_E_oof_predictions_by_subject.csv", index=False, encoding="utf-8-sig")

    fold_summary = summarize_fold_metrics(all_metrics[all_metrics["scheme"] != "E_LOSO_pooled"])
    fold_summary.to_csv(args.out_dir / "A_B_C_fold_metrics_summary.csv", index=False, encoding="utf-8-sig")
    fold_ci = make_ci_table(all_metrics[all_metrics["scheme"] != "E_LOSO_pooled"], args.bootstrap)
    fold_ci.to_csv(args.out_dir / "A_B_C_fold_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    pooled = pooled_subject_metrics(all_predictions)
    pooled.to_csv(args.out_dir / "D_E_pooled_subject_metrics.csv", index=False, encoding="utf-8-sig")
    pooled_ci = subject_bootstrap_ci_for_pooled(all_predictions, args.bootstrap)
    pooled_ci.to_csv(args.out_dir / "D_E_pooled_subject_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "purpose": "A-E final validation suite for the fixed service motor model.",
        "A": "5-fold x 100 repeated stratified CV",
        "B": "3-fold x 100 repeated stratified CV",
        "C": "Repeated stratified shuffle split 8:2 x 100",
        "D": "Pooled subject-level OOF evaluation from repeated CV predictions",
        "E": "LOSO pooled subject-level evaluation",
        "model": "10 sec all_combined Logistic Regression",
        "target": "DGI <= 19 OR TUG >= 12",
        "k": args.k,
        "C": args.C,
        "class_weight": "balanced",
        "threshold_strategies": THRESHOLD_STRATEGIES,
        "n_repeats": args.n_repeats,
        "bootstrap": args.bootstrap,
        "n_subjects": int(len(subject_table)),
        "class_counts": {str(k): int(v) for k, v in pd.Series(subject_y).value_counts().sort_index().items()},
        "feature_csv": str(args.feature_csv),
        "clinical_xlsx": str(args.clinical_xlsx),
    }
    (args.out_dir / "A_to_E_validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    focus_fold = fold_summary[
        (fold_summary["split"] == "test")
        & (fold_summary["threshold_strategy"].isin(["train_sens80_maxspec", "train_sens75_maxspec", "youden_train"]))
    ]
    focus_pooled = pooled[
        pooled["threshold_strategy"].isin(["train_sens80_maxspec", "train_sens75_maxspec", "youden_train"])
    ]
    print("\nA/B/C fold-level summary:")
    print(focus_fold.to_string(index=False))
    print("\nD/E pooled subject-level summary:")
    print(focus_pooled.to_string(index=False))
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
