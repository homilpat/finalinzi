from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
FEATURE_CSV = ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
CLINICAL_XLSX = ROOT / "final__2026" / "04_clinical_data" / "ClinicalDemogData_COFL.xlsx"
OUT_DIR = ROOT / "final__2026" / "10_corr_pruned_feature_search"

TARGET = "DGI_le19_or_TUG_ge12"
EXCLUDED_SUBJECTS = {"CO024", "FL020"}
FINAL_FIXED_FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]
N_REPEATS = 100
N_JOBS = 1
RANDOM_SEED = 20260713


CONFIGS = [
    {"name": "fixed_domain4_current", "mode": "fixed", "features": FINAL_FIXED_FEATURES, "k": 4, "corr_limit": None},
    {"name": "corrpruned_k4_rho075", "mode": "search", "k": 4, "corr_limit": 0.75},
    {"name": "corrpruned_k4_rho080", "mode": "search", "k": 4, "corr_limit": 0.80},
    {"name": "corrpruned_k5_rho075", "mode": "search", "k": 5, "corr_limit": 0.75},
    {"name": "corrpruned_k5_rho080", "mode": "search", "k": 5, "corr_limit": 0.80},
]


def normalize_subject_id(value: object) -> str:
    return str(value).strip().replace("-", "").upper()


def load_labels() -> pd.DataFrame:
    frames = []
    for sheet_name in ["Controls", "Fallers"]:
        frame = pd.read_excel(CLINICAL_XLSX, sheet_name=sheet_name)
        frame["subject_id"] = frame["#"].map(normalize_subject_id)
        frames.append(frame[["subject_id", "DGI", "TUG"]])
    labels = pd.concat(frames, ignore_index=True)
    labels["DGI"] = pd.to_numeric(labels["DGI"], errors="coerce")
    labels["TUG"] = pd.to_numeric(labels["TUG"], errors="coerce")
    labels[TARGET] = ((labels["DGI"] <= 19) | (labels["TUG"] >= 12)).astype("Int64")
    return labels.dropna(subset=[TARGET]).copy()


def load_subject_table() -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(FEATURE_CSV)
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    labels = load_labels()
    merged = features.merge(labels, on="subject_id", how="inner")
    merged = merged[~merged["subject_id"].isin(EXCLUDED_SUBJECTS)].copy()
    merged = merged[merged[TARGET].notna()].copy()

    excluded_cols = {"segment_idx", "start_sec", "end_sec", "window_sec", "DGI", "TUG", TARGET}
    numeric_cols = []
    for col in merged.columns:
        if col in excluded_cols or col in {"subject_id", "record", "group", "base_feature_status"}:
            continue
        if pd.api.types.is_numeric_dtype(merged[col]):
            numeric_cols.append(col)

    rows = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "target": int(group[TARGET].iloc[0])}
        for col in numeric_cols:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").median())
        rows.append(row)
    table = pd.DataFrame(rows)

    candidate_features = []
    for col in numeric_cols:
        valid = table[col].replace([np.inf, -np.inf], np.nan).notna().sum()
        unique = table[col].replace([np.inf, -np.inf], np.nan).nunique(dropna=True)
        if valid >= 50 and unique >= 5:
            candidate_features.append(col)
    table = table.dropna(subset=FINAL_FIXED_FEATURES, how="any").reset_index(drop=True)
    return table, candidate_features


def make_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def univariate_auc_rank(train: pd.DataFrame, features: list[str]) -> list[tuple[str, float]]:
    y = train["target"].to_numpy(dtype=int)
    ranked = []
    for feature in features:
        x = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = x.notna().to_numpy()
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        try:
            score = abs(roc_auc_score(y[mask], x[mask].to_numpy(dtype=float)) - 0.5)
        except ValueError:
            continue
        if np.isfinite(score):
            ranked.append((feature, float(score)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def abs_spearman(train: pd.DataFrame, a: str, b: str) -> float:
    corr = train[[a, b]].replace([np.inf, -np.inf], np.nan).corr(method="spearman").iloc[0, 1]
    return 1.0 if pd.isna(corr) else abs(float(corr))


def select_corr_pruned_features(train: pd.DataFrame, candidates: list[str], k: int, corr_limit: float) -> list[str]:
    selected = []
    for feature, _ in univariate_auc_rank(train, candidates):
        if all(abs_spearman(train, feature, existing) < corr_limit for existing in selected):
            selected.append(feature)
        if len(selected) >= k:
            break
    return selected


def choose_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_threshold = float(np.min(prob))
    best_score = -np.inf
    best_sens = -np.inf
    for threshold in np.unique(prob):
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        score = sens + spec - 1.0
        if score > best_score or (score == best_score and sens > best_sens):
            best_score = score
            best_sens = sens
            best_threshold = float(threshold)
    return best_threshold


def inner_threshold(table: pd.DataFrame, train_idx: np.ndarray, features: list[str], repeat: int, fold: int) -> float:
    train = table.iloc[train_idx].reset_index(drop=True)
    y = train["target"].to_numpy(dtype=int)
    n_splits = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=510000 + repeat * 100 + fold)
    prob = np.full(len(train), np.nan)
    for inner_fold, (tr, va) in enumerate(cv.split(train[features], y)):
        model = make_pipeline(610000 + repeat * 1000 + fold * 10 + inner_fold)
        model.fit(train.iloc[tr][features], y[tr])
        prob[va] = model.predict_proba(train.iloc[va][features])[:, 1]
    return choose_youden_threshold(y, prob)


def fold_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_one_fold(table: pd.DataFrame, candidates: list[str], config: dict, repeat: int, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict, list[dict]]:
    train = table.iloc[train_idx]
    if config["mode"] == "fixed":
        selected = list(config["features"])
    else:
        selected = select_corr_pruned_features(train, candidates, int(config["k"]), float(config["corr_limit"]))

    threshold = inner_threshold(table, train_idx, selected, repeat, fold)
    model = make_pipeline(710000 + repeat * 1000 + fold)
    y_train = table.iloc[train_idx]["target"].to_numpy(dtype=int)
    y_test = table.iloc[test_idx]["target"].to_numpy(dtype=int)
    model.fit(table.iloc[train_idx][selected], y_train)
    prob = model.predict_proba(table.iloc[test_idx][selected])[:, 1]
    metric = fold_metrics(y_test, prob, threshold)
    metric.update(
        {
            "config": config["name"],
            "repeat": repeat,
            "fold": fold,
            "threshold": threshold,
            "features_used": "|".join(selected),
            "n_features": len(selected),
            "n_test": int(len(test_idx)),
            "positive_test": int(y_test.sum()),
            "negative_test": int((1 - y_test).sum()),
        }
    )
    preds = []
    for subject_id, target, p in zip(table.iloc[test_idx]["subject_id"], y_test, prob):
        preds.append(
            {
                "config": config["name"],
                "repeat": repeat,
                "fold": fold,
                "subject_id": subject_id,
                "target": int(target),
                "probability": float(p),
                "threshold": float(threshold),
                "prediction": int(p >= threshold),
                "features_used": "|".join(selected),
            }
        )
    return metric, preds


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in pred.groupby("config", sort=True):
        y = group["target"].to_numpy(dtype=int)
        prob = group["probability"].to_numpy(dtype=float)
        # Fold thresholds differ, so use stored predictions for threshold metrics.
        pred_y = group["prediction"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred_y, labels=[0, 1]).ravel()
        rows.append(
            {
                "config": config,
                "n_predictions": int(len(group)),
                "n_subjects_per_repeat": int(group.groupby("repeat")["subject_id"].nunique().median()),
                "n_repeats": int(group["repeat"].nunique()),
                "auc": float(roc_auc_score(y, prob)),
                "accuracy": float(accuracy_score(y, pred_y)),
                "sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
                "f1": float(f1_score(y, pred_y)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def selection_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in metrics.groupby("config", sort=True):
        counter = Counter()
        for feature_string in group["features_used"]:
            counter.update(feature_string.split("|"))
        total_folds = group.shape[0]
        for feature, count in counter.most_common():
            rows.append({"config": config, "feature": feature, "selected_count": count, "selection_rate": count / total_folds})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, candidates = load_subject_table()
    y = table["target"].to_numpy(dtype=int)

    tasks = []
    for config in CONFIGS:
        for repeat in range(N_REPEATS):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv.split(table, y)):
                tasks.append((config, repeat, fold, train_idx, test_idx))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if N_JOBS == 1:
            results = []
            for i, (config, repeat, fold, train_idx, test_idx) in enumerate(tasks, start=1):
                if i % 100 == 0:
                    print(f"completed {i}/{len(tasks)} folds")
                results.append(run_one_fold(table, candidates, config, repeat, fold, train_idx, test_idx))
        else:
            results = Parallel(n_jobs=N_JOBS, verbose=10, prefer="threads")(
                delayed(run_one_fold)(table, candidates, config, repeat, fold, train_idx, test_idx)
                for config, repeat, fold, train_idx, test_idx in tasks
            )

    metric_rows = []
    pred_rows = []
    for metric, preds in results:
        metric_rows.append(metric)
        pred_rows.extend(preds)

    metrics_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)
    pooled_df = pooled_metrics(pred_df)
    counts_df = selection_counts(metrics_df)

    metrics_df.to_csv(OUT_DIR / "corr_pruned_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "corr_pruned_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled_df.to_csv(OUT_DIR / "corr_pruned_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    counts_df.to_csv(OUT_DIR / "corr_pruned_feature_selection_counts.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT_DIR / "subject_level_feature_table.csv", index=False, encoding="utf-8-sig")

    notes = {
        "analysis": "Correlation-pruned feature search inside outer training folds only.",
        "n_subjects": int(table.shape[0]),
        "positive": int(table["target"].sum()),
        "negative": int((1 - table["target"]).sum()),
        "n_candidate_features": len(candidates),
        "n_repeats": N_REPEATS,
        "configs": CONFIGS,
        "leakage_control": [
            "Feature ranking uses only outer-train subjects.",
            "Correlation pruning uses only outer-train subjects.",
            "Threshold selection uses inner OOF predictions within the outer-train subjects.",
            "Outer-test subjects are held out from selection, pruning, training, and threshold choice.",
        ],
    }
    (OUT_DIR / "corr_pruned_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pooled_df.to_string(index=False))
    print("\nTOP SELECTIONS")
    print(counts_df.groupby("config").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
