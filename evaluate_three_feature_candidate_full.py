from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
FEATURE_CSV = ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
DAILY_WINDOW_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_service10_model_windows_merged.csv"
DAILY_SUBJECT_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_fixed_model_subject_predictions.csv"
CLINICAL_XLSX = ROOT / "final__2026" / "04_clinical_data" / "ClinicalDemogData_COFL.xlsx"
OUT_DIR = ROOT / "final__2026" / "12_three_feature_candidate_validation"

TARGET = "DGI_le19_or_TUG_ge12"
EXCLUDED_SUBJECTS = {"CO024", "FL020"}
FEATURES_3 = ["v_amp_pool_median", "ml_amp_pool_iqr", "roll_amp_pool_iqr"]
FEATURES_4 = ["v_amp_pool_median", "ml_amp_pool_iqr", "base_v_stride_regularity", "roll_amp_pool_iqr"]
N_REPEATS = 100
RANDOM_SEED = 20260713


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


def load_lab_subject_table(feature_set: list[str]) -> pd.DataFrame:
    features = pd.read_csv(FEATURE_CSV)
    features["subject_id"] = features["subject_id"].map(normalize_subject_id)
    merged = features.merge(load_labels(), on="subject_id", how="inner")
    merged = merged[~merged["subject_id"].isin(EXCLUDED_SUBJECTS)].copy()
    merged = merged[merged[TARGET].notna()].copy()
    rows = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "target": int(group[TARGET].iloc[0])}
        for feature in feature_set:
            row[feature] = float(pd.to_numeric(group[feature], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=feature_set, how="any").reset_index(drop=True)


def make_pipeline(random_state: int = 0) -> Pipeline:
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


def inner_oof_threshold(table: pd.DataFrame, train_idx: np.ndarray, features: list[str], repeat: int, fold: int) -> float:
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


def full_oof_threshold(table: pd.DataFrame, features: list[str]) -> tuple[float, np.ndarray]:
    y = table["target"].to_numpy(dtype=int)
    prob_sum = np.zeros(len(table), dtype=float)
    count = np.zeros(len(table), dtype=int)
    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=810000 + repeat)
        for fold, (tr, va) in enumerate(cv.split(table[features], y)):
            model = make_pipeline(910000 + repeat * 10 + fold)
            model.fit(table.iloc[tr][features], y[tr])
            prob_sum[va] += model.predict_proba(table.iloc[va][features])[:, 1]
            count[va] += 1
    oof_prob = prob_sum / count
    return choose_youden_threshold(y, oof_prob), oof_prob


def metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, threshold: float, n_boot: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    values = {"auc": [], "accuracy": [], "sensitivity": [], "specificity": [], "f1": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        pred = (prob[idx] >= threshold).astype(int)
        m = metrics(y[idx], prob[idx], pred)
        for key in values:
            values[key].append(m[key])
    for key, arr in values.items():
        rows.append({"metric": key, "ci_low": float(np.quantile(arr, 0.025)), "ci_high": float(np.quantile(arr, 0.975))})
    return pd.DataFrame(rows)


def nested_cv(table: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["target"].to_numpy(dtype=int)
    pred_rows = []
    fold_rows = []
    total = N_REPEATS * 5
    done = 0
    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + repeat)
        for fold, (tr, te) in enumerate(cv.split(table, y)):
            threshold = inner_oof_threshold(table, tr, features, repeat, fold)
            model = make_pipeline(710000 + repeat * 1000 + fold)
            model.fit(table.iloc[tr][features], y[tr])
            prob = model.predict_proba(table.iloc[te][features])[:, 1]
            pred = (prob >= threshold).astype(int)
            fold_metric = metrics(y[te], prob, pred)
            fold_rows.append({"repeat": repeat, "fold": fold, "threshold": threshold, **fold_metric})
            for subject_id, target, p, pr in zip(table.iloc[te]["subject_id"], y[te], prob, pred):
                pred_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "subject_id": subject_id,
                        "target": int(target),
                        "probability": float(p),
                        "threshold": float(threshold),
                        "prediction": int(pr),
                    }
                )
            done += 1
            if done % 100 == 0:
                print(f"nested cv completed {done}/{total} folds")
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def aggregate_daily(mode: str, windows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for subject_id, group in windows.groupby("subject_id", sort=True):
        valid = group.dropna(subset=features + ["base_v_stride_regularity"])
        if valid.empty:
            continue
        if mode == "best_window":
            selected = valid.loc[[valid["base_v_stride_regularity"].idxmax()]]
        elif mode == "top10_regularity_median":
            cutoff = valid["base_v_stride_regularity"].quantile(0.90)
            selected = valid[valid["base_v_stride_regularity"] >= cutoff]
        elif mode == "all_window_median":
            selected = valid
        else:
            raise ValueError(mode)
        row = selected[features].median(numeric_only=True).to_dict()
        row.update({"subject_id": subject_id, "n_windows": int(valid.shape[0]), "aggregation": mode})
        rows.append(row)
    return pd.DataFrame(rows)


def eval_daily(model: Pipeline, threshold: float, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(DAILY_WINDOW_CSV)
    labels = pd.read_csv(DAILY_SUBJECT_CSV)
    label_table = labels[labels["aggregation"] == "top10_regularity_median"][["subject_id", "target", "cohort"]].drop_duplicates()
    agg_frames = [aggregate_daily(mode, windows, features) for mode in ["best_window", "top10_regularity_median", "all_window_median"]]
    daily = pd.concat(agg_frames, ignore_index=True).merge(label_table, on="subject_id", how="inner")
    rows = []
    preds = []
    for (aggregation, cohort), group in daily.groupby(["aggregation", "cohort"], sort=True):
        prob = model.predict_proba(group[features])[:, 1]
        pred = (prob >= threshold).astype(int)
        m = metrics(group["target"].to_numpy(dtype=int), prob, pred)
        rows.append({"aggregation": aggregation, "cohort": cohort, "n": int(group.shape[0]), **m})
        for subject_id, target, p, pr in zip(group["subject_id"], group["target"], prob, pred):
            preds.append({"aggregation": aggregation, "cohort": cohort, "subject_id": subject_id, "target": int(target), "probability": float(p), "prediction": int(pr)})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def lab20_simulation(model: Pipeline, threshold: float, table_features: list[str]) -> pd.DataFrame:
    lab = pd.read_csv(FEATURE_CSV)
    lab["subject_id"] = lab["subject_id"].map(normalize_subject_id)
    target_map = load_lab_subject_table(table_features).set_index("subject_id")["target"].astype(int).to_dict()
    lab = lab[lab["subject_id"].isin(target_map)].dropna(subset=table_features + ["base_v_stride_regularity"]).copy()
    grouped = {sid: g.sort_values(["record", "start_sec"]).reset_index(drop=True) for sid, g in lab.groupby("subject_id")}
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for mode in ["random10_in_random20", "best10_in_random20"]:
        metric_rows = []
        for repeat in range(1000):
            selected = []
            for sid, group in grouped.items():
                anchor = float(rng.choice(group["start_sec"].to_numpy(dtype=float)))
                cand = group[(group["start_sec"] >= anchor) & (group["end_sec"] <= anchor + 20)].dropna(subset=table_features)
                if cand.empty:
                    cand = group.dropna(subset=table_features)
                if mode == "best10_in_random20":
                    row = cand.loc[cand["base_v_stride_regularity"].idxmax()]
                else:
                    row = cand.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
                selected.append(row)
            frame = pd.DataFrame(selected)
            y = np.array([target_map[sid] for sid in frame["subject_id"]], dtype=int)
            prob = model.predict_proba(frame[table_features])[:, 1]
            pred = (prob >= threshold).astype(int)
            metric_rows.append(metrics(y, prob, pred))
        metric_df = pd.DataFrame(metric_rows)
        row = {"mode": mode, "n_repeats": 1000}
        for col in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
            row[col] = float(metric_df[col].mean())
            row[f"{col}_p025"] = float(metric_df[col].quantile(0.025))
            row[f"{col}_p975"] = float(metric_df[col].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_lab_subject_table(FEATURES_3)
    y = table["target"].to_numpy(dtype=int)

    pred_df, fold_df = nested_cv(table, FEATURES_3)
    pooled_y = pred_df["target"].to_numpy(dtype=int)
    pooled_prob = pred_df["probability"].to_numpy(dtype=float)
    pooled_pred = pred_df["prediction"].to_numpy(dtype=int)
    nested_metric = metrics(pooled_y, pooled_prob, pooled_pred)

    deployment_threshold, full_oof_prob = full_oof_threshold(table, FEATURES_3)
    final_model = make_pipeline(0)
    final_model.fit(table[FEATURES_3], y)
    apparent_prob = final_model.predict_proba(table[FEATURES_3])[:, 1]
    apparent_pred = (apparent_prob >= deployment_threshold).astype(int)
    apparent_metric = metrics(y, apparent_prob, apparent_pred)
    oof_ci = bootstrap_ci(y, full_oof_prob, deployment_threshold)

    daily_metrics, daily_preds = eval_daily(final_model, deployment_threshold, FEATURES_3)
    lab20_metrics = lab20_simulation(final_model, deployment_threshold, FEATURES_3)

    model_bundle = {
        "pipeline": final_model,
        "features": FEATURES_3,
        "threshold": deployment_threshold,
        "threshold_strategy": "pooled_5fold_x100_oof_youden_three_feature_candidate",
        "decision_rule": "probability >= threshold -> motor_impairment_possible",
    }
    joblib.dump(model_bundle, OUT_DIR / "three_feature_candidate_model.joblib")

    pd.DataFrame([{ "evaluation": "nested_5fold_x100_pooled", **nested_metric }]).to_csv(OUT_DIR / "three_feature_nested_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{ "evaluation": "apparent_full_train", **apparent_metric }]).to_csv(OUT_DIR / "three_feature_apparent_train_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "three_feature_nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT_DIR / "three_feature_nested_fold_metrics.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(OUT_DIR / "three_feature_daily_validation_metrics.csv", index=False, encoding="utf-8-sig")
    daily_preds.to_csv(OUT_DIR / "three_feature_daily_predictions.csv", index=False, encoding="utf-8-sig")
    lab20_metrics.to_csv(OUT_DIR / "three_feature_lab20_simulation_metrics.csv", index=False, encoding="utf-8-sig")
    oof_ci.to_csv(OUT_DIR / "three_feature_oof_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    notes = {
        "features": FEATURES_3,
        "n_subjects": int(table.shape[0]),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "deployment_threshold": float(deployment_threshold),
        "nested_pooled_metrics": nested_metric,
        "apparent_train_metrics": apparent_metric,
        "outputs": str(OUT_DIR),
    }
    (OUT_DIR / "three_feature_validation_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print("NESTED")
    print(pd.DataFrame([{**nested_metric}]).to_string(index=False))
    print("\nAPPARENT")
    print(pd.DataFrame([{**apparent_metric}]).to_string(index=False))
    print("\nTHRESHOLD", deployment_threshold)
    print("\nCI")
    print(oof_ci.to_string(index=False))
    print("\nDAILY")
    print(daily_metrics.to_string(index=False))
    print("\nLAB20")
    print(lab20_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
