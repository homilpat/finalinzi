from __future__ import annotations

from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "analysis_outputs" / "physionet_labwalks_smartphone_shape_extractor_all_or"
OUT_DIR = ROOT / "analysis_outputs" / "physionet_shape_all_or_modeling"

BASE_FEATURES = [
    "ml_spec_entropy",
    "ml_stride_regularity",
    "ml_acf_stride_peak",
    "ml_stride_shape_cv_mean",
    "v_peak_timing_sd_pct",
    "v_stride_shape_cv_mean",
    "ap_stride_regularity",
    "v_stride_regularity",
    "step_time_median",
    "stride_time_median",
]


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) == 0:
        return 0.5
    candidates = np.r_[vals.min() - 1e-9, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-9]
    best_t = 0.5
    best_j = -np.inf
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(threshold)
    return best_t


def metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def max_abs_spearman(df: pd.DataFrame, features: list[str]) -> float:
    max_corr = 0.0
    for a, b in combinations(features, 2):
        pair = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 5:
            continue
        corr = stats.spearmanr(pair[a], pair[b]).correlation
        if np.isfinite(corr):
            max_corr = max(max_corr, abs(float(corr)))
    return max_corr


def vif_values(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    if len(features) < 2:
        return {features[0]: 1.0} if features else {}
    x = df[features].replace([np.inf, -np.inf], np.nan)
    x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=features)
    vals = {}
    for feature in features:
        y = x[feature].to_numpy()
        others = [c for c in features if c != feature]
        x_other = x[others].to_numpy()
        model = np.linalg.lstsq(
            np.column_stack([np.ones(len(x_other)), x_other]),
            y,
            rcond=None,
        )[0]
        pred = np.column_stack([np.ones(len(x_other)), x_other]) @ model
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vals[feature] = 1.0 / max(1e-9, 1.0 - r2)
    return vals


def evaluate_combo(df: pd.DataFrame, cols: list[str], n_repeats: int, seed_offset: int = 0) -> dict:
    y = df["target"].astype(int).to_numpy()
    prob_sum = np.zeros(len(df))
    count = np.zeros(len(df), dtype=int)
    fold_rows = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=220000 + seed_offset + repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(df[cols], y)):
            model = make_model(230000 + seed_offset + repeat * 10 + fold)
            model.fit(df.iloc[train_idx][cols], y[train_idx])
            train_prob = model.predict_proba(df.iloc[train_idx][cols])[:, 1]
            threshold = youden_threshold(y[train_idx], train_prob)
            test_prob = model.predict_proba(df.iloc[test_idx][cols])[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
            m = metrics(y[test_idx], test_prob, test_pred)
            m.update({"repeat": repeat, "fold": fold, "threshold": threshold})
            fold_rows.append(m)
            prob_sum[test_idx] += test_prob
            count[test_idx] += 1
    oof_prob = prob_sum / count
    oof_threshold = youden_threshold(y, oof_prob)
    oof_pred = (oof_prob >= oof_threshold).astype(int)
    oof = metrics(y, oof_prob, oof_pred)
    vifs = vif_values(df, cols)
    row = {
        "features": " + ".join(cols),
        "n_features": len(cols),
        "n_repeats": n_repeats,
        "max_abs_spearman": max_abs_spearman(df, cols),
        "max_vif": max(vifs.values()) if vifs else np.nan,
        "oof_threshold": oof_threshold,
        **{f"oof_{k}": v for k, v in oof.items()},
    }
    folds = pd.DataFrame(fold_rows)
    for metric in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        row[f"fold_mean_{metric}"] = folds[metric].mean()
        row[f"fold_std_{metric}"] = folds[metric].std()
    return row


def combo_screen(df: pd.DataFrame, features: list[str], n_repeats: int, max_k: int = 4) -> pd.DataFrame:
    tasks = []
    for k in range(1, min(max_k, len(features)) + 1):
        for combo in combinations(features, k):
            cols = list(combo)
            if max_abs_spearman(df, cols) > 0.90:
                continue
            vifs = vif_values(df, cols)
            if vifs and max(vifs.values()) > 10.0:
                continue
            tasks.append(cols)
    rows = []
    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_combo, df, cols, n_repeats) for cols in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if i % 25 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)} combos")
    return pd.DataFrame(rows).sort_values(
        ["oof_auc", "oof_sensitivity", "oof_specificity", "n_features"],
        ascending=[False, False, False, True],
    )


def train_full_and_external(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["target"].astype(int).to_numpy()
    model = make_model(990000)
    model.fit(df[features], y)
    train_prob = model.predict_proba(df[features])[:, 1]
    threshold = youden_threshold(y, train_prob)

    external_frames = []
    smartphone_path = ROOT / "analysis_outputs" / "waveform_shape_feature_analysis" / "waveform_shape_features_same_preprocessing.csv"
    if smartphone_path.exists():
        sp = pd.read_csv(smartphone_path)
        available = [c for c in features if c in sp.columns]
        if available == features:
            ext = sp.dropna(subset=["target"]).copy()
            ext["probability_impaired"] = model.predict_proba(ext[features])[:, 1]
            ext["prediction"] = (ext["probability_impaired"] >= threshold).astype(int)
            external_frames.append(ext[["dataset", "label_group", "target", "source_id", "subject_id", "probability_impaired", "prediction", *features]])

    local_path = ROOT / "analysis_outputs" / "fixed_best10_quality_pipeline" / "fixed_best10_sample_features.csv"
    # Local sample does not always have the same shape features; keep this as explicit missing-status output.
    local_status = pd.DataFrame(
        [
            {
                "source": "Local_SAMPLE",
                "status": "not_scored_if_missing_shape_features",
                "required_features": " + ".join(features),
                "available_file": str(local_path),
            }
        ]
    )
    if external_frames:
        external = pd.concat(external_frames, ignore_index=True)
    else:
        external = pd.DataFrame()
    return external, local_status


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = IN_DIR / "physionet_labwalks_shape_best10_all_or.csv"
    df = pd.read_csv(data_path)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)
    features = [c for c in BASE_FEATURES if c in df.columns]

    screen = combo_screen(df, features, n_repeats=20, max_k=4)
    screen.to_csv(OUT_DIR / "physionet_shape_all_or_combo_screen_x20.csv", index=False, encoding="utf-8-sig")

    top_rows = []
    top_rows = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                evaluate_combo,
                df,
                feature_text.split(" + "),
                100,
                10000 + rank * 1000,
            )
            for rank, feature_text in enumerate(screen.head(12)["features"])
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            top_rows.append(future.result())
            print(f"completed top recheck {i}/{len(futures)}")
    top = pd.DataFrame(top_rows).sort_values(
        ["oof_auc", "oof_sensitivity", "oof_specificity", "n_features"],
        ascending=[False, False, False, True],
    )
    top.to_csv(OUT_DIR / "physionet_shape_all_or_top_combo_recheck_x100.csv", index=False, encoding="utf-8-sig")
    best_features = top.iloc[0]["features"].split(" + ")
    external, local_status = train_full_and_external(df, best_features)
    if not external.empty:
        external.to_csv(OUT_DIR / "best_model_smartphone_external_predictions.csv", index=False, encoding="utf-8-sig")
        summary = (
            external.groupby(["dataset", "label_group", "target"], dropna=False)
            .agg(
                n=("source_id", "count"),
                pred_impaired_rate=("prediction", "mean"),
                prob_median=("probability_impaired", "median"),
                prob_mean=("probability_impaired", "mean"),
                prob_min=("probability_impaired", "min"),
                prob_max=("probability_impaired", "max"),
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "best_model_smartphone_external_summary.csv", index=False, encoding="utf-8-sig")
    local_status.to_csv(OUT_DIR / "best_model_local_sample_status.csv", index=False, encoding="utf-8-sig")

    print("label counts", df["target"].value_counts().sort_index().to_dict())
    print("features considered", features)
    print("\nTOP 15 x20 SCREEN")
    show_cols = [
        "features",
        "n_features",
        "max_abs_spearman",
        "max_vif",
        "oof_auc",
        "oof_accuracy",
        "oof_sensitivity",
        "oof_specificity",
        "oof_f1",
        "oof_tn",
        "oof_fp",
        "oof_fn",
        "oof_tp",
    ]
    print(screen[show_cols].head(15).to_string(index=False))
    print("\nTOP x100 RECHECK")
    print(top[show_cols].head(12).to_string(index=False))
    if not external.empty:
        print("\nEXTERNAL")
        print(summary.to_string(index=False))
    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
