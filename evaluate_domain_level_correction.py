from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
MODEL_PATH = ROOT / "final__2026" / "02_model" / "final_motor_domain4_labwalks10_logistic_C0p5_nested_youden.joblib"
PAIRED_CSV = ROOT / "final__2026" / "06_lab_daily_domain_gap" / "paired_lab_daily_subject_feature_gaps.csv"
DAILY_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_fixed_model_subject_predictions.csv"
OUT_DIR = ROOT / "final__2026" / "07_domain_level_correction"

FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]
AGGREGATIONS = ["best_window", "top10_regularity_median", "all_window_median"]
RANDOM_SEED = 20260713
N_REPEATS = 100


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((1 - y_true).sum()),
        "auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_prob(model_bundle: dict, frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    cols = [f"{prefix}{f}" for f in FEATURES]
    x = frame[cols].copy()
    x.columns = FEATURES
    return model_bundle["pipeline"].predict_proba(x[FEATURES])[:, 1]


def apply_offsets(frame: pd.DataFrame, offsets: dict[str, float], prefix: str = "") -> pd.DataFrame:
    corrected = frame.copy()
    for feat in FEATURES:
        col = f"{prefix}{feat}"
        corrected[col] = corrected[col] - offsets[feat]
    return corrected


def paired_cv_correction(model_bundle: dict, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = float(model_bundle["threshold"])
    prediction_rows = []

    for aggregation in AGGREGATIONS:
        data = paired[paired["aggregation"] == aggregation].copy().reset_index(drop=True)
        y = data["target"].astype(int).to_numpy()
        for repeat in range(N_REPEATS):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv.split(data, y)):
                train = data.iloc[train_idx]
                test = data.iloc[test_idx]
                offsets = {
                    feat: float((train[f"daily_{feat}"] - train[f"lab_{feat}"]).mean())
                    for feat in FEATURES
                }
                corrected_test = apply_offsets(test, offsets, prefix="daily_")

                raw_prob = predict_prob(model_bundle, test, prefix="daily_")
                corrected_prob = predict_prob(model_bundle, corrected_test, prefix="daily_")

                for local_i, row_i in enumerate(test.index):
                    base = {
                        "aggregation": aggregation,
                        "repeat": repeat,
                        "fold": fold,
                        "subject_id": test.loc[row_i, "subject_id"],
                        "target": int(test.loc[row_i, "target"]),
                    }
                    prediction_rows.append({**base, "method": "uncorrected", "probability": float(raw_prob[local_i])})
                    prediction_rows.append({**base, "method": "cv_train_offset_corrected", "probability": float(corrected_prob[local_i])})

    pred_df = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (aggregation, method, repeat), group in pred_df.groupby(["aggregation", "method", "repeat"], sort=True):
        metric = metrics(group["target"].to_numpy(dtype=int), group["probability"].to_numpy(dtype=float), threshold)
        repeat_rows.append(
            {
                "aggregation": aggregation,
                "method": method,
                "repeat": int(repeat),
                **metric,
            }
        )
    repeat_df = pd.DataFrame(repeat_rows)

    summary_rows = []
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    count_cols = ["n", "positive", "negative"]
    for (aggregation, method), group in repeat_df.groupby(["aggregation", "method"], sort=True):
        row = {
            "evaluation": "paired_62_5fold_x100_no_leak_offset",
            "aggregation": aggregation,
            "method": method,
            "n_subjects": int(group["n"].iloc[0]),
            "positive": int(group["positive"].iloc[0]),
            "negative": int(group["negative"].iloc[0]),
            "n_repeats": int(group.shape[0]),
        }
        for col in metric_cols:
            row[col] = float(group[col].mean())
            row[f"{col}_sd"] = float(group[col].std(ddof=1))
        summary_rows.append(row)
    return pred_df, pd.DataFrame(summary_rows), repeat_df


def fixed_offset_daily69(model_bundle: dict, paired: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = float(model_bundle["threshold"])
    offset_rows = []
    pred_rows = []
    summary_rows = []

    daily69 = daily[daily["cohort"] == "exclude_CO024_FL020"].copy()
    for aggregation in AGGREGATIONS:
        paired_a = paired[paired["aggregation"] == aggregation].copy()
        offsets = {
            feat: float((paired_a[f"daily_{feat}"] - paired_a[f"lab_{feat}"]).mean())
            for feat in FEATURES
        }
        for feat, value in offsets.items():
            offset_rows.append(
                {
                    "aggregation": aggregation,
                    "feature": feat,
                    "n_paired_for_offset": int(paired_a.shape[0]),
                    "mean_daily_minus_lab_offset": value,
                    "correction_applied_to_daily": -value,
                }
            )

        eval_a = daily69[daily69["aggregation"] == aggregation].copy()
        raw_prob = predict_prob(model_bundle, eval_a)
        corrected = apply_offsets(eval_a, offsets)
        corrected_prob = predict_prob(model_bundle, corrected)

        for i, (_, row) in enumerate(eval_a.iterrows()):
            base = {
                "aggregation": aggregation,
                "subject_id": row["subject_id"],
                "target": int(row["target"]),
                "cohort": row["cohort"],
            }
            pred_rows.append({**base, "method": "uncorrected", "probability": float(raw_prob[i])})
            pred_rows.append({**base, "method": "fixed_paired62_offset_corrected", "probability": float(corrected_prob[i])})

    pred_df = pd.DataFrame(pred_rows)
    offsets_df = pd.DataFrame(offset_rows)
    for (aggregation, method), group in pred_df.groupby(["aggregation", "method"], sort=True):
        metric = metrics(group["target"].to_numpy(dtype=int), group["probability"].to_numpy(dtype=float), threshold)
        summary_rows.append(
            {
                "evaluation": "daily69_fixed_offset_from_paired62_exploratory",
                "aggregation": aggregation,
                "method": method,
                **metric,
            }
        )
    return pred_df, pd.DataFrame(summary_rows), offsets_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_bundle = joblib.load(MODEL_PATH)
    paired = pd.read_csv(PAIRED_CSV)
    daily = pd.read_csv(DAILY_CSV)

    cv_pred, cv_summary, cv_repeat_summary = paired_cv_correction(model_bundle, paired)
    fixed_pred, fixed_summary, offsets = fixed_offset_daily69(model_bundle, paired, daily)
    summary = pd.concat([cv_summary, fixed_summary], ignore_index=True)

    cv_pred.to_csv(OUT_DIR / "paired62_no_leak_domain_offset_predictions.csv", index=False, encoding="utf-8-sig")
    cv_repeat_summary.to_csv(OUT_DIR / "paired62_no_leak_domain_offset_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    fixed_pred.to_csv(OUT_DIR / "daily69_fixed_paired62_offset_predictions.csv", index=False, encoding="utf-8-sig")
    offsets.to_csv(OUT_DIR / "domain_level_offsets_from_paired62.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "domain_level_correction_metric_summary.csv", index=False, encoding="utf-8-sig")

    notes = {
        "model": str(MODEL_PATH),
        "threshold": float(model_bundle["threshold"]),
        "features": FEATURES,
        "evaluations": [
            "paired 62 subjects, 5-fold x100, offsets learned on train folds only and applied to held-out paired daily subjects",
            "daily 69 subjects excluding CO024/FL020, fixed offsets estimated from paired 62 subjects; exploratory because 62 paired subjects overlap the evaluation cohort",
        ],
        "leakage_note": "Use the paired 62 no-leak CV result as the defensible correction estimate. Treat daily69 fixed-offset result as deployment-style exploratory analysis, not strict independent validation.",
    }
    (OUT_DIR / "domain_level_correction_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print("\nOFFSETS")
    print(offsets.to_string(index=False))


if __name__ == "__main__":
    main()
