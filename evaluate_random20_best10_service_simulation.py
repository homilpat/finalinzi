from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path(r"C:\Users\whdgu\Desktop\파이널 프로젝트")
MODEL_PATH = ROOT / "final__2026" / "02_model" / "final_motor_domain4_labwalks10_logistic_C0p5_nested_youden.joblib"
WINDOW_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_service10_model_windows_merged.csv"
SUBJECT_CSV = ROOT / "final__2026" / "05_daily75h_validation" / "daily75h_fixed_model_subject_predictions.csv"
OUT_DIR = ROOT / "final__2026" / "08_random20_best10_service_simulation"

FEATURES = [
    "v_amp_pool_median",
    "ml_amp_pool_iqr",
    "base_v_stride_regularity",
    "roll_amp_pool_iqr",
]

N_REPEATS = 1000
RANDOM_SEED = 20260713


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


def sample_best10_from_random20(subject_windows: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    starts = subject_windows["start_sec"].dropna().to_numpy(dtype=float)
    if starts.size == 0:
        raise ValueError("No valid starts")

    anchor = float(rng.choice(starts))
    candidates = subject_windows[
        (subject_windows["start_sec"] >= anchor)
        & (subject_windows["end_sec"] <= anchor + 20.0)
    ].dropna(subset=FEATURES)

    if candidates.empty:
        candidates = subject_windows.dropna(subset=FEATURES)
    idx = candidates["base_v_stride_regularity"].idxmax()
    selected = candidates.loc[idx].copy()
    selected["random20_start_sec"] = anchor
    selected["random20_end_sec"] = anchor + 20.0
    selected["n_candidate_10s_in_20s"] = int(candidates.shape[0])
    return selected


def simulate(model_bundle: dict, windows: pd.DataFrame, labels: pd.DataFrame, cohort_name: str, subjects: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    threshold = float(model_bundle["threshold"])
    label_map = labels.set_index("subject_id")["target"].astype(int).to_dict()

    windows = windows[windows["subject_id"].isin(subjects)].dropna(subset=FEATURES).copy()
    grouped = {sid: g.sort_values("start_sec").reset_index(drop=True) for sid, g in windows.groupby("subject_id")}
    valid_subjects = sorted(set(grouped) & set(label_map))

    prediction_rows = []
    repeat_rows = []
    for repeat in range(N_REPEATS):
        selected_rows = []
        for subject_id in valid_subjects:
            selected = sample_best10_from_random20(grouped[subject_id], rng)
            selected["target"] = label_map[subject_id]
            selected_rows.append(selected)
        selected_df = pd.DataFrame(selected_rows)
        prob = model_bundle["pipeline"].predict_proba(selected_df[FEATURES])[:, 1]
        y = selected_df["target"].astype(int).to_numpy()
        metric = metrics(y, prob, threshold)
        repeat_rows.append({"cohort": cohort_name, "repeat": repeat, **metric})
        for i, (_, row) in enumerate(selected_df.iterrows()):
            prediction_rows.append(
                {
                    "cohort": cohort_name,
                    "repeat": repeat,
                    "subject_id": row["subject_id"],
                    "target": int(row["target"]),
                    "probability": float(prob[i]),
                    "prediction": int(prob[i] >= threshold),
                    "selected_start_sec": float(row["start_sec"]),
                    "selected_end_sec": float(row["end_sec"]),
                    "random20_start_sec": float(row["random20_start_sec"]),
                    "random20_end_sec": float(row["random20_end_sec"]),
                    "n_candidate_10s_in_20s": int(row["n_candidate_10s_in_20s"]),
                    **{f: float(row[f]) for f in FEATURES},
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(repeat_rows)


def summarize_repeats(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    for cohort, group in repeat_metrics.groupby("cohort", sort=True):
        row = {
            "cohort": cohort,
            "n_subjects": int(group["n"].iloc[0]),
            "positive": int(group["positive"].iloc[0]),
            "negative": int(group["negative"].iloc[0]),
            "n_repeats": int(group.shape[0]),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            row[f"{col}_sd"] = float(np.std(values, ddof=1))
            row[f"{col}_p025"] = float(np.quantile(values, 0.025))
            row[f"{col}_p975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_bundle = joblib.load(MODEL_PATH)
    windows = pd.read_csv(WINDOW_CSV)
    subject_preds = pd.read_csv(SUBJECT_CSV)

    labels_all = subject_preds[subject_preds["aggregation"] == "top10_regularity_median"].copy()
    all_subjects = set(labels_all[labels_all["cohort"] == "all_matched_valid"]["subject_id"])
    daily69_subjects = set(labels_all[labels_all["cohort"] == "exclude_CO024_FL020"]["subject_id"])

    all_pred, all_repeat = simulate(model_bundle, windows, labels_all, "all_matched_valid_71", all_subjects)
    excl_pred, excl_repeat = simulate(model_bundle, windows, labels_all, "exclude_CO024_FL020_69", daily69_subjects)

    predictions = pd.concat([all_pred, excl_pred], ignore_index=True)
    repeats = pd.concat([all_repeat, excl_repeat], ignore_index=True)
    summary = summarize_repeats(repeats)

    predictions.to_csv(OUT_DIR / "random20_best10_subject_predictions.csv", index=False, encoding="utf-8-sig")
    repeats.to_csv(OUT_DIR / "random20_best10_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "random20_best10_metric_summary.csv", index=False, encoding="utf-8-sig")

    notes = {
        "simulation": "For each subject and repeat, randomly sample a 20-second service collection window from daily 10-second windows, then select the 10-second window within it with the highest base_v_stride_regularity.",
        "n_repeats": N_REPEATS,
        "random_seed": RANDOM_SEED,
        "model": str(MODEL_PATH),
        "threshold": float(model_bundle["threshold"]),
        "features": FEATURES,
        "outputs": {
            "summary": str(OUT_DIR / "random20_best10_metric_summary.csv"),
            "repeat_metrics": str(OUT_DIR / "random20_best10_repeat_metrics.csv"),
            "predictions": str(OUT_DIR / "random20_best10_subject_predictions.csv"),
        },
    }
    (OUT_DIR / "random20_best10_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
