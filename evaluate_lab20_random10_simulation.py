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
LAB_CSV = ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
OOF_CSV = ROOT / "final__2026" / "02_model" / "domain4_nested_oof_predictions.csv"
OUT_DIR = ROOT / "final__2026" / "09_lab20_random10_service_simulation"

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


def select_from_random20(subject_windows: pd.DataFrame, rng: np.random.Generator, mode: str) -> pd.Series:
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

    if mode == "random10_in_random20":
        selected = candidates.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0].copy()
    elif mode == "best10_in_random20":
        selected = candidates.loc[candidates["base_v_stride_regularity"].idxmax()].copy()
    else:
        raise ValueError(mode)

    selected["random20_start_sec"] = anchor
    selected["random20_end_sec"] = anchor + 20.0
    selected["n_candidate_10s_in_20s"] = int(candidates.shape[0])
    return selected


def summarize(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    for mode, group in repeat_metrics.groupby("mode", sort=True):
        row = {
            "mode": mode,
            "n_subjects": int(group["n"].iloc[0]),
            "positive": int(group["positive"].iloc[0]),
            "negative": int(group["negative"].iloc[0]),
            "n_repeats": int(group.shape[0]),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[col] = float(values.mean())
            row[f"{col}_sd"] = float(values.std(ddof=1))
            row[f"{col}_p025"] = float(np.quantile(values, 0.025))
            row[f"{col}_p975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_bundle = joblib.load(MODEL_PATH)

    lab = pd.read_csv(LAB_CSV).dropna(subset=FEATURES).copy()
    targets = pd.read_csv(OOF_CSV)[["subject_id", "target"]].drop_duplicates()
    final_subjects = set(targets["subject_id"])
    target_map = targets.set_index("subject_id")["target"].astype(int).to_dict()
    lab = lab[lab["subject_id"].isin(final_subjects)].copy()
    grouped = {sid: g.sort_values(["record", "start_sec"]).reset_index(drop=True) for sid, g in lab.groupby("subject_id")}
    subjects = sorted(set(grouped) & set(target_map))

    rng = np.random.default_rng(RANDOM_SEED)
    threshold = float(model_bundle["threshold"])
    prediction_rows = []
    metric_rows = []
    modes = ["random10_in_random20", "best10_in_random20"]

    for mode in modes:
        for repeat in range(N_REPEATS):
            selected_rows = []
            for subject_id in subjects:
                row = select_from_random20(grouped[subject_id], rng, mode)
                row["target"] = target_map[subject_id]
                selected_rows.append(row)
            selected_df = pd.DataFrame(selected_rows)
            prob = model_bundle["pipeline"].predict_proba(selected_df[FEATURES])[:, 1]
            y = selected_df["target"].astype(int).to_numpy()
            metric_rows.append({"mode": mode, "repeat": repeat, **metrics(y, prob, threshold)})
            for i, (_, row) in enumerate(selected_df.iterrows()):
                prediction_rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "subject_id": row["subject_id"],
                        "target": int(row["target"]),
                        "probability": float(prob[i]),
                        "prediction": int(prob[i] >= threshold),
                        "record": row["record"],
                        "selected_start_sec": float(row["start_sec"]),
                        "selected_end_sec": float(row["end_sec"]),
                        "random20_start_sec": float(row["random20_start_sec"]),
                        "random20_end_sec": float(row["random20_end_sec"]),
                        "n_candidate_10s_in_20s": int(row["n_candidate_10s_in_20s"]),
                        **{feat: float(row[feat]) for feat in FEATURES},
                    }
                )

    repeat_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(repeat_metrics)

    summary.to_csv(OUT_DIR / "lab20_random10_metric_summary.csv", index=False, encoding="utf-8-sig")
    repeat_metrics.to_csv(OUT_DIR / "lab20_random10_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "lab20_random10_subject_predictions.csv", index=False, encoding="utf-8-sig")

    notes = {
        "simulation": "LabWalks structured-walking simulation. For each subject, sample a random 20-second interval from available lab 10-second windows and evaluate either a random 10-second candidate or the best-regularity 10-second candidate inside that 20-second interval.",
        "n_subjects": len(subjects),
        "n_repeats": N_REPEATS,
        "random_seed": RANDOM_SEED,
        "model": str(MODEL_PATH),
        "threshold": threshold,
        "features": FEATURES,
        "outputs": {
            "summary": str(OUT_DIR / "lab20_random10_metric_summary.csv"),
            "repeat_metrics": str(OUT_DIR / "lab20_random10_repeat_metrics.csv"),
            "predictions": str(OUT_DIR / "lab20_random10_subject_predictions.csv"),
        },
    }
    (OUT_DIR / "lab20_random10_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
