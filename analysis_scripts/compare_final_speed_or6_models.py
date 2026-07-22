"""
Final model comparison for the expanded clinical mobility label.

Label:
    TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19
    OR base_velocity < 1.0 OR s3_velocity < 1.0

Validation:
    - subject-level 5-fold StratifiedKFold
    - no subject appears in both train and test because each subject has one row
    - traditional ML uses fold-internal preprocessing pipelines
    - CNN/LSTM use fold-internal imputation/scaling fit on train sequences only
    - threshold fixed at 0.50 for screening-oriented service comparison

Outputs:
    analysis_outputs/final_model_comparison_speed_or6/
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final_model_comparison_speed_or6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception as exc:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
    DEVICE = None
    print(f"[skip] PyTorch unavailable: {exc}")


FEATURES = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
TABLE_CSV = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
THRESHOLD = 0.50
N_SPLITS = 5
SEED = 42
MAX_LEN = 120


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def make_label(clinical: pd.DataFrame) -> pd.Series:
    for col in ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)"]:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    return (
        (clinical["TUG"] >= 12)
        | (clinical["FSST"] >= 15)
        | (clinical["BERG"] < 52)
        | (clinical["DGI"] <= 19)
        | (clinical["base(velocity)"] < 1.0)
        | (clinical["s3(velocity)"] < 1.0)
    ).astype(int)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(TABLE_CSV)
    clinical = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    clinical["target"] = make_label(clinical)
    labels = clinical[["subject_id", "target"]].drop_duplicates("subject_id")
    table = table.merge(labels, on="subject_id", how="inner", suffixes=("_old", ""))
    table = table.drop(columns=[c for c in table.columns if c.endswith("_old")])
    table = table.dropna(subset=FEATURES).reset_index(drop=True)

    subject_table = (
        table.groupby("subject_id")[FEATURES + ["target"]]
        .agg({**{f: "median" for f in FEATURES}, "target": "first"})
        .reset_index()
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )
    subject_ids = subject_table["subject_id"].to_numpy()
    x_subject = subject_table[FEATURES].to_numpy(dtype=np.float32)
    y = subject_table["target"].to_numpy(dtype=int)

    x_seq = np.zeros((len(subject_ids), MAX_LEN, len(FEATURES)), dtype=np.float32)
    lengths = np.zeros(len(subject_ids), dtype=np.int64)
    for idx, sid in enumerate(subject_ids):
        rows = table.loc[table["subject_id"] == sid, FEATURES].to_numpy(dtype=np.float32)
        n = min(len(rows), MAX_LEN)
        x_seq[idx, :n] = rows[:n]
        lengths[idx] = max(n, 1)

    return subject_table, x_subject, x_seq, lengths, y


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def base_steps() -> list:
    return [("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]


def traditional_models() -> dict[str, Pipeline]:
    cpu = max(1, os.cpu_count() or 1)
    return {
        "LR": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "RF": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GBM": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "XGB": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.04,
                        max_depth=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=SEED,
                        n_jobs=cpu,
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "Voting": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    VotingClassifier(
                        voting="soft",
                        n_jobs=-1,
                        estimators=[
                            (
                                "lr",
                                LogisticRegression(
                                    C=1.0,
                                    class_weight="balanced",
                                    max_iter=3000,
                                    solver="liblinear",
                                    random_state=SEED,
                                ),
                            ),
                            (
                                "rf",
                                RandomForestClassifier(
                                    n_estimators=300,
                                    class_weight="balanced",
                                    random_state=SEED,
                                    n_jobs=-1,
                                ),
                            ),
                            (
                                "xgb",
                                XGBClassifier(
                                    n_estimators=250,
                                    learning_rate=0.04,
                                    max_depth=2,
                                    eval_metric="logloss",
                                    random_state=SEED,
                                    n_jobs=cpu,
                                    verbosity=0,
                                ),
                            ),
                        ],
                    ),
                ),
            ]
        ),
        "Stacking": Pipeline(
            [
                *base_steps(),
                (
                    "model",
                    StackingClassifier(
                        cv=3,
                        n_jobs=-1,
                        final_estimator=LogisticRegression(max_iter=3000, solver="liblinear"),
                        estimators=[
                            (
                                "rf",
                                RandomForestClassifier(
                                    n_estimators=250,
                                    class_weight="balanced",
                                    random_state=SEED,
                                    n_jobs=-1,
                                ),
                            ),
                            (
                                "xgb",
                                XGBClassifier(
                                    n_estimators=250,
                                    learning_rate=0.04,
                                    max_depth=2,
                                    eval_metric="logloss",
                                    random_state=SEED,
                                    n_jobs=cpu,
                                    verbosity=0,
                                ),
                            ),
                            (
                                "svm",
                                SVC(
                                    C=1.0,
                                    gamma="scale",
                                    kernel="rbf",
                                    class_weight="balanced",
                                    probability=True,
                                    random_state=SEED,
                                ),
                            ),
                        ],
                    ),
                ),
            ]
        ),
    }


def assert_no_subject_overlap(subject_ids: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = set(subject_ids[train_idx]).intersection(set(subject_ids[test_idx]))
    if overlap:
        raise RuntimeError(f"Subject leakage detected: {sorted(overlap)[:5]}")


def evaluate_traditional(
    name: str,
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    subject_ids: np.ndarray,
) -> dict:
    oof_prob = np.zeros(len(y), dtype=float)
    train_rows, test_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        assert_no_subject_overlap(subject_ids, train_idx, test_idx)
        model.fit(x[train_idx], y[train_idx])
        train_prob = model.predict_proba(x[train_idx])[:, 1]
        test_prob = model.predict_proba(x[test_idx])[:, 1]
        oof_prob[test_idx] = test_prob
        train_rows.append({"model": name, "fold": fold, **classifier_metrics(y[train_idx], train_prob)})
        test_rows.append({"model": name, "fold": fold, **classifier_metrics(y[test_idx], test_prob)})
    return {"oof_prob": oof_prob, "train": train_rows, "test": test_rows}


if TORCH_AVAILABLE:

    class LSTMNet(nn.Module):
        def __init__(self, input_size: int = 3, hidden_size: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.25)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            packed = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            return self.fc(self.dropout(hidden[-1])).squeeze(1)


    class CNN1DNet(nn.Module):
        def __init__(self, input_size: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            self.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(32, 1))

        def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            del lengths
            z = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.fc(z).squeeze(1)


def transform_sequences(
    x_seq: np.ndarray,
    lengths: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_chunks = []
    for idx in train_idx:
        train_chunks.append(x_seq[idx, : lengths[idx]])
    train_flat = np.vstack(train_chunks)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    scaler.fit(imputer.fit_transform(train_flat))

    def apply(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), x_seq.shape[1], x_seq.shape[2]), dtype=np.float32)
        for row_i, source_i in enumerate(indices):
            n = lengths[source_i]
            valid = imputer.transform(x_seq[source_i, :n])
            out[row_i, :n] = scaler.transform(valid).astype(np.float32)
        return out

    return apply(train_idx), apply(test_idx)


def train_torch_fold(
    model_cls,
    x_train: np.ndarray,
    y_train: np.ndarray,
    len_train: np.ndarray,
    x_test: np.ndarray,
    len_test: np.ndarray,
    seed: int,
    epochs: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE))

    x_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    l_t = torch.tensor(len_train, dtype=torch.long, device=DEVICE)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_t, l_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x_t, l_t)).detach().cpu().numpy()
        test_prob = torch.sigmoid(
            model(
                torch.tensor(x_test, dtype=torch.float32, device=DEVICE),
                torch.tensor(len_test, dtype=torch.long, device=DEVICE),
            )
        ).detach().cpu().numpy()
    return train_prob, test_prob


def evaluate_torch_model(
    name: str,
    model_cls,
    x_seq: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    splits: list,
    subject_ids: np.ndarray,
) -> dict:
    oof_prob = np.zeros(len(y), dtype=float)
    train_rows, test_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        assert_no_subject_overlap(subject_ids, train_idx, test_idx)
        train_seq, test_seq = transform_sequences(x_seq, lengths, train_idx, test_idx)
        train_prob, test_prob = train_torch_fold(
            model_cls,
            train_seq,
            y[train_idx],
            lengths[train_idx],
            test_seq,
            lengths[test_idx],
            seed=SEED + fold,
        )
        oof_prob[test_idx] = test_prob
        train_rows.append({"model": name, "fold": fold, **classifier_metrics(y[train_idx], train_prob)})
        test_rows.append({"model": name, "fold": fold, **classifier_metrics(y[test_idx], test_prob)})
    return {"oof_prob": oof_prob, "train": train_rows, "test": test_rows}


def save_confusion_matrix(name: str, y: np.ndarray, prob: np.ndarray) -> None:
    pred = (prob >= THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cm, cmap="Blues")
    del image
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=12,
            )
    ax.set_xticks([0, 1], ["Pred normal", "Pred impaired"])
    ax.set_yticks([0, 1], ["True normal", "True impaired"])
    ax.set_title(f"Confusion matrix - {name} (thr={THRESHOLD:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_matrix_{name}.png", dpi=180)
    plt.close(fig)


def save_roc(name: str, y: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2.3, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {name}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"roc_curve_{name}.png", dpi=180)
    plt.close(fig)
    return fpr, tpr, auc


def summarize_gap(train_rows: list, test_rows: list, oof_metric: dict) -> dict:
    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(test_rows)
    out = {"folds": int(len(test))}
    for metric in ["auc", "sensitivity", "specificity", "accuracy", "f1"]:
        out[f"train_{metric}_mean"] = float(train[metric].mean())
        out[f"test_{metric}_mean"] = float(test[metric].mean())
        out[f"gap_{metric}"] = float(train[metric].mean() - test[metric].mean())
    out.update({f"oof_{k}": v for k, v in oof_metric.items() if k in ["auc", "sensitivity", "specificity", "accuracy", "f1", "tn", "fp", "fn", "tp"]})
    return out


def main() -> None:
    set_seed(SEED)
    subject_table, x_subject, x_seq, lengths, y = load_data()
    subject_ids = subject_table["subject_id"].to_numpy()
    splits = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(x_subject, y))

    print(f"Output: {OUT_DIR}")
    print(f"CPU cores: {os.cpu_count() or 1}")
    print(f"PyTorch: {TORCH_AVAILABLE} device={DEVICE}")
    print(f"Subjects={len(y)} normal={(y == 0).sum()} impaired={(y == 1).sum()}")
    print(f"Threshold={THRESHOLD}")

    all_train_rows, all_test_rows, summary_rows = [], [], []
    roc_items = {}

    results = {}
    for name, model in traditional_models().items():
        print(f"\n[ML] {name}")
        results[name] = evaluate_traditional(name, model, x_subject, y, splits, subject_ids)

    if TORCH_AVAILABLE:
        for name, model_cls in [("LSTM", LSTMNet), ("CNN1D", CNN1DNet)]:
            print(f"\n[DL] {name}")
            results[name] = evaluate_torch_model(name, model_cls, x_seq, lengths, y, splits, subject_ids)

    for name, result in results.items():
        prob = result["oof_prob"]
        oof_metric = classifier_metrics(y, prob)
        all_train_rows.extend(result["train"])
        all_test_rows.extend(result["test"])
        summary = summarize_gap(result["train"], result["test"], oof_metric)
        summary["model"] = name
        summary_rows.append(summary)
        save_confusion_matrix(name, y, prob)
        roc_items[name] = save_roc(name, y, prob)
        print(
            f"  OOF AUC={oof_metric['auc']:.3f} sens={oof_metric['sensitivity']:.3f} "
            f"spec={oof_metric['specificity']:.3f} cm=[[{oof_metric['tn']},{oof_metric['fp']}],[{oof_metric['fn']},{oof_metric['tp']}]]"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("oof_auc", ascending=False)
    train_df = pd.DataFrame(all_train_rows)
    test_df = pd.DataFrame(all_test_rows)
    summary_df.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(OUT_DIR / "train_fold_metrics.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test_fold_metrics.csv", index=False, encoding="utf-8-sig")
    subject_table.to_csv(OUT_DIR / "subject_feature_table.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (fpr, tpr, auc) in sorted(roc_items.items(), key=lambda item: item[1][2], reverse=True):
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison - expanded mobility label")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_all_models.png", dpi=180)
    plt.close(fig)

    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "label": "TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19 OR base_velocity < 1.0 OR s3_velocity < 1.0",
                "threshold": THRESHOLD,
                "features": FEATURES,
                "n_subjects": int(len(y)),
                "n_normal": int((y == 0).sum()),
                "n_impaired": int((y == 1).sum()),
                "device": str(DEVICE),
                "cpu_cores": os.cpu_count() or 1,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSummary")
    print(summary_df[["model", "oof_auc", "oof_sensitivity", "oof_specificity", "oof_accuracy", "gap_auc"]].to_string(index=False))
    print(f"\nSaved summary: {OUT_DIR / 'model_comparison_summary.csv'}")
    print(f"Saved ROC: {OUT_DIR / 'roc_all_models.png'}")


if __name__ == "__main__":
    main()
