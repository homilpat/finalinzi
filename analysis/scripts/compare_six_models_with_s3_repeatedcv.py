"""Six-model comparison on the with-s3 clinical label (LR, RF, SVM, XGBoost,
1D-CNN, LSTM).

Same subject cohort / features / label as the with-s3 nested Optuna run:
    target = TUG>=12 OR FSST>=15 OR BERG<52 OR DGI<=19
             OR base(velocity)<1.0 OR s3(velocity)<1.0
    (31 normal / 40 impaired, 71 subjects)

Traditional models (LR, RF, SVM, XGB) use the 3 subject-level features; the deep
models (1D-CNN, LSTM) use the padded per-segment sequences (71 x 100 x 3) from
the cached array. Every model is scored with an *identical, paired* repeated
subject-level StratifiedKFold(5): on each repeat all six models share the same
folds. Threshold is chosen by fold-train Youden J (all models, same rule); AUC is
threshold independent. Fold-internal median-impute + RobustScaler only (no leak).

Outputs -> analysis_outputs/with_s3_six_model_repeatedcv/
  summary.csv, repeat_metrics.csv, subject_consensus_metrics.csv,
  <model>_consensus_roc.png, <model>_consensus_confusion_percent.png,
  all_models_consensus_roc.png, metadata.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis_outputs" / "final_training_conditions_nested_model_comparison_100rep"
TABLE_PATH = SRC / "subject_feature_table.csv"
CACHE_PATH = SRC / "subject_all_segments_final3_cache.npz"
OUT_DIR = ROOT / "analysis_outputs" / "with_s3_six_model_repeatedcv"

FEATURES = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
MAX_LEN = 100
N_SPLITS = 5
REPEATS = 100
BASE_SEED = 20260728
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ML_MODELS = ["LR", "RF", "SVM", "XGB"]
DL_MODELS = ["CNN1D", "LSTM"]
ALL_MODELS = ML_MODELS + DL_MODELS


# ----------------------------------------------------------------------------- models
def make_ml(name: str, seed: int) -> Pipeline:
    base = [("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]
    if name == "LR":
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", C=1.0,
            random_state=seed,
        )
    elif name == "RF":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    elif name == "SVM":
        clf = SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=seed,
        )
    elif name == "XGB":
        clf = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=seed,
            verbosity=0,
        )
    else:
        raise ValueError(name)
    return Pipeline([*base, ("clf", clf)])


class GaitLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        return self.fc(self.drop(hn[-1])).squeeze(1)


class GaitCNN1D(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x, lengths=None):
        return self.fc(self.conv(x.permute(0, 2, 1)).squeeze(-1)).squeeze(1)


DL_CLASSES = {"CNN1D": GaitCNN1D, "LSTM": GaitLSTM}


# ----------------------------------------------------------------------------- data
def load_data():
    table = pd.read_csv(TABLE_PATH).sort_values("cache_index").reset_index(drop=True)
    cache = np.load(CACHE_PATH)
    x_agg = table[FEATURES].to_numpy(np.float32)          # (71, 3)
    x_seq = cache["x_seq"].astype(np.float32)             # (71, 100, 3)
    seq_len = np.minimum(table["n_segments"].to_numpy(np.int64), MAX_LEN)
    y = table["target"].to_numpy(int)
    subject_ids = table["subject_id"].astype(str).to_numpy()
    return x_agg, x_seq, seq_len, y, subject_ids


# ----------------------------------------------------------------------------- cv utils
def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sen = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sen + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def ml_probs(name, tr, te, x, y, seq_len, seed):
    pipe = make_ml(name, seed)
    pipe.fit(x[tr], y[tr])
    return pipe.predict_proba(x[tr])[:, 1], pipe.predict_proba(x[te])[:, 1]


def dl_probs(name, tr, te, x_seq, y, seq_len, seed, epochs=150, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    imp, sc = SimpleImputer(strategy="median"), RobustScaler()
    flat_tr = x_seq[tr].reshape(-1, 3)
    imp.fit(flat_tr)
    sc.fit(imp.transform(flat_tr))

    def scale(idx):
        flat = x_seq[idx].reshape(-1, 3)
        out = sc.transform(imp.transform(flat)).reshape(len(idx), MAX_LEN, 3)
        return out.astype(np.float32)

    x_tr, x_te = scale(tr), scale(te)
    model = DL_CLASSES[name]().to(DEVICE)
    xt = torch.FloatTensor(x_tr).to(DEVICE)
    lt = torch.LongTensor(seq_len[tr])
    yt = torch.FloatTensor(y[tr]).to(DEVICE)
    n_pos = yt.sum()
    n_neg = len(yt) - n_pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(model(xt, lt), yt).backward()
        opt.step()

    def prob(x_sc, idx):
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.FloatTensor(x_sc).to(DEVICE), torch.LongTensor(seq_len[idx])
            )
        return torch.sigmoid(logits).cpu().numpy()

    return prob(x_tr, tr), prob(x_te, te)


# ----------------------------------------------------------------------------- run
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_agg, x_seq, seq_len, y, subject_ids = load_data()
    n = len(y)
    print(f"subjects={n} normal={(y==0).sum()} impaired={(y==1).sum()} device={DEVICE}")

    # accumulate probabilities per model per repeat, plus per-repeat / per-fold metrics
    repeat_rows = []
    fold_rows = []
    prob_accum = {m: np.zeros(n) for m in ALL_MODELS}  # sum of OOF probs across repeats

    for r in range(REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=BASE_SEED + r)
        folds = list(skf.split(x_agg, y))
        for m in ALL_MODELS:
            oof = np.zeros(n)
            fold_sen, fold_spec = [], []
            for fi, (tr, te) in enumerate(folds):
                seed = BASE_SEED + r * 100 + fi
                if m in ML_MODELS:
                    p_tr, p_te = ml_probs(m, tr, te, x_agg, y, seq_len, seed)
                else:
                    p_tr, p_te = dl_probs(m, tr, te, x_seq, y, seq_len, seed)
                oof[te] = p_te
                thr = youden_threshold(y[tr], p_tr)
                pred = (p_te >= thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
                sen = tp / (tp + fn) if tp + fn else 0.0
                spec = tn / (tn + fp) if tn + fp else 0.0
                # AUC on the single held-out fold (needs both classes present)
                fold_auc = (
                    float(roc_auc_score(y[te], p_te))
                    if len(np.unique(y[te])) == 2
                    else np.nan
                )
                fold_sen.append(sen)
                fold_spec.append(spec)
                fold_rows.append(
                    {
                        "model": m, "repeat": r, "fold": fi,
                        "auc": fold_auc, "sensitivity": sen, "specificity": spec,
                    }
                )
            prob_accum[m] += oof
            repeat_rows.append(
                {
                    "model": m,
                    "repeat": r,
                    "auc": float(roc_auc_score(y, oof)),
                    "sensitivity": float(np.mean(fold_sen)),
                    "specificity": float(np.mean(fold_spec)),
                }
            )
        print(f"  repeat {r + 1}/{REPEATS} done", flush=True)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)

    # summary: fold-level mean +/- std (across all N_SPLITS x REPEATS folds).
    # auc_oof/auc_oof_std keep the pooled-OOF-per-repeat estimate for reference.
    summary_rows = []
    for m in ALL_MODELS:
        part = fold_df[fold_df["model"] == m]
        rep = repeat_df[repeat_df["model"] == m]
        summary_rows.append(
            {
                "model": m,
                "auc_oof_mean": rep["auc"].mean(),
                "auc_oof_std_repeat": rep["auc"].std(ddof=1),
                "auc_fold_mean": part["auc"].mean(),
                "auc_fold_std": part["auc"].std(ddof=1),
                "sensitivity_fold_mean": part["sensitivity"].mean(),
                "sensitivity_fold_std": part["sensitivity"].std(ddof=1),
                "specificity_fold_mean": part["specificity"].mean(),
                "specificity_fold_std": part["specificity"].std(ddof=1),
                "n_folds": int(len(part)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("auc_oof_mean", ascending=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # subject-consensus (mean prob over repeats) -> ROC + confusion
    consensus_rows = []
    roc_all = {}
    for m in ALL_MODELS:
        prob = prob_accum[m] / REPEATS
        thr = youden_threshold(y, prob)
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        auc = float(roc_auc_score(y, prob))
        sen = tp / (tp + fn)
        spec = tn / (tn + fp)
        consensus_rows.append(
            {
                "model": m, "auc": auc, "sensitivity": sen, "specificity": spec,
                "accuracy": (tp + tn) / n, "threshold": thr,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }
        )
        fpr, tpr, _ = roc_curve(y, prob)
        roc_all[m] = (fpr, tpr, auc)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"{m}: subject-consensus ROC (with s3)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{m.lower()}_consensus_roc.png", dpi=180)
        plt.close(fig)

        cm = confusion_matrix(y, pred, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{pct[i, j]:.1f}%", ha="center", va="center",
                        color="white" if pct[i, j] >= 50 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Impaired"],
               yticklabels=["Normal", "Impaired"], xlabel="Predicted", ylabel="Actual",
               title=f"{m} Confusion Matrix")
        fig.colorbar(im, ax=ax, label="Row %")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{m.lower()}_consensus_confusion_percent.png", dpi=180)
        plt.close(fig)

    consensus = pd.DataFrame(consensus_rows).sort_values("auc", ascending=False)
    consensus.to_csv(OUT_DIR / "subject_consensus_metrics.csv", index=False)

    # combined ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in sorted(ALL_MODELS, key=lambda k: -roc_all[k][2]):
        fpr, tpr, auc = roc_all[m]
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{m} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Six-model subject-consensus ROC (with s3 label)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "all_models_consensus_roc.png", dpi=180)
    plt.close(fig)

    metadata = {
        "label_rule": ("TUG>=12 OR FSST>=15 OR BERG<52 OR DGI<=19 "
                       "OR base(velocity)<1.0 OR s3(velocity)<1.0"),
        "subjects": int(n), "normal": int((y == 0).sum()),
        "impaired": int((y == 1).sum()),
        "features_ml": FEATURES, "dl_input": "per-segment sequence (max 100 x 3)",
        "models": ALL_MODELS, "cv": f"subject-level StratifiedKFold({N_SPLITS}), paired",
        "repeats": REPEATS, "threshold": "fold-train Youden J (all models)",
        "device": str(DEVICE),
        "note": ("All six models share identical paired folds each repeat. "
                 "Deep models fixed-architecture (no per-fold Optuna); tree/linear "
                 "models fixed sensible hyperparameters. AUC is threshold-independent."),
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Summary (mean over repeats) ===")
    print(summary.to_string(index=False))
    print("\n=== Subject consensus ===")
    print(consensus.to_string(index=False))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    run()
