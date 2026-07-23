"""
Service-matched model comparison.

Protocol:
  - Build every available 20s segment.
  - Within each 20s segment, compute 10s sliding-window features.
  - Aggregate those 10s windows into one feature row per 20s segment.
  - For each repeat, randomly select exactly one 20s segment per subject.
  - Evaluate models with subject-level StratifiedKFold.
  - Pick each fold threshold on train predictions only:
      sensitivity >= 0.80, then maximum specificity.

Outputs:
  analysis_outputs/single_20s_segment_model_comparison_100rep/
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from gait_axis_aligned_core import TARGET_FS_HZ, window_features

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception as exc:
    TORCH_OK = False
    DEVICE = None
    print(f"[DL skip] {exc}")


OUT_DIR = ROOT / "analysis_outputs" / "single_20s_segment_acc_qc_expanded_100rep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "single_20s_segment_feature_cache.npz"
LEGACY_CACHE_PATH = ROOT / "analysis_outputs" / "single_20s_segment_model_comparison_100rep" / "single_20s_segment_feature_cache.npz"

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
RAW_DIR = GAIT_PROJECT / "physionet_AWS"
V2_CSV = RAW_DIR / "strict_preprocessing_runs" / "strict_preprocessed_accgyro_v2" / "gait_features_strict_20s_accgyro_v2.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))

FS = int(TARGET_FS_HZ)
WIN20 = int(20 * FS)
SUB_WIN = int(10 * FS)
SUB_STEP = int(2 * FS)
N_SUBWINDOWS = (WIN20 - SUB_WIN) // SUB_STEP + 1
N_SPLITS = 5
N_REPEATS = 100
TARGET_SENSITIVITY = 0.80

BASE = ["v_harmonic_ratio", "ap_harmonic_ratio", "v_stride_freq_hz", "ap_spec_entropy", "v_jerk_rms"]
AGG_FEATURES = [f"{name}_{stat}" for name in BASE for stat in ("median", "iqr")]
QC_LIMITS = {
    "v_stride_freq_hz_iqr": 0.12,
    "ap_spec_entropy_iqr": 0.12,
    "v_harmonic_ratio_iqr": 0.50,
    "ap_harmonic_ratio_iqr": 0.50,
}
SELECT_K = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def clinical_expanded_target(clinical: pd.DataFrame) -> pd.Series:
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


def parse_hea(sid: str) -> dict:
    lines = (RAW_DIR / f"{sid}.hea").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    n, ch = int(parts[3]), int(parts[1])
    gains, baselines = [], []
    for line in lines[1 : 1 + ch]:
        match = re.match(r".*?([0-9.]+)\((-?\d+)\)/", line.split()[2])
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2)))
    return {"n": n, "ch": ch, "gains": np.array(gains[:3]), "baselines": np.array(baselines[:3])}


def read_20s(sid: str, start_sec: float, hea: dict) -> np.ndarray | None:
    dat = RAW_DIR / f"{sid}.dat"
    if not dat.exists():
        return None
    start = int(round(start_sec * FS))
    end = start + WIN20
    if end > hea["n"]:
        return None
    raw = np.memmap(dat, dtype="<i2", mode="r", shape=(hea["n"], hea["ch"]))
    segment = raw[start:end, :3].astype(float)
    return (segment - hea["baselines"]) / hea["gains"]


def segment_features(vmlap_20s: np.ndarray) -> tuple[dict, np.ndarray] | None:
    rows = []
    for start in range(0, WIN20 - SUB_WIN + 1, SUB_STEP):
        sub = vmlap_20s[start : start + SUB_WIN]
        if len(sub) < int(0.8 * SUB_WIN):
            continue
        try:
            feat = window_features(sub)
            rows.append({name: float(feat[name]) for name in BASE})
        except Exception:
            continue
    if len(rows) < 2:
        return None
    arr = pd.DataFrame(rows)
    seq = np.full((N_SUBWINDOWS, len(BASE)), np.nan, dtype=np.float32)
    seq[: len(arr), :] = arr[BASE].to_numpy(np.float32)
    out = {}
    for feat in BASE:
        vals = arr[feat].dropna()
        out[f"{feat}_median"] = float(vals.median()) if len(vals) else np.nan
        out[f"{feat}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan
    return out, seq


def aggregate_sequence(x_seq: np.ndarray) -> np.ndarray:
    rows = []
    for seq in x_seq:
        values = []
        for col in range(seq.shape[1]):
            finite = seq[:, col][np.isfinite(seq[:, col])]
            values.extend([
                float(np.median(finite)) if len(finite) else np.nan,
                float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if len(finite) else np.nan,
            ])
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def add_acc_qc(meta: pd.DataFrame, x_agg: np.ndarray) -> pd.DataFrame:
    out = meta.copy()
    feature_idx = {name: idx for idx, name in enumerate(AGG_FEATURES)}
    passed = np.ones(len(out), dtype=bool)
    for name, limit in QC_LIMITS.items():
        values = x_agg[:, feature_idx[name]]
        out[f"qc_{name}"] = values
        passed &= np.isfinite(values) & (values <= limit)
    out["acc_qc_pass"] = passed
    return out


def build_or_load_segment_cache() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        meta = pd.DataFrame(cached["meta"].tolist())
        x_agg = cached["x_agg"].astype(np.float32)
        x_seq = cached["x_seq"].astype(np.float32)
        return add_acc_qc(meta, x_agg), x_agg, x_seq
    if LEGACY_CACHE_PATH.exists():
        cached = np.load(LEGACY_CACHE_PATH, allow_pickle=True)
        meta = pd.DataFrame(cached["meta"].tolist())
        x_seq = cached["x_seq"].astype(np.float32)
        x_agg = aggregate_sequence(x_seq)
        meta = add_acc_qc(meta, x_agg)
        np.savez_compressed(CACHE_PATH, meta=meta.to_dict("records"), x_agg=x_agg, x_seq=x_seq)
        meta.to_csv(OUT_DIR / "single_20s_segment_cache_index.csv", index=False, encoding="utf-8-sig")
        return meta, x_agg, x_seq

    print("[1] Build 20s segment cache from raw PhysioNet files")
    v2 = pd.read_csv(V2_CSV)
    v2["start_sec"] = pd.to_numeric(v2["start_sec"], errors="coerce")
    v2 = v2.dropna(subset=["start_sec"])

    clinical = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    clinical["target"] = clinical_expanded_target(clinical)
    labels = clinical[["subject_id", "target"]].drop_duplicates("subject_id")
    v2 = v2.merge(labels, on="subject_id", how="inner")

    meta_rows, agg_rows, seq_rows = [], [], []
    for sid, part in v2.groupby("subject_id", sort=True):
        try:
            hea = parse_hea(str(sid))
        except Exception as exc:
            print(f"  [skip] {sid}: {exc}")
            continue
        n_ok = 0
        for seg_idx, row in part.reset_index(drop=True).iterrows():
            vmlap = read_20s(str(sid), float(row["start_sec"]), hea)
            if vmlap is None:
                continue
            result = segment_features(vmlap)
            if result is None:
                continue
            agg, seq = result
            if not np.isfinite([agg[f] for f in AGG_FEATURES]).all():
                continue
            meta_rows.append({
                "segment_id": f"{sid}__{seg_idx}",
                "subject_id": sid,
                "group": row.get("group", ""),
                "target": int(row["target"]),
                "start_sec": float(row["start_sec"]),
            })
            agg_rows.append([agg[f] for f in AGG_FEATURES])
            seq_rows.append(seq)
            n_ok += 1
        print(f"  {sid}: {n_ok} valid 20s segments")

    meta = pd.DataFrame(meta_rows)
    x_agg = np.asarray(agg_rows, dtype=np.float32)
    x_seq = np.asarray(seq_rows, dtype=np.float32)
    meta = add_acc_qc(meta, x_agg)
    np.savez_compressed(CACHE_PATH, meta=meta.to_dict("records"), x_agg=x_agg, x_seq=x_seq)
    meta.to_csv(OUT_DIR / "single_20s_segment_cache_index.csv", index=False, encoding="utf-8-sig")
    return meta, x_agg, x_seq


def base_steps(select: bool = False) -> list:
    steps = [("imp", SimpleImputer(strategy="median"))]
    if select:
        steps.append(("select", SelectKBest(score_func=f_classif, k=min(SELECT_K, len(AGG_FEATURES)))))
    steps.append(("sc", RobustScaler()))
    return steps


def make_lr(seed: int) -> Pipeline:
    return Pipeline([*base_steps(select=True), ("m", LogisticRegression(
        C=0.5, max_iter=1000, class_weight="balanced", random_state=seed))])


def make_rf(seed: int) -> Pipeline:
    return Pipeline([*base_steps(), ("m", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=seed, n_jobs=1))])


def make_xgb(seed: int) -> Pipeline:
    return Pipeline([*base_steps(), ("m", XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.9,
        eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0))])


def make_models(seed: int) -> dict[str, Pipeline]:
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=seed, n_jobs=1)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, min_child_weight=5,
                        reg_alpha=0.5, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0)
    svm = SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=seed)
    return {
        "LR_final": make_lr(seed),
        "SVM": Pipeline([*base_steps(), ("m", svm)]),
        "RF": make_rf(seed),
        "GBM": Pipeline([*base_steps(), ("m", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, min_samples_leaf=5,
            subsample=0.7, random_state=seed))]),
        "XGB": make_xgb(seed),
        "Voting": Pipeline([*base_steps(), ("m", VotingClassifier(
            voting="soft", n_jobs=1, estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)]))]),
        "Stacking": Pipeline([*base_steps(), ("m", StackingClassifier(
            cv=3, n_jobs=1, final_estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)]))]),
    }


def threshold_for_min_sensitivity(y_true: np.ndarray, prob: np.ndarray) -> float:
    vals = np.unique(prob[np.isfinite(prob)])
    if len(vals) == 0:
        return 0.5
    mids = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else np.array([])
    candidates = np.r_[vals.min() - 1e-9, mids, vals.max() + 1e-9]
    best_t, best_spec = float(candidates[0]), -np.inf
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= TARGET_SENSITIVITY and spec > best_spec:
            best_t, best_spec = float(threshold), spec
    return best_t


def metric_row(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    out = metric_row_from_pred(y_true, prob, pred)
    out["threshold"] = float(threshold)
    return out


def metric_row_from_pred(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


if TORCH_OK:
    class LSTMNet(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 24, batch_first=True)
            self.drop = nn.Dropout(0.35)
            self.fc = nn.Linear(24, 1)
        def forward(self, x, lengths):
            _, (h, _) = self.lstm(pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False))
            return self.fc(self.drop(h[-1])).squeeze(1)

    class CNN1DNet(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 24, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(24),
                nn.Conv1d(24, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
            )
            self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(24, 1))
        def forward(self, x, lengths):
            return self.fc(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(1)


def norm_seq(x_seq: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    sc = RobustScaler()
    flat = x_seq[train_idx].reshape(-1, x_seq.shape[-1])
    sc.fit(imp.fit_transform(flat))
    def apply(idx: np.ndarray) -> np.ndarray:
        shaped = x_seq[idx].reshape(-1, x_seq.shape[-1])
        out = sc.transform(imp.transform(shaped)).reshape(len(idx), x_seq.shape[1], x_seq.shape[2])
        return out.astype(np.float32)
    return apply(train_idx), apply(test_idx)


def train_dl_fold(cls, x_tr, y_tr, x_te, seed: int) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    model = cls(x_tr.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    n_val = max(2, int(len(y_tr) * 0.2))
    val_idx = rng.choice(len(y_tr), n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(y_tr)), val_idx)
    pos = max(1, int((y_tr[train_idx] == 1).sum()))
    neg = max(1, int((y_tr[train_idx] == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    lengths_tr = np.full(len(y_tr), x_tr.shape[1], dtype=np.int64)
    lengths_te = np.full(len(x_te), x_te.shape[1], dtype=np.int64)
    def tensor(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype, device=DEVICE)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(120):
        model.train()
        opt.zero_grad()
        loss = crit(model(tensor(x_tr[train_idx]), tensor(lengths_tr[train_idx], torch.long)),
                    tensor(y_tr[train_idx].astype(np.float32)))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.BCEWithLogitsLoss()(
                model(tensor(x_tr[val_idx]), tensor(lengths_tr[val_idx], torch.long)),
                tensor(y_tr[val_idx].astype(np.float32)),
            ).item()
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= 12:
            break
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(tensor(x_tr), tensor(lengths_tr, torch.long))).cpu().numpy()
        p_te = torch.sigmoid(model(tensor(x_te), tensor(lengths_te, torch.long))).cpu().numpy()
    return p_tr, p_te


def run_comparison(meta: pd.DataFrame, x_agg_all: np.ndarray, x_seq_all: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    subject_ids = np.array(sorted(meta["subject_id"].unique()))
    by_subject = {sid: meta.index[meta["subject_id"].eq(sid)].to_numpy() for sid in subject_ids}

    model_names = list(make_models(0).keys()) + (["LSTM", "CNN1D"] if TORCH_OK else [])
    for repeat in range(N_REPEATS):
        if repeat % 10 == 0:
            print(f"  repeat {repeat}/{N_REPEATS}", flush=True)
        rng = np.random.default_rng(20260724 + repeat)
        chosen = np.array([rng.choice(by_subject[sid]) for sid in subject_ids])
        chosen_meta = meta.iloc[chosen].reset_index(drop=True)
        x_agg = x_agg_all[chosen]
        x_seq = x_seq_all[chosen]
        y = chosen_meta["target"].to_numpy(int)
        sids = chosen_meta["subject_id"].to_numpy()
        splits = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=810000 + repeat).split(x_agg, y))

        for name, model in make_models(900000 + repeat).items():
            train_auc, test_auc, thresholds = [], [], []
            oof = np.zeros(len(y), dtype=float)
            oof_pred = np.zeros(len(y), dtype=int)
            oof_threshold = np.zeros(len(y), dtype=float)
            for fold, (tr, te) in enumerate(splits):
                assert len(set(sids[tr]) & set(sids[te])) == 0
                model.fit(x_agg[tr], y[tr])
                p_tr = model.predict_proba(x_agg[tr])[:, 1]
                p_te = model.predict_proba(x_agg[te])[:, 1]
                thr = threshold_for_min_sensitivity(y[tr], p_tr)
                thresholds.append(thr)
                oof[te] = p_te
                oof_pred[te] = (p_te >= thr).astype(int)
                oof_threshold[te] = thr
                train_auc.append(roc_auc_score(y[tr], p_tr))
                if len(np.unique(y[te])) == 2:
                    test_auc.append(roc_auc_score(y[te], p_te))
            threshold = float(np.median(thresholds))
            row = metric_row_from_pred(y, oof, oof_pred)
            row.update({
                "model": name, "repeat": repeat, "train_auc": float(np.mean(train_auc)),
                "test_auc_fold_mean": float(np.mean(test_auc)), "threshold": threshold,
            })
            metrics.append(row)
            pred_df = chosen_meta[["subject_id", "segment_id", "target", "start_sec"]].copy()
            pred_df["model"] = name
            pred_df["repeat"] = repeat
            pred_df["probability"] = oof
            pred_df["threshold"] = oof_threshold
            pred_df["prediction"] = oof_pred
            predictions.append(pred_df)

        if TORCH_OK:
            for name, cls in [("LSTM", LSTMNet), ("CNN1D", CNN1DNet)]:
                train_auc, test_auc, thresholds = [], [], []
                oof = np.zeros(len(y), dtype=float)
                oof_pred = np.zeros(len(y), dtype=int)
                oof_threshold = np.zeros(len(y), dtype=float)
                for fold, (tr, te) in enumerate(splits):
                    x_tr, x_te = norm_seq(x_seq, tr, te)
                    p_tr, p_te = train_dl_fold(cls, x_tr, y[tr], x_te, seed=930000 + repeat * 10 + fold)
                    thr = threshold_for_min_sensitivity(y[tr], p_tr)
                    thresholds.append(thr)
                    oof[te] = p_te
                    oof_pred[te] = (p_te >= thr).astype(int)
                    oof_threshold[te] = thr
                    train_auc.append(roc_auc_score(y[tr], p_tr))
                    if len(np.unique(y[te])) == 2:
                        test_auc.append(roc_auc_score(y[te], p_te))
                threshold = float(np.median(thresholds))
                row = metric_row_from_pred(y, oof, oof_pred)
                row.update({
                    "model": name, "repeat": repeat, "train_auc": float(np.mean(train_auc)),
                    "test_auc_fold_mean": float(np.mean(test_auc)), "threshold": threshold,
                })
                metrics.append(row)
                pred_df = chosen_meta[["subject_id", "segment_id", "target", "start_sec"]].copy()
                pred_df["model"] = name
                pred_df["repeat"] = repeat
                pred_df["probability"] = oof
                pred_df["threshold"] = oof_threshold
                pred_df["prediction"] = oof_pred
                predictions.append(pred_df)

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model")
        .agg(
            n_repeats=("repeat", "count"),
            train_auc_mean=("train_auc", "mean"),
            test_auc_mean=("auc", "mean"),
            test_auc_std=("auc", "std"),
            auc_ci_lo=("auc", lambda x: x.quantile(0.025)),
            auc_ci_hi=("auc", lambda x: x.quantile(0.975)),
            gap=("train_auc", lambda x: np.nan),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_mean=("f1", "mean"),
            threshold_median=("threshold", "median"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
        )
        .reset_index()
    )


def save_plots(summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    roc_dir = OUT_DIR / "roc_curves"
    cm_dir = OUT_DIR / "confusion_matrices"
    roc_dir.mkdir(exist_ok=True)
    cm_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    for model, part in predictions.groupby("model"):
        y = part["target"].to_numpy(int)
        p = part["probability"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=1.8, label=f"{model} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC pooled over 100 random single-20s repeats")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_dir / "roc_all_models_pooled_100rep.png", dpi=180)
    plt.close(fig)

    cm_rows = []
    for model, part in predictions.groupby("model"):
        y = part["target"].to_numpy(int)
        pred = part["prediction"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cm_rows.append({"model": model, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(f"{model} confusion matrix")
        fig.tight_layout()
        fig.savefig(cm_dir / f"confusion_matrix_{model}.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrix_pooled_100rep.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    print("=== Single 20s segment per subject model comparison ===")
    print(f"Target sensitivity on train fold: >= {TARGET_SENSITIVITY:.2f}")
    meta, x_agg, x_seq = build_or_load_segment_cache()
    print(f"segments={len(meta)} subjects={meta['subject_id'].nunique()} normal_subjects={(meta.drop_duplicates('subject_id')['target'] == 0).sum()} impaired_subjects={(meta.drop_duplicates('subject_id')['target'] == 1).sum()}")

    metrics, predictions = run_comparison(meta, x_agg, x_seq)
    metrics.to_csv(OUT_DIR / "metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "predictions_by_repeat.csv", index=False, encoding="utf-8-sig")

    summary = summarize(metrics)
    summary["gap"] = summary["train_auc_mean"] - summary["test_auc_mean"]
    summary = summary.sort_values(
        ["sensitivity_mean", "test_auc_mean", "specificity_mean"],
        ascending=[False, False, False],
    )
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    save_plots(summary, predictions)

    printable = summary[[
        "model", "test_auc_mean", "test_auc_std", "auc_ci_lo", "auc_ci_hi",
        "sensitivity_mean", "specificity_mean", "recall_mean", "precision_mean",
        "accuracy_mean", "f1_mean", "gap", "threshold_median",
    ]]
    print(printable.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
