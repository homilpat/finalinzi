"""
모델 비교 (정규화 버전) - train=1.0 방지

원본 compare_final_speed_or6_models.py 대비 변경 사항:
  RF  : max_depth=5, min_samples_leaf=5 추가  (원본: 제한 없음)
  GBM : n_estimators 250→100, min_samples_leaf=5, subsample=0.7 추가
  XGB : n_estimators 300→100, min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0
  Voting/Stacking 내 RF/XGB 컴포넌트도 동일 수준 정규화
  CNN1D/LSTM : early stopping (patience=15, validation 20% from train fold)
출력: analysis_outputs/model_comparison_regularized/
"""
from __future__ import annotations

import json, os, random, sys, warnings
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "model_comparison_regularized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
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
except Exception as exc:
    TORCH_AVAILABLE = False
    DEVICE = None
    print(f"[skip] PyTorch unavailable: {exc}")

FEATURES   = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
TABLE_CSV  = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = Path("C:/Users/whdgu/Desktop/파이널 보행 프로젝트/physionet_AWS/strict_preprocessing_runs/clinical_motor_label_modeling/subject_features_with_clinical.csv")
THRESHOLD  = 0.50
N_SPLITS   = 5
SEED       = 42
MAX_LEN    = 120


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_label(clin):
    for col in ["TUG","FSST","BERG","DGI","base(velocity)","s3(velocity)"]:
        clin[col] = pd.to_numeric(clin[col], errors="coerce")
    return ((clin["TUG"]>=12)|(clin["FSST"]>=15)|(clin["BERG"]<52)|
            (clin["DGI"]<=19)|(clin["base(velocity)"]<1.0)|(clin["s3(velocity)"]<1.0)).astype(int)


def load_data():
    table   = pd.read_csv(TABLE_CSV)
    clinical = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    clinical["target"] = make_label(clinical)
    labels  = clinical[["subject_id","target"]].drop_duplicates("subject_id")
    table   = table.merge(labels, on="subject_id", how="inner", suffixes=("_old",""))
    table   = table.drop(columns=[c for c in table.columns if c.endswith("_old")])
    table   = table.dropna(subset=FEATURES).reset_index(drop=True)

    subject_table = (
        table.groupby("subject_id")[FEATURES+["target"]]
        .agg({**{f:"median" for f in FEATURES}, "target":"first"})
        .reset_index().dropna(subset=FEATURES).reset_index(drop=True)
    )
    subject_ids = subject_table["subject_id"].to_numpy()
    x_subject   = subject_table[FEATURES].to_numpy(dtype=np.float32)
    y           = subject_table["target"].to_numpy(dtype=int)

    x_seq   = np.zeros((len(subject_ids), MAX_LEN, len(FEATURES)), dtype=np.float32)
    lengths = np.zeros(len(subject_ids), dtype=np.int64)
    for idx, sid in enumerate(subject_ids):
        rows = table.loc[table["subject_id"]==sid, FEATURES].to_numpy(dtype=np.float32)
        n = min(len(rows), MAX_LEN)
        x_seq[idx, :n] = rows[:n]; lengths[idx] = max(n,1)

    return subject_table, x_subject, x_seq, lengths, y


def metrics(y_true, prob, thr=THRESHOLD):
    pred = (prob>=thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true,prob)) if len(np.unique(y_true))==2 else np.nan,
        "sensitivity": float(tp/(tp+fn)) if tp+fn else 0.,
        "specificity": float(tn/(tn+fp)) if tn+fp else 0.,
        "accuracy": float(accuracy_score(y_true,pred)),
        "f1": float(f1_score(y_true,pred,zero_division=0)),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }


def base_steps():
    return [("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]


def make_rf(n=300):
    return RandomForestClassifier(
        n_estimators=n, max_depth=5, min_samples_leaf=5,  # ← 정규화 핵심
        class_weight="balanced", random_state=SEED, n_jobs=-1)


def make_xgb(n=100, cpu=1):
    return XGBClassifier(
        n_estimators=n, learning_rate=0.05, max_depth=2,
        min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,  # ← 정규화 핵심
        subsample=0.8, colsample_bytree=0.9,
        eval_metric="logloss", random_state=SEED, n_jobs=cpu, verbosity=0)


def traditional_models():
    cpu = max(1, os.cpu_count() or 1)
    return {
        "LR": Pipeline([*base_steps(),
            ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                         max_iter=3000, solver="liblinear", random_state=SEED))]),
        "SVM": Pipeline([*base_steps(),
            ("model", SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced",
                          probability=True, random_state=SEED))]),
        "RF": Pipeline([*base_steps(), ("model", make_rf(300))]),
        "GBM": Pipeline([*base_steps(),
            ("model", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                min_samples_leaf=5, subsample=0.7, random_state=SEED))]),  # ← 정규화 핵심
        "XGB": Pipeline([*base_steps(), ("model", make_xgb(100, cpu))]),
        "Voting": Pipeline([*base_steps(),
            ("model", VotingClassifier(voting="soft", n_jobs=-1, estimators=[
                ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                          max_iter=3000, solver="liblinear", random_state=SEED)),
                ("rf", make_rf(200)),
                ("xgb", make_xgb(100, cpu)),
            ]))]),
        "Stacking": Pipeline([*base_steps(),
            ("model", StackingClassifier(cv=3, n_jobs=-1,
                final_estimator=LogisticRegression(max_iter=3000, solver="liblinear"),
                estimators=[("rf", make_rf(200)), ("xgb", make_xgb(100,cpu)),
                            ("svm", SVC(C=1.0, gamma="scale", kernel="rbf",
                                        class_weight="balanced", probability=True, random_state=SEED))]))]),
    }


def assert_no_overlap(sids, tr, te):
    overlap = set(sids[tr]) & set(sids[te])
    if overlap: raise RuntimeError(f"Subject leakage: {sorted(overlap)[:5]}")


def evaluate_ml(name, model, x, y, splits, sids):
    oof = np.zeros(len(y))
    tr_rows, te_rows = [], []
    for fold, (tr, te) in enumerate(splits, 1):
        assert_no_overlap(sids, tr, te)
        model.fit(x[tr], y[tr])
        tr_rows.append({"model":name,"fold":fold, **metrics(y[tr], model.predict_proba(x[tr])[:,1])})
        te_prob = model.predict_proba(x[te])[:,1]
        oof[te] = te_prob
        te_rows.append({"model":name,"fold":fold, **metrics(y[te], te_prob)})
    return {"oof_prob":oof, "train":tr_rows, "test":te_rows}


if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, input_size=3, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.35)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            return self.fc(self.dropout(h[-1])).squeeze(1)

    class CNN1DNet(nn.Module):
        def __init__(self, input_size=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_size, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(32, 1))
        def forward(self, x, lengths):
            return self.fc(self.net(x.transpose(1,2)).squeeze(-1)).squeeze(1)


def transform_seq(x_seq, lengths, tr, te):
    flat = np.vstack([x_seq[i, :lengths[i]] for i in tr])
    imp = SimpleImputer(strategy="median"); sc = RobustScaler()
    sc.fit(imp.fit_transform(flat))
    def apply(idx):
        out = np.zeros((len(idx), x_seq.shape[1], x_seq.shape[2]), dtype=np.float32)
        for ri, si in enumerate(idx):
            n = lengths[si]
            out[ri,:n] = sc.transform(imp.transform(x_seq[si,:n])).astype(np.float32)
        return out
    return apply(tr), apply(te)


def train_torch_fold(model_cls, x_tr, y_tr, l_tr, x_te, l_te, seed, max_epochs=200, patience=15):
    set_seed(seed)
    # train에서 20% val split (early stopping용)
    n_val = max(2, int(len(y_tr)*0.2))
    val_idx = np.random.choice(len(y_tr), n_val, replace=False)
    trn_idx = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    model = model_cls().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
    pos = max(1, int((y_tr[trn_idx]==1).sum())); neg = max(1, int((y_tr[trn_idx]==0).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], dtype=torch.float32, device=DEVICE))

    def to_t(arr): return torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    def to_l(arr): return torch.tensor(arr, dtype=torch.long, device=DEVICE)

    best_val, best_ep, best_state = float("inf"), 0, None
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        loss = criterion(model(to_t(x_tr[trn_idx]), to_l(l_tr[trn_idx])),
                         to_t(y_tr[trn_idx].astype(np.float32)))
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.BCEWithLogitsLoss()(
                model(to_t(x_tr[val_idx]), to_l(l_tr[val_idx])),
                to_t(y_tr[val_idx].astype(np.float32))).item()
        if val_loss < best_val - 1e-4:
            best_val, best_ep = val_loss, ep
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        elif ep - best_ep >= patience:
            break

    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    model.eval()
    with torch.no_grad():
        tr_prob = torch.sigmoid(model(to_t(x_tr), to_l(l_tr))).cpu().numpy()
        te_prob = torch.sigmoid(model(to_t(x_te), to_l(l_te))).cpu().numpy()
    print(f"    early stop ep={best_ep+1}/{max_epochs}")
    return tr_prob, te_prob


def evaluate_torch(name, model_cls, x_seq, lengths, y, splits, sids):
    oof = np.zeros(len(y))
    tr_rows, te_rows = [], []
    for fold, (tr, te) in enumerate(splits, 1):
        assert_no_overlap(sids, tr, te)
        x_tr, x_te = transform_seq(x_seq, lengths, tr, te)
        tr_prob, te_prob = train_torch_fold(model_cls, x_tr, y[tr], lengths[tr], x_te, lengths[te], seed=SEED+fold)
        oof[te] = te_prob
        tr_rows.append({"model":name,"fold":fold, **metrics(y[tr], tr_prob)})
        te_rows.append({"model":name,"fold":fold, **metrics(y[te], te_prob)})
    return {"oof_prob":oof, "train":tr_rows, "test":te_rows}


def save_cm(name, y, prob):
    pred = (prob>=THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{'TNFP'.split()[0] if i==0 else 'FNTP'.split()[0]}\n{cm[i,j]}",
                    ha="center",va="center",fontweight="bold",fontsize=12,
                    color="white" if cm[i,j]>cm.max()*0.55 else "black")
    labels = [["TN","FP"],["FN","TP"]]
    for i in range(2):
        for j in range(2):
            ax.texts[i*2+j].set_text(f"{labels[i][j]}\n{cm[i,j]}")
    ax.set_xticks([0,1],["Pred normal","Pred impaired"])
    ax.set_yticks([0,1],["True normal","True impaired"])
    ax.set_title(f"Confusion matrix - {name} (thr={THRESHOLD:.2f})")
    fig.tight_layout(); fig.savefig(OUT_DIR/f"confusion_matrix_{name}.png", dpi=180); plt.close(fig)


def save_roc(name, y, prob):
    fpr, tpr, _ = roc_curve(y, prob); auc = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr,tpr,lw=2.3,label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC - {name}"); ax.grid(alpha=0.25); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT_DIR/f"roc_curve_{name}.png", dpi=180); plt.close(fig)
    return fpr, tpr, auc


def summarize(tr_rows, te_rows, oof_m):
    tr = pd.DataFrame(tr_rows); te = pd.DataFrame(te_rows)
    out = {"folds": int(len(te))}
    for m in ["auc","sensitivity","specificity","accuracy","f1"]:
        out[f"train_{m}_mean"] = float(tr[m].mean())
        out[f"test_{m}_mean"]  = float(te[m].mean())
        out[f"gap_{m}"]        = float(tr[m].mean()-te[m].mean())
    out.update({f"oof_{k}":v for k,v in oof_m.items()
                if k in ["auc","sensitivity","specificity","accuracy","f1","tn","fp","fn","tp"]})
    return out


def main():
    set_seed(SEED)
    subject_table, x_subject, x_seq, lengths, y = load_data()
    sids   = subject_table["subject_id"].to_numpy()
    splits = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(x_subject, y))

    print(f"Output: {OUT_DIR}")
    print(f"Subjects={len(y)} normal={(y==0).sum()} impaired={(y==1).sum()}")
    print(f"Threshold={THRESHOLD}  [정규화 버전]")

    results = {}
    for name, model in traditional_models().items():
        print(f"\n[ML] {name}")
        results[name] = evaluate_ml(name, model, x_subject, y, splits, sids)

    if TORCH_AVAILABLE:
        for name, cls in [("LSTM",LSTMNet),("CNN1D",CNN1DNet)]:
            print(f"\n[DL] {name}")
            results[name] = evaluate_torch(name, cls, x_seq, lengths, y, splits, sids)

    summary_rows, all_tr, all_te, roc_items = [], [], [], {}
    for name, res in results.items():
        oof_m = metrics(y, res["oof_prob"])
        all_tr.extend(res["train"]); all_te.extend(res["test"])
        s = summarize(res["train"], res["test"], oof_m); s["model"] = name
        summary_rows.append(s)
        save_cm(name, y, res["oof_prob"])
        roc_items[name] = save_roc(name, y, res["oof_prob"])
        print(f"  OOF AUC={oof_m['auc']:.3f} sens={oof_m['sensitivity']:.3f} spec={oof_m['specificity']:.3f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("oof_auc", ascending=False)
    pd.DataFrame(all_tr).to_csv(OUT_DIR/"train_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_te).to_csv(OUT_DIR/"test_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR/"model_comparison_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8,7))
    for name, (fpr,tpr,auc) in sorted(roc_items.items(), key=lambda x:x[1][2], reverse=True):
        ax.plot(fpr,tpr,lw=2,label=f"{name} AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC comparison - regularized models")
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR/"roc_all_models.png", dpi=180); plt.close(fig)

    print("\n=== 결과 요약 ===")
    print(summary_df[["model","oof_auc","oof_sensitivity","oof_specificity","gap_auc","train_auc_mean","test_auc_mean"]].to_string(index=False))

    # 원본 vs 정규화 비교 출력
    orig = pd.read_csv(ROOT/"analysis_outputs"/"final_model_comparison_speed_or6"/"model_comparison_summary.csv")
    print("\n=== 원본 vs 정규화 비교 (train AUC) ===")
    for _, row in summary_df.iterrows():
        orig_row = orig[orig["model"]==row["model"]]
        orig_tr = orig_row["train_auc_mean"].values[0] if len(orig_row) else float("nan")
        print(f"  {row['model']:10s}  원본 train={orig_tr:.3f}  정규화 train={row['train_auc_mean']:.3f}"
              f"  OOF AUC  원본={orig_row['oof_auc'].values[0] if len(orig_row) else float('nan'):.3f}"
              f"  정규화={row['oof_auc']:.3f}")


if __name__ == "__main__":
    main()
