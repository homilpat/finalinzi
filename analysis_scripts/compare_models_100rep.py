"""
모델 비교 100-rep 버전

설계 원칙:
  - StratifiedKFold(5) × 100 seeds → per-seed 5-fold mean → 100개 AUC 분포
  - 데이터 누수 없음: scaler/imputer는 매 fold train 안에서만 fit
  - subject 단위 1행 (window median) → 같은 subject가 train/test 양쪽 절대 불가
  - 정규화: RF max_depth=5/min_leaf=5, GBM n=100/min_leaf=5/sub=0.7,
            XGB n=100/min_child=5/reg, DL dropout=0.35 + early stopping(patience=15)
  - DL: 100-rep 동일 적용 (data 소형이라 가능)

출력: analysis_outputs/model_comparison_100rep/
"""
from __future__ import annotations
import json, os, random, sys, warnings
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "model_comparison_100rep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               StackingClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import torch, torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    TORCH_OK = True
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"[DL] device={DEVICE}")
except Exception as e:
    TORCH_OK = False; DEVICE = None
    print(f"[DL skip] {e}")

# ── 설정 ──────────────────────────────────────────────────
FEATURES  = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
TABLE_CSV = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLIN_CSV  = Path("C:/Users/whdgu/Desktop/파이널 보행 프로젝트/physionet_AWS"
                 "/strict_preprocessing_runs/clinical_motor_label_modeling"
                 "/subject_features_with_clinical.csv")
THR       = 0.50
N_SPLITS  = 5
N_SEEDS   = 100
MAX_LEN   = 120


# ── 유틸 ──────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    if TORCH_OK:
        torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def make_label(df):
    for c in ["TUG","FSST","BERG","DGI","base(velocity)","s3(velocity)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return ((df["TUG"]>=12)|(df["FSST"]>=15)|(df["BERG"]<52)|
            (df["DGI"]<=19)|(df["base(velocity)"]<1.0)|(df["s3(velocity)"]<1.0)).astype(int)


def fold_auc(y_tr, y_te, prob_tr, prob_te):
    tr_auc = float(roc_auc_score(y_tr, prob_tr)) if len(np.unique(y_tr))==2 else np.nan
    te_auc = float(roc_auc_score(y_te, prob_te)) if len(np.unique(y_te))==2 else np.nan
    return tr_auc, te_auc


# ── 데이터 로드 ───────────────────────────────────────────
def load():
    tbl  = pd.read_csv(TABLE_CSV)
    clin = pd.read_csv(CLIN_CSV, encoding="utf-8-sig")
    clin["label"] = make_label(clin)
    lbl  = clin[["subject_id","label"]].drop_duplicates("subject_id")
    tbl  = tbl.merge(lbl, on="subject_id", how="inner")
    tbl  = tbl.dropna(subset=FEATURES).reset_index(drop=True)

    subj = (tbl.groupby("subject_id")[FEATURES+["label"]]
               .agg({**{f:"median" for f in FEATURES}, "label":"first"})
               .reset_index().dropna(subset=FEATURES).reset_index(drop=True))

    sids = subj["subject_id"].to_numpy()
    X    = subj[FEATURES].to_numpy(np.float32)
    y    = subj["label"].to_numpy(int)

    # 시퀀스 (DL용)
    Xseq = np.zeros((len(sids), MAX_LEN, len(FEATURES)), np.float32)
    lens = np.zeros(len(sids), np.int64)
    for i, sid in enumerate(sids):
        rows = tbl.loc[tbl["subject_id"]==sid, FEATURES].to_numpy(np.float32)
        n = min(len(rows), MAX_LEN)
        Xseq[i,:n] = rows[:n]; lens[i] = max(n,1)

    return sids, X, Xseq, lens, y


# ── ML 모델 정의 (정규화 포함) ────────────────────────────
def base():
    return [("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]

def make_rf(n=300, seed=42):
    return RandomForestClassifier(n_estimators=n, max_depth=5, min_samples_leaf=5,
                                   class_weight="balanced", random_state=seed, n_jobs=-1)

def make_xgb(n=100, seed=42):
    return XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=2,
                          min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
                          subsample=0.8, colsample_bytree=0.9,
                          eval_metric="logloss", random_state=seed,
                          n_jobs=max(1,os.cpu_count()or 1), verbosity=0)

def make_gbm(seed=42):
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                                       min_samples_leaf=5, subsample=0.7, random_state=seed)

def get_ml_models(seed=42):
    return {
        "LR":      Pipeline([*base(), ("m", LogisticRegression(C=1.0, class_weight="balanced",
                                        max_iter=3000, solver="liblinear", random_state=seed))]),
        "SVM":     Pipeline([*base(), ("m", SVC(C=1.0, gamma="scale", kernel="rbf",
                                        class_weight="balanced", probability=True, random_state=seed))]),
        "RF":      Pipeline([*base(), ("m", make_rf(300, seed))]),
        "GBM":     Pipeline([*base(), ("m", make_gbm(seed))]),
        "XGB":     Pipeline([*base(), ("m", make_xgb(100, seed))]),
        "Voting":  Pipeline([*base(), ("m", VotingClassifier(voting="soft", n_jobs=-1, estimators=[
                                ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=3000, solver="liblinear", random_state=seed)),
                                ("rf", make_rf(200, seed)),
                                ("xgb", make_xgb(100, seed)),]))]),
        "Stacking":Pipeline([*base(), ("m", StackingClassifier(cv=3, n_jobs=-1,
                                final_estimator=LogisticRegression(max_iter=3000, solver="liblinear"),
                                estimators=[("rf",make_rf(200,seed)),("xgb",make_xgb(100,seed)),
                                            ("svm",SVC(C=1.0,gamma="scale",kernel="rbf",
                                                        class_weight="balanced",probability=True,random_state=seed))]))]),
    }


# ── DL 모델 ───────────────────────────────────────────────
if TORCH_OK:
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 32, batch_first=True)
            self.drop = nn.Dropout(0.35)
            self.fc   = nn.Linear(32, 1)
        def forward(self, x, l):
            _, (h,_) = self.lstm(pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False))
            return self.fc(self.drop(h[-1])).squeeze(1)

    class CNN1DNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(3,32,5,padding=2), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32,32,3,padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(32,1))
        def forward(self, x, l):
            return self.fc(self.net(x.transpose(1,2)).squeeze(-1)).squeeze(1)


def norm_seq(Xseq, lens, tr, te):
    flat = np.vstack([Xseq[i,:lens[i]] for i in tr])
    imp = SimpleImputer(strategy="median"); sc = RobustScaler()
    sc.fit(imp.fit_transform(flat))
    def apply(idx):
        out = np.zeros((len(idx), Xseq.shape[1], Xseq.shape[2]), np.float32)
        for ri,si in enumerate(idx):
            n = lens[si]
            out[ri,:n] = sc.transform(imp.transform(Xseq[si,:n])).astype(np.float32)
        return out
    return apply(tr), apply(te)


def train_dl_fold(cls, Xtr, ytr, Ltr, Xte, Lte, seed, max_ep=150, patience=15):
    set_seed(seed)
    model = cls().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
    # val split for early stopping (20% of train)
    n_val = max(2, int(len(ytr)*0.2))
    rng   = np.random.default_rng(seed)
    vi    = rng.choice(len(ytr), n_val, replace=False)
    ti    = np.setdiff1d(np.arange(len(ytr)), vi)
    pos   = max(1,int((ytr[ti]==1).sum())); neg = max(1,int((ytr[ti]==0).sum()))
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], device=DEVICE))

    def t(a, dt=torch.float32): return torch.tensor(a, dtype=dt, device=DEVICE)
    best_loss, best_ep, best_state = float("inf"), 0, None

    for ep in range(max_ep):
        model.train(); opt.zero_grad()
        loss = crit(model(t(Xtr[ti]), t(Ltr[ti], torch.long)), t(ytr[ti].astype(np.float32)))
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.BCEWithLogitsLoss()(
                model(t(Xtr[vi]), t(Ltr[vi], torch.long)),
                t(ytr[vi].astype(np.float32))).item()
        if vl < best_loss - 1e-4:
            best_loss, best_ep = vl, ep
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        elif ep - best_ep >= patience:
            break

    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(t(Xtr), t(Ltr, torch.long))).cpu().numpy()
        ep = torch.sigmoid(model(t(Xte), t(Lte, torch.long))).cpu().numpy()
    return tp, ep


# ── 100-rep CV ────────────────────────────────────────────
def run_100rep(sids, X, Xseq, lens, y):
    ml_names = list(get_ml_models(0).keys())
    dl_names = ["LSTM", "CNN1D"] if TORCH_OK else []
    all_names = ml_names + dl_names

    # seed별 집계 {name: list of values over seeds}
    tr_aucs = {n: [] for n in all_names}
    te_aucs = {n: [] for n in all_names}
    sens_list = {n: [] for n in all_names}   # OOF sensitivity per seed
    spec_list = {n: [] for n in all_names}   # OOF specificity per seed

    for seed in range(N_SEEDS):
        if seed % 10 == 0:
            print(f"  seed {seed}/{N_SEEDS} ...", flush=True)
        set_seed(seed)
        skf    = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))

        # ML
        ml_models = get_ml_models(seed)
        for name, model in ml_models.items():
            fold_tr, fold_te = [], []
            oof = np.zeros(len(y))
            for tr, te in splits:
                assert len(set(sids[tr]) & set(sids[te])) == 0, "subject overlap!"
                model.fit(X[tr], y[tr])
                tr_p = model.predict_proba(X[tr])[:,1]
                te_p = model.predict_proba(X[te])[:,1]
                oof[te] = te_p
                ta, ea = fold_auc(y[tr], y[te], tr_p, te_p)
                if not np.isnan(ta): fold_tr.append(ta)
                if not np.isnan(ea): fold_te.append(ea)
            tr_aucs[name].append(np.mean(fold_tr))
            te_aucs[name].append(np.mean(fold_te))
            # OOF 기준 sens/spec
            pred = (oof >= THR).astype(int)
            tn,fp,fn,tp_ = confusion_matrix(y, pred, labels=[0,1]).ravel()
            sens_list[name].append(tp_/(tp_+fn) if tp_+fn else 0.)
            spec_list[name].append(tn/(tn+fp) if tn+fp else 0.)

        # DL
        if TORCH_OK:
            for name, cls in [("LSTM", LSTMNet), ("CNN1D", CNN1DNet)]:
                fold_tr, fold_te = [], []
                oof = np.zeros(len(y))
                for tr, te in splits:
                    assert len(set(sids[tr]) & set(sids[te])) == 0
                    Xtr, Xte = norm_seq(Xseq, lens, tr, te)
                    tp, ep2 = train_dl_fold(cls, Xtr, y[tr], lens[tr], Xte, lens[te], seed=seed*10)
                    oof[te] = ep2
                    ta, ea = fold_auc(y[tr], y[te], tp, ep2)
                    if not np.isnan(ta): fold_tr.append(ta)
                    if not np.isnan(ea): fold_te.append(ea)
                tr_aucs[name].append(np.mean(fold_tr))
                te_aucs[name].append(np.mean(fold_te))
                pred = (oof >= THR).astype(int)
                tn,fp,fn,tp_ = confusion_matrix(y, pred, labels=[0,1]).ravel()
                sens_list[name].append(tp_/(tp_+fn) if tp_+fn else 0.)
                spec_list[name].append(tn/(tn+fp) if tn+fp else 0.)

    return tr_aucs, te_aucs, sens_list, spec_list


# ── 결과 집계 및 저장 ─────────────────────────────────────
def summarize(tr_aucs, te_aucs, sens_list, spec_list):
    rows = []
    for name in tr_aucs:
        tr   = np.array(tr_aucs[name])
        te   = np.array(te_aucs[name])
        sens = np.array(sens_list[name])
        spec = np.array(spec_list[name])
        rows.append({
            "model":         name,
            "train_auc":     round(float(tr.mean()), 4),
            "test_auc_mean": round(float(te.mean()), 4),
            "test_auc_std":  round(float(te.std()),  4),
            "auc_ci_lo":     round(float(np.percentile(te,  2.5)), 4),
            "auc_ci_hi":     round(float(np.percentile(te, 97.5)), 4),
            "gap":           round(float(tr.mean()-te.mean()), 4),
            "sens_mean":     round(float(sens.mean()), 4),
            "sens_std":      round(float(sens.std()),  4),
            "spec_mean":     round(float(spec.mean()), 4),
            "spec_std":      round(float(spec.std()),  4),
            "n_seeds":       len(te),
        })
    df = pd.DataFrame(rows).sort_values("test_auc_mean", ascending=False).reset_index(drop=True)
    return df


def save_roc_all(sids, X, Xseq, lens, y):
    """seed=0 OOF로 ROC 커브 그리기"""
    set_seed(0)
    skf    = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)
    splits = list(skf.split(X, y))
    ml_models = get_ml_models(0)

    fig, ax = plt.subplots(figsize=(8,7))
    for name, model in ml_models.items():
        oof = np.zeros(len(y))
        for tr, te in splits:
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:,1]
        fpr, tpr, _ = roc_curve(y, oof)
        auc = roc_auc_score(y, oof)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} {auc:.3f}")

    if TORCH_OK:
        for name, cls in [("LSTM", LSTMNet), ("CNN1D", CNN1DNet)]:
            oof = np.zeros(len(y))
            for tr, te in splits:
                Xtr, Xte = norm_seq(Xseq, lens, tr, te)
                _, ep2 = train_dl_fold(cls, Xtr, y[tr], lens[tr], Xte, lens[te], seed=0)
                oof[te] = ep2
            fpr, tpr, _ = roc_curve(y, oof)
            auc = roc_auc_score(y, oof)
            ax.plot(fpr, tpr, lw=1.8, linestyle="--", label=f"{name} {auc:.3f}")

    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (seed=0 OOF) — n={len(y)}")
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"roc_all_models.png", dpi=180)
    plt.close(fig)
    print(f"ROC 저장: {OUT_DIR/'roc_all_models.png'}")


def main():
    print(f"=== 100-rep 5-fold 모델 비교 ===")
    print(f"Seeds: {N_SEEDS}  Folds: {N_SPLITS}  Threshold: {THR}")
    sids, X, Xseq, lens, y = load()
    print(f"Subjects={len(y)}  normal={(y==0).sum()}  impaired={(y==1).sum()}\n")

    tr_aucs, te_aucs, sens_list, spec_list = run_100rep(sids, X, Xseq, lens, y)

    df = summarize(tr_aucs, te_aucs, sens_list, spec_list)
    df.to_csv(OUT_DIR/"comparison_100rep.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*115)
    print(f"{'모델':10s}  {'Train':7s}  {'Test AUC':9s}  {'±std':7s}  {'95% CI':16s}  {'Gap':7s}  "
          f"{'Sens':6s}  {'±std':6s}  {'Spec':6s}  {'±std':6s}")
    print("="*115)
    for _, r in df.iterrows():
        print(f"  {r['model']:8s}  {r['train_auc']:.4f}   {r['test_auc_mean']:.4f}    "
              f"±{r['test_auc_std']:.4f}  [{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]  "
              f"{r['gap']:+.4f}  {r['sens_mean']:.4f}  ±{r['sens_std']:.4f}  "
              f"{r['spec_mean']:.4f}  ±{r['spec_std']:.4f}")
    print("="*115)

    save_roc_all(sids, X, Xseq, lens, y)

    # json 저장
    (OUT_DIR/"result.json").write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {OUT_DIR/'comparison_100rep.csv'}")


if __name__ == "__main__":
    main()
