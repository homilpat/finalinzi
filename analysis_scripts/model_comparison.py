"""
모델 비교: LR, SVM, RF, GBM, XGB, Voting, Stacking, LSTM, CNN-1D
[기준] 전통 ML + 딥러닝 모두 subject 71명 단위로 학습/평가

수정 사항:
  1. 글로벌 스케일러 제거 → 전통 ML은 pipeline 내부에서만 스케일(train fold only)
  2. DL: fold 안에서 X_subj[tr] 기준 IMP+SC fit → test에 transform만
  3. threshold: OOF 전체에서 Youden 최적화 금지
     → fold마다 train 예측값으로 Youden → 그 임계값을 test fold에 적용 → fold별 Sen/Spec 평균
저장: analysis_outputs/model_comparison/
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch {torch.__version__}  device={DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("[skip] PyTorch 없음 → LSTM, CNN1D 건너뜀")

# ── 경로 ────────────────────────────────────────────────────────
TABLE_CSV    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/subject_features_with_clinical.csv"))
OUT_DIR      = ROOT / "analysis_outputs" / "model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST3    = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]
MAX_LEN  = 100
N_SPLITS = 5
SEED     = 42

# ── 1. 데이터 로드 ───────────────────────────────────────────────
print("\n[1] 데이터 로드")
table = pd.read_csv(TABLE_CSV)
clin  = pd.read_csv(CLINICAL_CSV, encoding='utf-8-sig')[['subject_id', 'motor_impairment_score']]
clin['clinical_target'] = (pd.to_numeric(clin['motor_impairment_score'], errors='coerce') >= 0.5).astype(int)
table = table.merge(clin, on='subject_id', how='inner').dropna(subset=BEST3).reset_index(drop=True)

subject_ids = table.groupby('subject_id')['clinical_target'].first().reset_index()
subject_ids.columns = ['subject_id', 'target']
n_subj   = len(subject_ids)
y_subj   = subject_ids['target'].astype(int).to_numpy()
sid_list = subject_ids['subject_id'].to_numpy()
print(f"  window={len(table)}  subject={n_subj}  normal={(y_subj==0).sum()}  impaired={(y_subj==1).sum()}")

# ── 2. 피처 행렬 구성 (raw, 스케일 없음) ────────────────────────
print("\n[2] Subject-level 피처 / 시퀀스 구성")

# 전통 ML: 71 × 3 (raw, 스케일 pipeline 안에서만)
X_subj = np.array([
    table[table['subject_id'] == sid][BEST3].median().to_numpy()
    for sid in sid_list
], dtype=np.float32)

# DL: 71 × MAX_LEN × 3 (raw, fold 안에서 스케일)
X_seq      = np.zeros((n_subj, MAX_LEN, 3), dtype=np.float32)
seq_lengths = np.zeros(n_subj, dtype=np.int64)
for i, sid in enumerate(sid_list):
    rows = table[table['subject_id'] == sid][BEST3].to_numpy()
    n = min(len(rows), MAX_LEN)
    seq_lengths[i] = n
    X_seq[i, :n] = rows[:n]

print(f"  X_subj {X_subj.shape}  X_seq {X_seq.shape}  avg_len={seq_lengths.mean():.1f}")


# ── 3. 유틸 ─────────────────────────────────────────────────────
def youden_thr_on(y, p):
    """y / p 기반 Youden 최적 임계값 — train fold에서만 호출"""
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        j = (tp/(tp+fn) if tp+fn else 0.) + (tn/(tn+fp) if tn+fp else 0.) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def fold_metrics(y_te, p_te, thr):
    """test fold에 고정 임계값 적용 → AUC, Sen, Spec"""
    pred = (p_te >= thr).astype(int)
    if len(np.unique(y_te)) < 2:
        return None
    auc  = float(roc_auc_score(y_te, p_te))
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {
        'auc':  auc,
        'sen':  float(tp/(tp+fn)) if tp+fn else 0.,
        'spec': float(tn/(tn+fp)) if tn+fp else 0.,
        'thr':  thr,
    }


def run_cv(get_prob_fn, X, y, use_seq=False):
    """
    5-fold StratifiedKFold CV 실행
    get_prob_fn(tr, te, X, y) → (p_tr, p_te)
    fold마다 train Youden → test fold 평가
    반환: OOF prob (71,), fold별 metrics list
    """
    skf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    data = X if not use_seq else X
    oof  = np.zeros(n_subj)
    fold_rows = []

    for tr, te in skf.split(data, y):
        p_tr, p_te = get_prob_fn(tr, te, X, y)
        oof[te] = p_te

        thr = youden_thr_on(y[tr], p_tr)          # ← train fold에서만 threshold 결정
        m   = fold_metrics(y[te], p_te, thr)
        if m:
            fold_rows.append(m)

    auc_oof = float(roc_auc_score(y, oof))
    fpr, tpr, _ = roc_curve(y, oof)

    # OOF 전체 혼동행렬은 fold-mean threshold 사용
    mean_thr = float(np.mean([r['thr'] for r in fold_rows]))
    pred_oof = (oof >= mean_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_oof, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    return {
        'auc_oof':  auc_oof,
        'sen_mean': float(np.mean([r['sen']  for r in fold_rows])),
        'sen_std':  float(np.std( [r['sen']  for r in fold_rows])),
        'spec_mean':float(np.mean([r['spec'] for r in fold_rows])),
        'spec_std': float(np.std( [r['spec'] for r in fold_rows])),
        'thr_mean': mean_thr,
        'cm':  cm,
        'fpr': fpr,
        'tpr': tpr,
        'oof': oof,
    }


# ── 4. 플롯 ─────────────────────────────────────────────────────
MODEL_COLORS = {
    'LR':'#1f77b4','SVM':'#ff7f0e','RF':'#2ca02c','GBM':'#d62728',
    'XGB':'#9467bd','Voting':'#8c564b','Stacking':'#e377c2',
    'LSTM':'#7f7f7f','CNN1D':'#bcbd22',
}

def save_cm(cm, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    labels = [['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            pct = cm[i,j]/cm.sum()*100
            col = 'white' if cm[i,j] > cm.max()*0.6 else 'black'
            ax.text(j,i,f'{labels[i][j]}\n{cm[i,j]} ({pct:.1f}%)',
                    ha='center',va='center',fontsize=12,color=col,fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Normal','Pred: Impaired'])
    ax.set_yticklabels(['True: Normal','True: Impaired'])
    ax.set_title(f'Confusion Matrix — {name}', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f'confusion_matrix_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_roc_single(fpr, tpr, auc, name):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr,tpr,color=MODEL_COLORS.get(name,'#333'),linewidth=2.5,label=f'AUC={auc:.3f}')
    ax.plot([0,1],[0,1],'k--',linewidth=1)
    ax.set_xlabel('1 - Specificity (FPR)',fontsize=11)
    ax.set_ylabel('Sensitivity (TPR)',fontsize=11)
    ax.set_title(f'ROC Curve — {name}',fontsize=12)
    ax.legend(loc='lower right',fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f'roc_curve_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_roc_all(results):
    fig, ax = plt.subplots(figsize=(8,7))
    for name, r in results.items():
        ax.plot(r['fpr'],r['tpr'],color=MODEL_COLORS.get(name),linewidth=2,
                label=f'{name}  AUC={r["auc_oof"]:.3f}')
    ax.plot([0,1],[0,1],'k--',linewidth=1,label='Random')
    ax.set_xlabel('1 - Specificity (FPR)',fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)',fontsize=12)
    ax.set_title('ROC Curve Comparison\n(Subject-level n=71, fold-train Youden threshold)',fontsize=12)
    ax.legend(loc='lower right',fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/'roc_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── 5. 전통 ML 파이프라인 ────────────────────────────────────────
def _base():
    return [('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())]

TRADITIONAL = {
    'LR':       lambda: Pipeline([*_base(), ('clf', LogisticRegression(
                    max_iter=3000, class_weight='balanced', solver='liblinear', C=1.0))]),
    'SVM':      lambda: Pipeline([*_base(), ('clf', SVC(
                    kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True))]),
    'RF':       lambda: Pipeline([*_base(), ('clf', RandomForestClassifier(
                    n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1))]),
    'GBM':      lambda: Pipeline([*_base(), ('clf', GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED))]),
    'XGB':      lambda: Pipeline([*_base(), ('clf', xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=3,
                    eval_metric='logloss', random_state=SEED, verbosity=0))]),
    'Voting':   lambda: Pipeline([*_base(), ('clf', VotingClassifier(voting='soft', estimators=[
                    ('lr',  LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear')),
                    ('rf',  RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)),
                    ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                               eval_metric='logloss', random_state=SEED, verbosity=0)),
                ]))]),
    'Stacking': lambda: Pipeline([*_base(), ('clf', StackingClassifier(cv=3,
                    final_estimator=LogisticRegression(max_iter=3000, solver='liblinear'),
                    estimators=[
                        ('rf',  RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)),
                        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                   eval_metric='logloss', random_state=SEED, verbosity=0)),
                        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)),
                    ]))]),
}


# ── 6. 딥러닝 모델 ───────────────────────────────────────────────
if TORCH_AVAILABLE:
    class GaitLSTM(nn.Module):
        def __init__(self, input_size=3, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc   = nn.Linear(hidden_size, 1)
        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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


# ── 7. get_prob 함수 래퍼 ────────────────────────────────────────
def make_ml_fn(make_pipe):
    """전통 ML: X_subj raw → pipeline 내부에서만 스케일"""
    def fn(tr, te, X, y):
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        return pipe.predict_proba(X[tr])[:,1], pipe.predict_proba(X[te])[:,1]
    return fn


def make_dl_fn(ModelClass, epochs=150, lr=1e-3):
    """DL: fold 안에서 X_seq[tr] 기준 IMP+SC 피팅 → test에 transform만"""
    def fn(tr, te, X_seq_raw, y):
        # fold 내 스케일 (누수 없음)
        imp_f = SimpleImputer(strategy='median')
        sc_f  = RobustScaler()
        flat_tr = X_seq_raw[tr].reshape(-1, 3)
        imp_f.fit(flat_tr); sc_f.fit(imp_f.transform(flat_tr))

        def scale(idx):
            flat = X_seq_raw[idx].reshape(-1, 3)
            return sc_f.transform(imp_f.transform(flat)).reshape(len(idx), MAX_LEN, 3).astype(np.float32)

        X_tr_sc = scale(tr)
        X_te_sc = scale(te)

        model  = ModelClass().to(DEVICE)
        X_t    = torch.FloatTensor(X_tr_sc).to(DEVICE)
        l_t    = torch.LongTensor(seq_lengths[tr])
        y_t    = torch.FloatTensor(y[tr]).to(DEVICE)
        n_pos  = y_t.sum(); n_neg = len(y_t) - n_pos
        crit   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/max(n_pos,1)]).to(DEVICE))
        opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            crit(model(X_t, l_t), y_t).backward()
            opt.step()

        def prob(X_sc, idx):
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_sc).to(DEVICE),
                               torch.LongTensor(seq_lengths[idx]))
                return torch.sigmoid(logits).cpu().numpy()

        return prob(X_tr_sc, tr), prob(X_te_sc, te)
    return fn


# ── 8. 실험 실행 ─────────────────────────────────────────────────
RESULTS = {}
print(f"\n[3] 전통 ML (subject 71명, raw X_subj, fold-train Youden threshold)")
for name, make_pipe in TRADITIONAL.items():
    print(f"  {name:10s} ...", end='', flush=True)
    r = run_cv(make_ml_fn(make_pipe), X_subj, y_subj)
    RESULTS[name] = r
    save_cm(r['cm'], name)
    save_roc_single(r['fpr'], r['tpr'], r['auc_oof'], name)
    print(f"  AUC={r['auc_oof']:.3f}  Sen={r['sen_mean']:.3f}±{r['sen_std']:.3f}"
          f"  Spec={r['spec_mean']:.3f}±{r['spec_std']:.3f}  thr≈{r['thr_mean']:.3f}")

if TORCH_AVAILABLE:
    print(f"\n[4] 딥러닝 (subject 71명, fold-내 스케일, fold-train Youden threshold)")
    for name, ModelClass in [('LSTM', GaitLSTM), ('CNN1D', GaitCNN1D)]:
        print(f"  {name:10s} ...", end='', flush=True)
        r = run_cv(make_dl_fn(ModelClass), X_seq, y_subj, use_seq=True)
        RESULTS[name] = r
        save_cm(r['cm'], name)
        save_roc_single(r['fpr'], r['tpr'], r['auc_oof'], name)
        print(f"  AUC={r['auc_oof']:.3f}  Sen={r['sen_mean']:.3f}±{r['sen_std']:.3f}"
              f"  Spec={r['spec_mean']:.3f}±{r['spec_std']:.3f}  thr≈{r['thr_mean']:.3f}")


# ── 9. 통합 ROC 커브 ─────────────────────────────────────────────
print(f"\n[5] 통합 ROC 커브 저장")
save_roc_all(RESULTS)


# ── 10. 결과 저장 ─────────────────────────────────────────────────
print(f"\n[6] 결과 저장")
rows = [{
    'Model':       n,
    'AUC_OOF':     round(r['auc_oof'],  4),
    'Sen_mean':    round(r['sen_mean'], 4),
    'Sen_std':     round(r['sen_std'],  4),
    'Spec_mean':   round(r['spec_mean'],4),
    'Spec_std':    round(r['spec_std'], 4),
    'Threshold':   round(r['thr_mean'], 3),
} for n, r in RESULTS.items()]

df_res = pd.DataFrame(rows).sort_values('AUC_OOF', ascending=False).reset_index(drop=True)
df_res.to_csv(OUT_DIR/'model_comparison_results.csv', index=False)
print(df_res.to_string(index=False))
print("\n[완료]")
