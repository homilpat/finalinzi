"""
모델 비교: LR, SVM, RF, GBM, XGB, Voting, Stacking, LSTM, CNN-1D
[기준] 전통 ML + 딥러닝 모두 subject 71명 단위로 학습/평가
  - 전통 ML 입력: subject당 window feature 중앙값 → 71 × 3 행렬
  - LSTM/CNN 입력: subject당 window sequence → 71 × MAX_LEN × 3
  - 분할: StratifiedKFold(5-fold) on subjects
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

# Subject 목록 (order 고정)
subject_ids = table.groupby('subject_id')['clinical_target'].first().reset_index()
subject_ids.columns = ['subject_id', 'target']
n_subj  = len(subject_ids)
y_subj  = subject_ids['target'].astype(int).to_numpy()
sid_list = subject_ids['subject_id'].to_numpy()

print(f"  window={len(table)}  subject={n_subj}  normal={int((y_subj==0).sum())}  impaired={int((y_subj==1).sum())}")

# ── 2. Subject-level 피처 행렬 (전통 ML용) ──────────────────────
# 각 subject의 window feature들의 중앙값 → 71 × 3
print("\n[2] Subject-level 피처 행렬 구성 (71명 × 3피처)")
X_subj = np.array([
    table[table['subject_id'] == sid][BEST3].median().to_numpy()
    for sid in sid_list
], dtype=np.float32)
print(f"  X_subj shape: {X_subj.shape}")

# ── 3. Subject-level 시퀀스 (LSTM/CNN용) ────────────────────────
# 각 subject의 window feature rows → (MAX_LEN, 3) zero-padded
print("\n[3] Subject-level 시퀀스 구성 (71명 × 100 × 3)")
X_seq = np.zeros((n_subj, MAX_LEN, 3), dtype=np.float32)
seq_lengths = np.zeros(n_subj, dtype=np.int64)

for i, sid in enumerate(sid_list):
    rows = table[table['subject_id'] == sid][BEST3].to_numpy()
    n = min(len(rows), MAX_LEN)
    seq_lengths[i] = n
    X_seq[i, :n] = rows[:n]

print(f"  X_seq shape: {X_seq.shape}  avg_len={seq_lengths.mean():.1f}")

# 스케일: X_subj 기준으로 피팅 → X_subj, X_seq 동시 적용
_imp = SimpleImputer(strategy='median')
_sc  = RobustScaler()
X_subj_sc = _sc.fit_transform(_imp.fit_transform(X_subj))

X_seq_flat = X_seq.reshape(-1, 3)
X_seq_sc   = _sc.transform(_imp.transform(X_seq_flat)).reshape(n_subj, MAX_LEN, 3).astype(np.float32)


# ── 4. 유틸 ─────────────────────────────────────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        j = (tp / (tp + fn) if tp + fn else 0.) + (tn / (tn + fp) if tn + fp else 0.) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def compute_metrics(y_true, prob):
    thr  = youden_thr(y_true, prob)
    auc  = float(roc_auc_score(y_true, prob))
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sen  = float(tp / (tp + fn)) if tp + fn else 0.
    spec = float(tn / (tn + fp)) if tn + fp else 0.
    fpr, tpr, _ = roc_curve(y_true, prob)
    cm   = np.array([[tn, fp], [fn, tp]])
    return {'auc': auc, 'sen': sen, 'spec': spec, 'thr': thr, 'cm': cm, 'fpr': fpr, 'tpr': tpr}


# ── 5. 플롯 ─────────────────────────────────────────────────────
MODEL_COLORS = {
    'LR': '#1f77b4', 'SVM': '#ff7f0e', 'RF': '#2ca02c',
    'GBM': '#d62728', 'XGB': '#9467bd', 'Voting': '#8c564b',
    'Stacking': '#e377c2', 'LSTM': '#7f7f7f', 'CNN1D': '#bcbd22',
}

def save_cm(cm, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm.sum() * 100
            col = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{cm[i,j]} ({pct:.1f}%)',
                    ha='center', va='center', fontsize=12, color=col, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Normal', 'Pred: Impaired'])
    ax.set_yticklabels(['True: Normal', 'True: Impaired'])
    ax.set_title(f'Confusion Matrix — {name}', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'confusion_matrix_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_roc_single(fpr, tpr, auc, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, '#333'), linewidth=2.5,
            label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
    ax.set_title(f'ROC Curve — {name}', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'roc_curve_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_roc_all(results):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, r in results.items():
        ax.plot(r['fpr'], r['tpr'], color=MODEL_COLORS.get(name, None), linewidth=2,
                label=f'{name}  AUC={r["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax.set_title('ROC Curve Comparison\n(Subject-level, 5-fold StratifiedKFold, n=71)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'roc_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()


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
                nn.Conv1d(32, 64, kernel_size=5, padding=2),          nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x, lengths=None):
            return self.fc(self.conv(x.permute(0, 2, 1)).squeeze(-1)).squeeze(1)

    def train_dl(ModelClass, X_seq, lengths, y_subj, tr_idx, epochs=150, lr=1e-3):
        model = ModelClass().to(DEVICE)
        X_t = torch.FloatTensor(X_seq[tr_idx]).to(DEVICE)
        l_t = torch.LongTensor(lengths[tr_idx])
        y_t = torch.FloatTensor(y_subj[tr_idx]).to(DEVICE)
        n_pos = y_t.sum(); n_neg = len(y_t) - n_pos
        crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE))
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            crit(model(X_t, l_t), y_t).backward()
            opt.step()
        return model

    def predict_dl(model, X_seq, lengths, te_idx):
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_seq[te_idx]).to(DEVICE),
                           torch.LongTensor(lengths[te_idx]))
            return torch.sigmoid(logits).cpu().numpy()


# ── 7. 전통 ML 파이프라인 ────────────────────────────────────────
def _base():
    return [('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())]

TRADITIONAL = {
    'LR': lambda: Pipeline([*_base(),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced',
                                   solver='liblinear', C=1.0))]),
    'SVM': lambda: Pipeline([*_base(),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                    class_weight='balanced', probability=True))]),
    'RF': lambda: Pipeline([*_base(),
        ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                        random_state=SEED, n_jobs=-1))]),
    'GBM': lambda: Pipeline([*_base(),
        ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                            max_depth=3, random_state=SEED))]),
    'XGB': lambda: Pipeline([*_base(),
        ('clf', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                   eval_metric='logloss', random_state=SEED, verbosity=0))]),
    'Voting': lambda: Pipeline([*_base(),
        ('clf', VotingClassifier(voting='soft', estimators=[
            ('lr',  LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear')),
            ('rf',  RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)),
            ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                       eval_metric='logloss', random_state=SEED, verbosity=0)),
        ]))]),
    'Stacking': lambda: Pipeline([*_base(),
        ('clf', StackingClassifier(cv=3,
            final_estimator=LogisticRegression(max_iter=3000, solver='liblinear'),
            estimators=[
                ('rf',  RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                           eval_metric='logloss', random_state=SEED, verbosity=0)),
                ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)),
            ]))]),
}

# ── 8. 실험 실행 (모두 subject 71명 기준) ───────────────────────
RESULTS = {}
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# --- 전통 ML: subject-level feature 행렬 (71 × 3) ---
print(f"\n[4] 전통 ML (5-fold StratifiedKFold on subjects, n=71)")
for name, make_fn in TRADITIONAL.items():
    print(f"  {name:10s} ... ", end='', flush=True)
    oof = np.zeros(n_subj)

    for tr, te in skf.split(X_subj_sc, y_subj):
        pipe = make_fn()
        pipe.fit(X_subj_sc[tr], y_subj[tr])
        oof[te] = pipe.predict_proba(X_subj_sc[te])[:, 1]

    m = compute_metrics(y_subj, oof)
    RESULTS[name] = m
    save_cm(m['cm'], name)
    save_roc_single(m['fpr'], m['tpr'], m['auc'], name)
    print(f"AUC={m['auc']:.3f}  Sen={m['sen']:.3f}  Spec={m['spec']:.3f}  thr={m['thr']:.3f}")

# --- 딥러닝: subject-level sequence (71 × MAX_LEN × 3) ---
if TORCH_AVAILABLE:
    print(f"\n[5] 딥러닝 (5-fold StratifiedKFold on subjects, n=71)")
    for name, ModelClass in [('LSTM', GaitLSTM), ('CNN1D', GaitCNN1D)]:
        print(f"  {name:10s} ... ", end='', flush=True)
        oof = np.zeros(n_subj)

        for tr, te in skf.split(X_seq_sc, y_subj):
            model = train_dl(ModelClass, X_seq_sc, seq_lengths, y_subj, tr)
            oof[te] = predict_dl(model, X_seq_sc, seq_lengths, te)

        m = compute_metrics(y_subj, oof)
        RESULTS[name] = m
        save_cm(m['cm'], name)
        save_roc_single(m['fpr'], m['tpr'], m['auc'], name)
        print(f"AUC={m['auc']:.3f}  Sen={m['sen']:.3f}  Spec={m['spec']:.3f}  thr={m['thr']:.3f}")

# ── 9. 통합 ROC 커브 ─────────────────────────────────────────────
print(f"\n[6] 통합 ROC 커브 저장")
save_roc_all(RESULTS)

# ── 10. 결과 CSV ──────────────────────────────────────────────────
print(f"\n[7] 결과 저장")
rows = [{'Model': n, 'AUC': round(r['auc'], 4),
         'Sensitivity': round(r['sen'], 4), 'Specificity': round(r['spec'], 4),
         'Threshold': round(r['thr'], 3)}
        for n, r in RESULTS.items()]
df_res = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
df_res.to_csv(OUT_DIR / 'model_comparison_results.csv', index=False)
print(df_res.to_string(index=False))

print(f"\n=== 저장 파일 ({OUT_DIR}) ===")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")

print("\n[완료]")
