"""
PhysioNet 원본 Control/Fall 라벨 vs OR 재라벨 (TUG/FSST/BERG/DGI/속도) 상관관계 분석 및 시각화
- 재라벨 결과를 clinical_OR_relabel.csv 로 저장
- 라벨 일치/불일치 시각화
- 각 OR 기준별 기여도 시각화
- 임상 지표 분포 비교
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOCA"))
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 한글 폰트 설정
_font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
for _fc in _font_candidates:
    if any(_fc.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rc("font", family=_fc)
        break
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── 경로 ────────────────────────────────────────────────────────
SUBWIN_CSV   = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
OUT_DIR      = ROOT / "analysis_outputs" / "label_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RELABEL_CSV  = OUT_DIR / "clinical_OR_relabel.csv"

# ── OR 라벨 기준 ─────────────────────────────────────────────────
OR_CRITERIA = {
    "TUG≥12":        lambda d: d["TUG"] >= 12,
    "FSST≥15":       lambda d: d["FSST"] >= 15,
    "BERG<52":       lambda d: d["BERG"] < 52,
    "DGI≤19":        lambda d: d["DGI"] <= 19,
    "base_v<1.0":    lambda d: d["base(velocity)"] < 1.0,
    "s3_v<1.0":      lambda d: d["s3(velocity)"] < 1.0,
}

# ── 데이터 로드 ──────────────────────────────────────────────────
print("[1] 데이터 로드")
sub  = pd.read_csv(SUBWIN_CSV)
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")

for col in ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)"]:
    clin[col] = pd.to_numeric(clin[col], errors="coerce")

# subject-level 임상 테이블 (subwindow CSV에 있는 group 컬럼 포함)
sub_subj = sub[["subject_id", "group"]].drop_duplicates("subject_id")
clin_subj = clin[["subject_id", "TUG", "FSST", "BERG", "DGI",
                   "base(velocity)", "s3(velocity)"]].drop_duplicates("subject_id")

df = sub_subj.merge(clin_subj, on="subject_id", how="inner")
print(f"  매칭 subject: {len(df)}명")

# ── OR 기준 각각 적용 ────────────────────────────────────────────
print("\n[2] OR 기준별 플래그 생성")
for crit_name, crit_fn in OR_CRITERIA.items():
    df[crit_name] = crit_fn(df).astype(int)
    print(f"  {crit_name:12s}: {df[crit_name].sum()}명 양성")

df["OR_label"]       = (df[[c for c in OR_CRITERIA]].sum(axis=1) > 0).astype(int)
df["original_label"] = (df["group"] != "Control").astype(int)  # Control=0, Faller=1
df["n_criteria_met"] = df[[c for c in OR_CRITERIA]].sum(axis=1)

faller_name = df[df["original_label"]==1]["group"].iloc[0] if (df["original_label"]==1).any() else "Faller"
print(f"\n  원본 라벨 — Control: {(df['original_label']==0).sum()}명  {faller_name}: {(df['original_label']==1).sum()}명")
print(f"  OR 라벨   — 정상:    {(df['OR_label']==0).sum()}명  저하: {(df['OR_label']==1).sum()}명")

# ── 재라벨 CSV 저장 ──────────────────────────────────────────────
print(f"\n[3] 재라벨 CSV 저장: {RELABEL_CSV}")
save_cols = ["subject_id", "group", "original_label",
             "TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)",
             "TUG≥12", "FSST≥15", "BERG<52", "DGI≤19", "base_v<1.0", "s3_v<1.0",
             "n_criteria_met", "OR_label"]
df[save_cols].to_csv(RELABEL_CSV, index=False, encoding="utf-8-sig")
print(f"  저장 완료: {len(df)}명")

# ── 라벨 불일치 분석 ─────────────────────────────────────────────
print("\n[4] 라벨 일치/불일치 분석")
cross = pd.crosstab(df["group"], df["OR_label"],
                    rownames=["원본(Control/Fall)"], colnames=["OR 라벨(0=정상/1=저하)"])
print(cross)

match        = (df["original_label"] == df["OR_label"]).sum()
mismatch     = (df["original_label"] != df["OR_label"]).sum()
ctrl_to_imp  = ((df["original_label"]==0) & (df["OR_label"]==1)).sum()  # Control인데 OR=저하
fall_to_norm = ((df["original_label"]==1) & (df["OR_label"]==0)).sum()  # Fall인데 OR=정상

print(f"\n  일치: {match}명 ({match/len(df)*100:.1f}%)")
print(f"  불일치: {mismatch}명 ({mismatch/len(df)*100:.1f}%)")
print(f"    Control → OR저하: {ctrl_to_imp}명")
print(f"    Fall    → OR정상: {fall_to_norm}명")

# ════════════════════════════════════════════════════════════════
# 시각화 1: 라벨 비교 개요 (2×2 교차표 + 파이차트)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PhysioNet 원본 라벨 vs OR 재라벨 비교", fontsize=14, fontweight="bold")

# -- 1a: 교차표 히트맵
ct_vals = np.array([
    [(df["original_label"]==0)&(df["OR_label"]==0),
     (df["original_label"]==0)&(df["OR_label"]==1)],
    [(df["original_label"]==1)&(df["OR_label"]==0),
     (df["original_label"]==1)&(df["OR_label"]==1)],
], dtype=object)
ct_counts = np.array([[v.sum() for v in row] for row in ct_vals])

ax = axes[0]
im = ax.imshow(ct_counts, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(ct_counts[i, j]), ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if ct_counts[i, j] > ct_counts.max()*0.5 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["OR=정상(0)", "OR=저하(1)"], fontsize=10)
ax.set_yticklabels(["원본=Control", "원본=Fall"], fontsize=10)
ax.set_title("교차표 (명수)", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)

# -- 1b: 일치/불일치 파이
ax = axes[1]
sizes  = [match, ctrl_to_imp, fall_to_norm]
labels = [f"일치\n({match}명)", f"Control→OR저하\n({ctrl_to_imp}명)", f"Fall→OR정상\n({fall_to_norm}명)"]
colors = ["#4CAF50", "#FF7043", "#42A5F5"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                   autopct="%1.1f%%", startangle=90,
                                   textprops={"fontsize": 9})
ax.set_title("라벨 일치/불일치 비율", fontsize=11)

# -- 1c: OR 충족 기준 수 분포 (group별)
ax = axes[2]
for grp, color, label in [("Control", "#42A5F5", "Control"), ("Fall", "#FF7043", "Fall")]:
    vals = df[df["group"]==grp]["n_criteria_met"].value_counts().sort_index()
    ax.bar(vals.index + (0.2 if grp=="Fall" else -0.2),
           vals.values, width=0.35, color=color, alpha=0.8, label=label)
ax.set_xlabel("충족 OR 기준 수", fontsize=10)
ax.set_ylabel("명수", fontsize=10)
ax.set_title("OR 기준 충족 수 분포 (그룹별)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xticks(range(7))

plt.tight_layout()
p1 = OUT_DIR / "01_label_comparison_overview.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[시각화1] {p1}")

# ════════════════════════════════════════════════════════════════
# 시각화 2: OR 기준별 기여도 (Control vs Fall 별로)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("OR 기준별 양성 비율 — Control vs Fall", fontsize=13, fontweight="bold")

crit_names = list(OR_CRITERIA.keys())
ctrl_df = df[df["group"] == "Control"]
fall_df = df[df["group"] == "Fall"]

ctrl_rates = [ctrl_df[c].mean() * 100 for c in crit_names]
fall_rates = [fall_df[c].mean() * 100 for c in crit_names]

x = np.arange(len(crit_names))
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, ctrl_rates, w, label="Control", color="#42A5F5", alpha=0.85)
bars2 = ax.bar(x + w/2, fall_rates, w, label="Fall",    color="#FF7043", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(crit_names, fontsize=10, rotation=15)
ax.set_ylabel("양성 비율 (%)", fontsize=10)
ax.set_title("기준별 양성 비율", fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)

# 절대 인원수
ctrl_counts = [ctrl_df[c].sum() for c in crit_names]
fall_counts = [fall_df[c].sum() for c in crit_names]

ax = axes[1]
bars3 = ax.bar(x - w/2, ctrl_counts, w, label="Control", color="#42A5F5", alpha=0.85)
bars4 = ax.bar(x + w/2, fall_counts, w, label="Fall",    color="#FF7043", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(crit_names, fontsize=10, rotation=15)
ax.set_ylabel("명수", fontsize=10)
ax.set_title("기준별 양성 인원", fontsize=11)
ax.legend(fontsize=10)
for bar in bars3:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
for bar in bars4:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
p2 = OUT_DIR / "02_or_criteria_contribution.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[시각화2] {p2}")

# ════════════════════════════════════════════════════════════════
# 시각화 3: 임상 지표 분포 (4그룹 비교)
# Control/OR정상 | Control/OR저하 | Fall/OR정상 | Fall/OR저하
# ════════════════════════════════════════════════════════════════
df["group4"] = df["group"] + "/" + df["OR_label"].map({0:"OR정상", 1:"OR저하"})
group4_order = ["Control/OR정상", "Control/OR저하", "Fall/OR정상", "Fall/OR저하"]
colors4 = ["#1565C0", "#FF7043", "#42A5F5", "#B71C1C"]

clinical_cols = {
    "TUG (초)":        "TUG",
    "FSST (초)":       "FSST",
    "BERG (점)":       "BERG",
    "DGI (점)":        "DGI",
    "기본보행속도 (m/s)": "base(velocity)",
    "S3보행속도 (m/s)":  "s3(velocity)",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("임상 지표 분포 — 4그룹 비교 (원본라벨 × OR라벨)", fontsize=13, fontweight="bold")

cutoffs = {"TUG": 12, "FSST": 15, "BERG": 52, "DGI": 19,
           "base(velocity)": 1.0, "s3(velocity)": 1.0}
cutoff_above = {"TUG": False, "FSST": False, "BERG": True, "DGI": True,
                "base(velocity)": True, "s3(velocity)": True}

for ax, (ylabel, col) in zip(axes.flat, clinical_cols.items()):
    data_by_group = [df[df["group4"]==g][col].dropna().values for g in group4_order]
    counts = [len(d) for d in data_by_group]

    bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors4):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # OR 기준선
    cut = cutoffs[col]
    ax.axhline(cut, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"컷오프 {cut}")

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(
        [f"{g}\n(n={c})" for g, c in zip(group4_order, counts)],
        fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
p3 = OUT_DIR / "03_clinical_distribution_4groups.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[시각화3] {p3}")

# ════════════════════════════════════════════════════════════════
# 시각화 4: 불일치 subject 상세 (Control인데 OR=저하 / Fall인데 OR=정상)
# ════════════════════════════════════════════════════════════════
mismatch_df = df[df["original_label"] != df["OR_label"]].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(mismatch_df)*0.35 + 2)))
fig.suptitle(f"라벨 불일치 subject 상세 (총 {len(mismatch_df)}명)", fontsize=13, fontweight="bold")

for ax_idx, (grp_filter, title, bar_color) in enumerate([
    ("Control", f"Control인데 OR=저하 ({ctrl_to_imp}명)\n→ PhysioNet은 정상, 임상지표로는 저하", "#FF7043"),
    ("Fall",    f"Fall인데 OR=정상 ({fall_to_norm}명)\n→ PhysioNet은 낙상이력, 임상지표로는 정상", "#42A5F5"),
]):
    ax = axes[ax_idx]
    sub_m = mismatch_df[mismatch_df["group"] == grp_filter].sort_values("subject_id")

    if len(sub_m) == 0:
        ax.text(0.5, 0.5, "해당 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        continue

    y_pos = np.arange(len(sub_m))
    crit_cols = list(OR_CRITERIA.keys())

    # 각 기준 충족 여부를 색상으로 표현
    cell_data = sub_m[crit_cols].values
    im = ax.imshow(cell_data.T, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=1, extent=[-0.5, len(sub_m)-0.5, -0.5, len(crit_cols)-0.5])
    for i in range(len(sub_m)):
        for j in range(len(crit_cols)):
            ax.text(i, j, "✓" if cell_data[i, j] else "·",
                    ha="center", va="center", fontsize=10,
                    color="white" if cell_data[i, j] else "gray")

    ax.set_xticks(range(len(sub_m)))
    ax.set_xticklabels(sub_m["subject_id"].values, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(crit_cols)))
    ax.set_yticklabels(crit_cols, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="기준 충족(1=빨강)")

plt.tight_layout()
p4 = OUT_DIR / "04_mismatch_subjects_detail.png"
plt.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"[시각화4] {p4}")

# ════════════════════════════════════════════════════════════════
# 시각화 5: 상관관계 히트맵 (임상 지표 간 + OR 라벨/원본 라벨)
# ════════════════════════════════════════════════════════════════
corr_cols = ["TUG", "FSST", "BERG", "DGI", "base(velocity)", "s3(velocity)",
             "original_label", "OR_label"]
corr_labels = ["TUG", "FSST", "BERG", "DGI", "기본속도", "S3속도", "원본라벨", "OR라벨"]
corr_df = df[corr_cols].dropna()
corr_mat = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_labels))); ax.set_xticklabels(corr_labels, fontsize=10, rotation=30)
ax.set_yticks(range(len(corr_labels))); ax.set_yticklabels(corr_labels, fontsize=10)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_mat.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("임상 지표 × 라벨 간 상관관계 히트맵", fontsize=12, fontweight="bold")
plt.tight_layout()
p5 = OUT_DIR / "05_correlation_heatmap.png"
plt.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()
print(f"[시각화5] {p5}")

# ════════════════════════════════════════════════════════════════
# 요약 출력
# ════════════════════════════════════════════════════════════════
print(f"""
═══════════════════════════════════════════════
요약
═══════════════════════════════════════════════
전체 subject      : {len(df)}명
원본 Control      : {(df['group']=='Control').sum()}명
원본 Fall         : {(df['group']=='Fall').sum()}명

OR 라벨 정상      : {(df['OR_label']==0).sum()}명
OR 라벨 저하      : {(df['OR_label']==1).sum()}명

라벨 일치         : {match}명 ({match/len(df)*100:.1f}%)
라벨 불일치       : {mismatch}명 ({mismatch/len(df)*100:.1f}%)
  Control→OR저하  : {ctrl_to_imp}명
  Fall→OR정상     : {fall_to_norm}명

저장 파일
  CSV  : {RELABEL_CSV}
  그림1: {p1}
  그림2: {p2}
  그림3: {p3}
  그림4: {p4}
  그림5: {p5}
═══════════════════════════════════════════════
""")
