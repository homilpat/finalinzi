from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import seaborn as sns
except ImportError:
    sns = None

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MODEL_DIR     = PROJECT_ROOT / "final__2026" / "02_model"
FEATURE_CSV   = PROJECT_ROOT / "final__2026" / "01_preprocessing" / "labwalks_service10_amp_spec_features.csv"
CLINICAL_XLSX = PROJECT_ROOT / "final__2026" / "04_clinical_data" / "ClinicalDemogData_COFL.xlsx"
OUT_DIR       = PROJECT_ROOT / "시각화_domain4"

MODEL_PATH     = MODEL_DIR / "final_motor_domain4_labwalks10_logistic_C0p5.joblib"
METADATA_PATH  = MODEL_DIR / "final_motor_domain4_labwalks10_logistic_C0p5_metadata.json"
VALIDATION_CSV = MODEL_DIR / "domain4_full_validation_metrics.csv"
OOF_CSV        = MODEL_DIR / "domain4_oof_predictions.csv"

EXCLUDED_SUBJECTS  = {"CO024", "FL020"}
THRESHOLD_STRATEGY = "sens90_maxspec"

KOREAN_LABELS = {
    "v_amp_pool_median":        "수직 진폭 중앙값",
    "ml_amp_pool_iqr":          "좌우 진폭 변동성",
    "base_v_stride_regularity": "수직 stride 규칙성",
    "roll_amp_pool_iqr":        "roll 진폭 변동성",
}


# ── 공통 유틸 ────────────────────────────────────────────────────────

def setup_plot_style() -> None:
    for font in ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"]:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    if sns is not None:
        sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])


def savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def load_artifacts():
    model_obj = joblib.load(MODEL_PATH)
    metadata  = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features  = list(model_obj["features"])
    threshold = float(model_obj["threshold"])
    pipeline  = model_obj["pipeline"]
    return pipeline, metadata, features, threshold


def load_subject_feature_df(features: list[str]) -> pd.DataFrame:
    """window-level 피처 → subject-level 중앙값 집계 + 임상 라벨 결합"""
    feat = pd.read_csv(FEATURE_CSV)
    feat = feat[~feat["subject_id"].isin(EXCLUDED_SUBJECTS)]
    subj = feat.groupby("subject_id")[features].median().reset_index()

    clin = pd.read_excel(CLINICAL_XLSX)
    clin["subject_id"] = clin["#"].str.replace("-", "", regex=False)
    clin["target"] = ((clin["DGI"] <= 19) | (clin["TUG"] >= 12)).astype(int)
    clin = clin[["subject_id", "target"]]

    merged = subj.merge(clin, on="subject_id", how="inner")
    merged["group_label"] = merged["target"].map({0: "운동 정상", 1: "운동저하 가능"})
    return merged


def compute_pooled_metrics(scheme: str) -> dict:
    """OOF predictions → subject-level pooled AUC/Sens/Spec"""
    oof = pd.read_csv(OOF_CSV)
    s = oof[oof["scheme"] == scheme]
    pooled = s.groupby("subject_id", as_index=False).agg(
        target=("target", "first"),
        probability=("probability", "mean"),
        threshold=("threshold", "mean"),
    )
    pooled["pred"] = (pooled["probability"] >= pooled["threshold"]).astype(int)
    tp = int(((pooled["pred"] == 1) & (pooled["target"] == 1)).sum())
    fn = int(((pooled["pred"] == 0) & (pooled["target"] == 1)).sum())
    tn = int(((pooled["pred"] == 0) & (pooled["target"] == 0)).sum())
    fp = int(((pooled["pred"] == 1) & (pooled["target"] == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc  = roc_auc_score(pooled["target"], pooled["probability"]) if pooled["target"].nunique() > 1 else float("nan")
    return {"auc": auc, "sensitivity": sens, "specificity": spec}


# ── 개별 플롯 ────────────────────────────────────────────────────────

def plot_cv_summary(metadata: dict) -> None:
    cv = metadata["cv_metrics"]
    metrics   = ["AUC", "Sensitivity", "Specificity"]
    test_vals = [cv["test_auc"], cv["test_sensitivity"], cv["test_specificity"]]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, test_vals, width, label="Test CV", color="#4C78A8")
    plt.bar(x[0] - width, cv["train_auc"], width, label="Train CV (AUC)", color="#F58518", alpha=0.8)
    for bar, val in zip(bars, test_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10)
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("최종 모델 교차검증 성능 (5-fold × 20 repeats, sens90_maxspec)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig("01_cv_performance_summary.png")


def plot_confusion_matrix(metadata: dict) -> None:
    m  = metadata["apparent_train_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    labels = np.array([[f"TN\n{m['tn']}", f"FP\n{m['fp']}"],
                       [f"FN\n{m['fn']}", f"TP\n{m['tp']}"]])
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                    square=True, linewidths=1, linecolor="white")
    else:
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, labels[i, j], ha="center", va="center", fontsize=13)
    plt.xticks([0.5, 1.5], ["운동 정상 예측", "운동저하 예측"])
    plt.yticks([0.5, 1.5], ["실제 운동 정상", "실제 운동저하"], rotation=0)
    plt.title(f"Confusion matrix (apparent train) — Sens={m['sensitivity']:.3f}, Spec={m['specificity']:.3f}")
    savefig("02_confusion_matrix.png")


def plot_abce_bar() -> None:
    """A/B/C/LOSO subject-level pooled 성능 비교"""
    scheme_map = {
        "A_5fold_x100":          "5-fold×100",
        "B_3fold_x100":          "3-fold×100",
        "C_repeated_80_20_x100": "8:2×100",
        "E_LOSO_pooled":         "LOSO",
    }
    rows = []
    for scheme, label in scheme_map.items():
        m = compute_pooled_metrics(scheme)
        rows.append({"검증방식": label, **m})
    df = pd.DataFrame(rows)

    long = df.melt(id_vars="검증방식", value_vars=["auc", "sensitivity", "specificity"],
                   var_name="지표", value_name="값")
    long["지표"] = long["지표"].map({"auc": "AUC", "sensitivity": "Sensitivity", "specificity": "Specificity"})

    plt.figure(figsize=(9, 5))
    if sns is not None:
        sns.barplot(data=long, x="검증방식", y="값", hue="지표", palette="Set2")
    else:
        pivot = long.pivot(index="검증방식", columns="지표", values="값")
        pivot.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1)
    plt.title("A/B/C/LOSO 검증방식별 subject-level 성능 비교 (train_sens80_maxspec)")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.25)
    savefig("03_abce_validation_barplot.png")


def plot_train_test_gap() -> None:
    df = pd.read_csv(VALIDATION_CSV)
    df = df[df["threshold_strategy"] == THRESHOLD_STRATEGY]
    label_map = {"A_5fold_x100": "5-fold×100", "B_3fold_x100": "3-fold×100", "C_80_20_x100": "8:2×100"}
    rows = []
    for scheme, label in label_map.items():
        s = df[df["scheme"] == scheme]
        rows.append({
            "검증방식": label,
            "Train AUC": s[s["split"] == "train"]["auc"].mean(),
            "Test AUC":  s[s["split"] == "test"]["auc"].mean(),
        })
    g = pd.DataFrame(rows)
    x = np.arange(len(g))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, g["Train AUC"], width, label="Train AUC", color="#4C78A8")
    plt.bar(x + width / 2, g["Test AUC"],  width, label="Test AUC",  color="#F58518")
    for i, row in g.iterrows():
        gap = row["Train AUC"] - row["Test AUC"]
        plt.text(i, max(row["Train AUC"], row["Test AUC"]) + 0.012,
                 f"gap={gap:.3f}", ha="center", fontsize=9)
    plt.xticks(x, g["검증방식"])
    plt.ylim(0.6, 1.0)
    plt.ylabel("AUC")
    plt.title("Train vs Test AUC gap — 과적합 점검 (sens90_maxspec)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig("04_train_test_auc_gap.png")


def plot_sensitivity_boxplot() -> None:
    df = pd.read_csv(VALIDATION_CSV)
    df = df[(df["threshold_strategy"] == THRESHOLD_STRATEGY) & (df["split"] == "test")]
    label_map = {"A_5fold_x100": "5-fold×100", "B_3fold_x100": "3-fold×100", "C_80_20_x100": "8:2×100"}
    df["검증방식"] = df["scheme"].map(label_map)
    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.boxplot(data=df, x="검증방식", y="sensitivity", color="#A0CBE8")
        sns.stripplot(data=df, x="검증방식", y="sensitivity", color="black", alpha=0.2, size=2)
    else:
        data = [df[df["검증방식"] == v]["sensitivity"] for v in label_map.values()]
        plt.boxplot(data, labels=list(label_map.values()))
    plt.ylim(-0.02, 1.02)
    plt.ylabel("Sensitivity")
    plt.title("Fold별 Sensitivity 분포 (test, sens90_maxspec)")
    plt.grid(axis="y", alpha=0.25)
    savefig("05_sensitivity_by_fold_boxplot.png")


def plot_feature_violin(df: pd.DataFrame, features: list[str]) -> None:
    long = df.melt(id_vars=["subject_id", "target", "group_label"],
                   value_vars=features, var_name="feature", value_name="value")
    long["feature_label"] = long["feature"].map(KOREAN_LABELS).fillna(long["feature"])
    plt.figure(figsize=(11, 5.5))
    if sns is not None:
        sns.violinplot(data=long, x="feature_label", y="value", hue="group_label",
                       split=False, inner=None, alpha=0.35)
        sns.boxplot(data=long, x="feature_label", y="value", hue="group_label",
                    width=0.28, dodge=True, showcaps=True,
                    boxprops={"facecolor": "none", "zorder": 3}, showfliers=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title="")
    else:
        for i, f in enumerate(features):
            plt.boxplot([df[df["target"] == 0][f], df[df["target"] == 1][f]],
                        positions=[i * 3, i * 3 + 1], widths=0.6)
        plt.xticks([i * 3 + 0.5 for i in range(len(features))],
                   [KOREAN_LABELS.get(f, f) for f in features])
    plt.title("최종 4개 feature 분포 비교 (subject-level 중앙값)")
    plt.xlabel("")
    plt.ylabel("Feature value")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig("06_feature_violin_boxplot.png")


def plot_feature_correlation(df: pd.DataFrame, features: list[str]) -> None:
    corr   = df[features].corr(method="spearman")
    labels = [KOREAN_LABELS.get(f, f) for f in features]
    plt.figure(figsize=(7, 6))
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                    xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("4개 feature Spearman correlation")
    savefig("07_feature_correlation_heatmap.png")


def plot_coefficient(pipeline, features: list[str]) -> None:
    coefs = pipeline.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({
        "label":       [KOREAN_LABELS.get(f, f) for f in features],
        "coefficient": coefs,
    }).sort_values("coefficient")
    colors = np.where(df["coefficient"] >= 0, "#E45756", "#4C78A8")
    plt.figure(figsize=(7.5, 4.5))
    plt.barh(df["label"], df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Logistic coefficient")
    plt.title("최종 Logistic Regression coefficient\n(음수 = 운동저하 가능군 위험 증가)")
    plt.grid(axis="x", alpha=0.25)
    savefig("08_logistic_coefficient.png")


# ── 메인 ────────────────────────────────────────────────────────────

def main() -> None:
    setup_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline, metadata, features, threshold = load_artifacts()
    feature_df = load_subject_feature_df(features)

    plot_cv_summary(metadata)
    plot_confusion_matrix(metadata)
    plot_abce_bar()
    plot_train_test_gap()
    plot_sensitivity_boxplot()
    plot_feature_violin(feature_df, features)
    plot_feature_correlation(feature_df, features)
    plot_coefficient(pipeline, features)

    print(f"\ndone. output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
