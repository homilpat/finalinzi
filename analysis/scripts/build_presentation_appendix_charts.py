"""Build presentation-ready charts for the technical appendix."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "presentation_appendix_pack"
AUDIT_PATH = (
    ROOT
    / "analysis_outputs"
    / "service_sample_qc_audit"
    / "sample_qc_audit.csv"
)
OPTUNA_MODELS_PATH = (
    ROOT
    / "analysis_outputs"
    / "with_s3_optuna_4model_nested_100rep"
    / "summary.csv"
)
PAIRED_LR_PATH = (
    ROOT
    / "analysis_outputs"
    / "with_s3_lr_optuna_maxiter_paired_nested_100rep"
    / "summary.csv"
)


def setup_style() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def short_sample_name(name: str) -> str:
    text = str(name)
    text = text.replace("hazi_gait_anatomical_14cols_", "")
    text = text.replace("hazi_gait_calibrated_20s_", "")
    text = text.replace(".csv", "")
    return text


def plot_turn_audit() -> None:
    table = pd.read_csv(AUDIT_PATH)
    table["label"] = table["file"].map(short_sample_name)
    table = table.sort_values("probability", ascending=True).reset_index(drop=True)
    y = np.arange(len(table))

    fig, (ax_prob, ax_turn) = plt.subplots(
        1,
        2,
        figsize=(15, 7.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    colors = np.where(table["target"].eq(1), "#C65D4B", "#93A4B7")
    ax_prob.barh(y, table["probability"], color=colors)
    threshold = float(table["threshold"].dropna().iloc[0])
    ax_prob.axvline(
        threshold, color="#2D3748", linewidth=1.2, linestyle="--"
    )
    ax_prob.set(
        yticks=y,
        yticklabels=table["label"],
        xlim=(0, 1),
        xlabel="보행저하 예측확률",
        title="현재 LR 예측",
    )
    for index, value in enumerate(table["probability"]):
        ax_prob.text(
            min(float(value) + 0.018, 0.94),
            index,
            f"{float(value):.3f}",
            va="center",
            fontsize=9,
        )

    turn_percent = table["turn_fraction"] * 100
    ax_turn.barh(y, turn_percent.fillna(0), color="#6BAED6")
    ax_turn.set(
        yticks=y,
        yticklabels=[],
        xlim=(0, max(22, float(turn_percent.max(skipna=True)) + 3)),
        xlabel="20초 중 회전 검출 비율 (%)",
        title="Gyro 회전 Audit",
    )
    for index, value in enumerate(turn_percent):
        label = "N/A" if not np.isfinite(value) else f"{float(value):.1f}%"
        x = 0.4 if not np.isfinite(value) else float(value) + 0.4
        ax_turn.text(x, index, label, va="center", fontsize=9)

    for axis in (ax_prob, ax_turn):
        axis.grid(axis="x", color="#D9DEE5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    fig.suptitle(
        "20초 자유보행의 회전 영향 Audit",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.018,
        "회전은 모델 입력이 아닌 QC용 · n=9 gyro 가용 표본에서 확률과 회전비율 Spearman ρ=0.079",
        ha="right",
        fontsize=9,
        color="#5B6775",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.94))
    fig.savefig(
        OUT_DIR / "01_turn_audit_20s.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_optuna_models() -> None:
    table = pd.read_csv(OPTUNA_MODELS_PATH)
    order = ["LR", "SVM", "RF", "GBM"]
    table = table.set_index("model").loc[order].reset_index()
    metrics = [
        ("auc_mean", "AUC"),
        ("sensitivity_mean", "민감도"),
        ("specificity_mean", "특이도"),
        ("accuracy_mean", "정확도"),
    ]
    x = np.arange(len(table))
    width = 0.19
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"]

    fig, ax = plt.subplots(figsize=(13, 7.2))
    for index, ((column, label), color) in enumerate(zip(metrics, colors)):
        values = table[column].to_numpy(float)
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8.5)

    ax.axhline(0.80, color="#2D3748", linewidth=1.1, linestyle="--")
    ax.text(
        len(table) - 0.45,
        0.805,
        "민감도 목표 0.80",
        ha="right",
        fontsize=9,
        color="#2D3748",
    )
    ax.set(
        xticks=x,
        xticklabels=table["model"],
        ylim=(0.65, 0.90),
        ylabel="Outer-test 성능",
        title="S3 포함 OR 라벨 · Optuna Nested CV 모델별 결과",
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.018,
        "Subject-level outer 5-fold × 100회 · inner 3-fold · 모델별 15 trials · fold별 threshold 재선정",
        ha="right",
        fontsize=9,
        color="#5B6775",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.98))
    fig.savefig(
        OUT_DIR / "02_optuna_4model_nested_100rep.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_fixed_vs_optuna() -> None:
    table = pd.read_csv(PAIRED_LR_PATH)
    table = table.set_index("model").loc[["Fixed LR", "Optuna LR"]]
    metrics = [
        ("auc_mean", "AUC"),
        ("sensitivity_mean", "민감도"),
        ("specificity_mean", "특이도"),
        ("accuracy_mean", "정확도"),
    ]
    x = np.arange(len(metrics))
    width = 0.33
    fixed = [float(table.loc["Fixed LR", column]) for column, _ in metrics]
    tuned = [float(table.loc["Optuna LR", column]) for column, _ in metrics]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    bars_fixed = ax.bar(
        x - width / 2,
        fixed,
        width,
        label="고정 LR",
        color="#4C78A8",
    )
    bars_tuned = ax.bar(
        x + width / 2,
        tuned,
        width,
        label="Optuna LR",
        color="#F28E2B",
    )
    ax.bar_label(bars_fixed, fmt="%.3f", padding=3, fontsize=10)
    ax.bar_label(bars_tuned, fmt="%.3f", padding=3, fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=[label for _, label in metrics],
        ylim=(0.65, 0.90),
        ylabel="Outer-test 성능",
        title="고정 LR과 Optuna LR의 paired nested CV 비교",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99,
        0.018,
        "동일 outer fold × 100회 · Optuna AUC 우위 11회 / 열위 89회",
        ha="right",
        fontsize=9,
        color="#5B6775",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.98))
    fig.savefig(
        OUT_DIR / "03_fixed_vs_optuna_lr.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_turn_audit()
    plot_optuna_models()
    plot_fixed_vs_optuna()
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
