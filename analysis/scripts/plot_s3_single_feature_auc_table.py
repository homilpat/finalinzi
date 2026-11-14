"""Render the S3-label single-feature AUC ranking as a presentation table."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "analysis_outputs"
    / "s3_exhaustive_feature_combinations_100rep"
    / "size_1_combinations.csv"
)
OUT_DIR = SOURCE.parent
PNG_PATH = OUT_DIR / "single_feature_auc_table.png"
PDF_PATH = OUT_DIR / "single_feature_auc_table.pdf"


def main() -> None:
    table = pd.read_csv(SOURCE).sort_values("auc_mean", ascending=False)
    rows = [
        [str(feature), f"{auc:.4f}", f"{std:.4f}"]
        for feature, auc, std in table[
            ["features", "auc_mean", "auc_std"]
        ].itertuples(index=False, name=None)
    ]

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.axis("off")
    ax.set_title(
        "1. 단일 파생변수 판별력 확인",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    rendered = ax.table(
        cellText=rows,
        colLabels=["파생 변수", "평균 AUC", "표준편차"],
        colWidths=[0.68, 0.16, 0.16],
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.07, 0.96, 0.83],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(11.5)

    for (row, column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#B8C0C8")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#D8F2FA")
            cell.set_text_props(fontweight="bold", color="#111827")
        else:
            cell.set_facecolor("#FFFFFF")
            cell.set_text_props(color="#111827")
        if column == 0 and row > 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.035

    fig.text(
        0.98,
        0.025,
        "S3 포함 OR 라벨 · subject-level 5-fold CV × 100회",
        ha="right",
        fontsize=9.5,
        color="#5B6775",
    )
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
