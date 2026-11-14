"""Plot the top 20 S3-label feature combinations from the 100-repeat screen."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "analysis_outputs"
    / "s3_exhaustive_feature_combinations_100rep"
    / "all_1_to_4_feature_combinations.csv"
)
OUT_DIR = (
    ROOT
    / "analysis_outputs"
    / "s3_exhaustive_feature_combinations_100rep"
)
PNG_PATH = OUT_DIR / "top20_feature_combinations_auc.png"
PDF_PATH = OUT_DIR / "top20_feature_combinations_auc.pdf"
PLAIN_PNG_PATH = OUT_DIR / "top20_feature_combinations_auc_plain.png"
PLAIN_PDF_PATH = OUT_DIR / "top20_feature_combinations_auc_plain.pdf"

CURRENT_FEATURES = {
    "v_acf_step_stride_symmetry_iqr",
    "v_jerk_rms_median",
    "v_jerk_rms_iqr",
}

DISPLAY_NAMES = {
    "v_acf_step_stride_symmetry_median": "V ACF med",
    "v_acf_step_stride_symmetry_iqr": "V ACF IQR",
    "ap_acf_step_stride_symmetry_median": "AP ACF med",
    "ap_acf_step_stride_symmetry_iqr": "AP ACF IQR",
    "v_stride_freq_hz_median": "V SF med",
    "v_stride_freq_hz_iqr": "V SF IQR",
    "ap_spec_entropy_median": "AP Ent med",
    "ap_spec_entropy_iqr": "AP Ent IQR",
    "v_jerk_rms_median": "V Jerk med",
    "v_jerk_rms_iqr": "V Jerk IQR",
}


def display_label(row: pd.Series) -> str:
    features = row["features"].split("|")
    names = [DISPLAY_NAMES.get(feature, feature) for feature in features]
    return f"#{int(row['rank'])} · {int(row['n_features'])}F · " + " + ".join(names)


def render_chart(
    table: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    *,
    highlight_current: bool,
    show_error: bool,
) -> None:
    plot = table.iloc[::-1].reset_index(drop=True)

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = [
        "#18A999" if highlight_current and is_current else "#9AA9BA"
        for is_current in plot["is_current"]
    ]
    bars = ax.barh(
        plot["label"],
        plot["auc_mean"],
        xerr=plot["auc_std"] if show_error else None,
        color=colors,
        edgecolor="none",
        error_kw={
            "ecolor": "#5B6775",
            "elinewidth": 1.0,
            "capsize": 2.5,
            "alpha": 0.75,
        },
    )

    lower = max(0.84, float((plot["auc_mean"] - plot["auc_std"]).min()) - 0.003)
    upper = min(1.0, float((plot["auc_mean"] + plot["auc_std"]).max()) + 0.008)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Mean subject-level OOF ROC-AUC")
    ax.set_title(
        "Top 20 feature combinations by repeated OOF AUC\n"
        "S3-inclusive OR label · 5-fold CV × 100 repeats",
        pad=16,
        fontweight="bold",
    )
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.2, length=0)
    ax.tick_params(axis="x", labelsize=10)

    for bar, (_, row) in zip(bars, plot.iterrows()):
        value = float(row["auc_mean"])
        text = (
            f"{value:.3f} ± {float(row['auc_std']):.3f}"
            if show_error
            else f"{value:.3f}"
        )
        is_highlighted = bool(row["is_current"]) and highlight_current
        ax.text(
            min(value + 0.0008, upper - 0.012),
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if is_highlighted else "normal",
            color="#075E54" if is_highlighted else "#303842",
        )
        if is_highlighted:
            ax.text(
                upper - 0.001,
                bar.get_y() + bar.get_height() / 2,
                "CURRENT",
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="#075E54",
            )

    if show_error:
        fig.text(
            0.985,
            0.018,
            "Error bars: SD across 100 repeated CV runs",
            ha="right",
            color="#5B6775",
            fontsize=9,
        )
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.98))
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(SOURCE).head(20).copy()
    table["label"] = table.apply(display_label, axis=1)
    table["is_current"] = table["features"].apply(
        lambda value: set(value.split("|")) == CURRENT_FEATURES
    )
    render_chart(
        table,
        PNG_PATH,
        PDF_PATH,
        highlight_current=True,
        show_error=True,
    )
    render_chart(
        table,
        PLAIN_PNG_PATH,
        PLAIN_PDF_PATH,
        highlight_current=False,
        show_error=False,
    )
    print(PNG_PATH)
    print(PDF_PATH)
    print(PLAIN_PNG_PATH)
    print(PLAIN_PDF_PATH)


if __name__ == "__main__":
    main()
