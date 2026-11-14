"""Render presentation-ready gait equations as PNG files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "gait_formula_images"

FORMULAS = [
    (
        "01_v_acf_symmetry",
        "V ACF step/stride symmetry",
        r"$\mathrm{v\_acf\_step\_stride\_symmetry}"
        r"=\dfrac{\mathrm{ACF}_{V}(k_{\mathrm{step},V})}"
        r"{\mathrm{ACF}_{V}(k_{\mathrm{stride},V})}$",
    ),
    (
        "02_ap_acf_symmetry",
        "AP ACF step/stride symmetry",
        r"$\mathrm{ap\_acf\_step\_stride\_symmetry}"
        r"=\dfrac{\mathrm{ACF}_{AP}(k_{\mathrm{step},AP})}"
        r"{\mathrm{ACF}_{AP}(k_{\mathrm{stride},AP})}$",
    ),
    (
        "03_v_stride_frequency",
        "Vertical stride frequency",
        r"$\mathrm{v\_stride\_freq\_hz}"
        r"=\dfrac{f_s}{k_{\mathrm{stride},V}}"
        r"=\dfrac{100}{k_{\mathrm{stride},V}}$",
    ),
    (
        "04_ap_spectral_entropy",
        "AP spectral entropy",
        r"$\mathrm{ap\_spec\_entropy}"
        r"=-\dfrac{\sum_{i=1}^{M}p_i\log_{2}(p_i)}{\log_{2}(M)},"
        r"\quad p_i=\dfrac{S_i}{\sum_{j=1}^{M}S_j}$",
    ),
    (
        "05_v_jerk_rms",
        "Vertical jerk RMS",
        r"$\mathrm{v\_jerk\_rms}"
        r"=100\sqrt{\dfrac{1}{999}\sum_{n=0}^{998}"
        r"\left(v_{\mathrm{acc}}[n+1]-v_{\mathrm{acc}}[n]\right)^2}$",
    ),
    (
        "06_window_median",
        "Median across six 10-second windows",
        r"$F_{\mathrm{median}}"
        r"=\mathrm{median}\left(F^{(1)},F^{(2)},F^{(3)},"
        r"F^{(4)},F^{(5)},F^{(6)}\right)$",
    ),
    (
        "07_window_iqr",
        "IQR across six 10-second windows",
        r"$F_{\mathrm{IQR}}"
        r"=Q_{0.75}\left(\mathbf{F}\right)"
        r"-Q_{0.25}\left(\mathbf{F}\right)$",
    ),
]


def render_formula(stem: str, title: str, formula: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.2), facecolor="white")
    ax.axis("off")
    ax.text(
        0.02,
        0.78,
        title,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#1F2937",
        va="center",
    )
    ax.text(
        0.5,
        0.35,
        formula,
        transform=ax.transAxes,
        fontsize=21,
        color="#111827",
        ha="center",
        va="center",
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(
        OUT_DIR / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def render_sheet() -> None:
    fig, axes = plt.subplots(
        len(FORMULAS),
        1,
        figsize=(14, 13),
        facecolor="white",
    )
    for ax, (_, title, formula) in zip(axes, FORMULAS):
        ax.axis("off")
        ax.text(
            0.01,
            0.76,
            title,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color="#374151",
            va="center",
        )
        ax.text(
            0.5,
            0.29,
            formula,
            transform=ax.transAxes,
            fontsize=17,
            color="#111827",
            ha="center",
            va="center",
        )
    fig.tight_layout(h_pad=1.2)
    fig.savefig(
        OUT_DIR / "00_all_gait_formulas.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, title, formula in FORMULAS:
        render_formula(stem, title, formula)
    render_sheet()
    print(f"Rendered {len(FORMULAS) + 1} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
