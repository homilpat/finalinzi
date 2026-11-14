"""Create row-normalized percentage confusion matrices for no-s3 results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
)
OUT_DIR = SOURCE_DIR / "confusion_matrices_percent"


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0,
    ) * 100.0


def save_plot(
    model: str,
    matrix: np.ndarray,
    suffix: str,
    title: str,
    show_counts: bool = True,
) -> None:
    percent = normalized_rows(matrix)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    for row in range(2):
        for column in range(2):
            color = "white" if percent[row, column] >= 55 else "black"
            label = f"{percent[row, column]:.1f}%"
            if show_counts:
                label += f"\n(n={matrix[row, column]:,})"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], labels=["Predicted normal", "Predicted impaired"])
    ax.set_yticks([0, 1], labels=["Actual normal", "Actual impaired"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row percentage (%)")
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"confusion_matrix_{model}_{suffix}_percent.png",
        dpi=200,
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = pd.read_csv(SOURCE_DIR / "confusion_matrix_pooled_100rep.csv")
    rows = []
    for item in pooled.itertuples(index=False):
        matrix = np.asarray([[item.tn, item.fp], [item.fn, item.tp]], dtype=int)
        percent = normalized_rows(matrix)
        save_plot(
            item.model,
            matrix,
            "pooled_100rep",
            f"{item.model}: pooled 100-repeat confusion matrix",
        )
        rows.append(
            {
                "scope": "pooled_100rep",
                "model": item.model,
                "actual_normal_pred_normal_pct": percent[0, 0],
                "actual_normal_pred_impaired_pct": percent[0, 1],
                "actual_impaired_pred_normal_pct": percent[1, 0],
                "actual_impaired_pred_impaired_pct": percent[1, 1],
                "tn": item.tn,
                "fp": item.fp,
                "fn": item.fn,
                "tp": item.tp,
            }
        )

    consensus = pd.read_csv(SOURCE_DIR / "subject_consensus_predictions.csv")
    for model, part in consensus.groupby("model", sort=True):
        matrix = confusion_matrix(
            part["target"],
            part["prediction"],
            labels=[0, 1],
        )
        percent = normalized_rows(matrix)
        save_plot(
            model,
            matrix,
            "subject_level",
            f"{model}: subject-level confusion matrix",
            show_counts=False,
        )
        rows.append(
            {
                "scope": "subject_consensus",
                "model": model,
                "actual_normal_pred_normal_pct": percent[0, 0],
                "actual_normal_pred_impaired_pct": percent[0, 1],
                "actual_impaired_pred_normal_pct": percent[1, 0],
                "actual_impaired_pred_impaired_pct": percent[1, 1],
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
            }
        )

    output = pd.DataFrame(rows).sort_values(["scope", "model"])
    output.to_csv(
        OUT_DIR / "confusion_matrix_percentages.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output.loc[
        output["scope"].eq("subject_consensus"),
        [
            "model",
            "actual_normal_pred_normal_pct",
            "actual_normal_pred_impaired_pct",
            "actual_impaired_pred_normal_pct",
            "actual_impaired_pred_impaired_pct",
        ],
    ].to_csv(
        OUT_DIR / "subject_confusion_matrix_percentages_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(output.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()
