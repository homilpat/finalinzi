from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_outputs" / "with_s3_six_model_library_defaults_100rep"
PREDICTION_PATH = OUT_DIR / "subject_oof_predictions.csv"
MODEL_ORDER = ["LR", "RF", "XGB", "SVM", "CNN1D", "LSTM"]


def percent_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = confusion_matrix(
        frame["label"], frame["prediction"], labels=[0, 1]
    ).astype(float)
    return matrix / matrix.sum(axis=1, keepdims=True) * 100


def draw(ax: plt.Axes, matrix: np.ndarray, model: str) -> None:
    labels = np.array(
        [[f"{matrix[row, col]:.1f}%" for col in range(2)] for row in range(2)]
    )
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=False,
        square=True,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Normal", "Impaired"],
        yticklabels=["Normal", "Impaired"],
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(model, fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="both", rotation=0)


def main() -> None:
    predictions = pd.read_csv(PREDICTION_PATH)
    predictions = predictions[predictions["threshold_rule"] == "fixed_0.5"]

    matrices = {}
    for model in MODEL_ORDER:
        matrices[model] = percent_matrix(predictions[predictions["model"] == model])
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        draw(ax, matrices[model], model)
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"confusion_percent_{model.lower()}.png",
            dpi=240,
            bbox_inches="tight",
        )
        fig.savefig(
            OUT_DIR / f"confusion_percent_{model.lower()}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ax, model in zip(axes.flat, MODEL_ORDER):
        draw(ax, matrices[model], model)
    fig.tight_layout(w_pad=2.5, h_pad=2.2)
    fig.savefig(
        OUT_DIR / "confusion_percent_all_models.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "confusion_percent_all_models.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    rows = []
    for model, matrix in matrices.items():
        rows.append(
            {
                "model": model,
                "actual_normal_pred_normal_percent": matrix[0, 0],
                "actual_normal_pred_impaired_percent": matrix[0, 1],
                "actual_impaired_pred_normal_percent": matrix[1, 0],
                "actual_impaired_pred_impaired_percent": matrix[1, 1],
            }
        )
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "confusion_percent_values.csv", index=False
    )


if __name__ == "__main__":
    main()
