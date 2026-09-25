"""Evaluation reports and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

from zh_hate_speech.config import LABELS  # noqa: E402


def text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, digits=3, zero_division=0
    )


def dict_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax1.set(xlabel="epoch", ylabel="weighted CE loss", title="Loss")
    ax1.legend()
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", label="macro-F1")
    ax2.plot(epochs, [h["val_micro_f1"] for h in history], marker="o", label="micro-F1")
    ax2.set(xlabel="epoch", ylabel="F1", title="Validation F1")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
