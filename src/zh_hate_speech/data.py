"""Dataset loading, tokenisation and class weighting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zh_hate_speech.config import LABELS

TEXT_COLUMN = "Tweet"
LABEL_COLUMN = "Label"


def load_csv(path: str | Path) -> tuple[list[str], list[int]]:
    """Read a CSV with ``Tweet`` and integer ``Label`` columns.

    ``utf-8-sig`` strips the BOM that Excel adds when exporting Chinese text.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s) {sorted(missing)}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    labels = df[LABEL_COLUMN].astype(int)
    bad = sorted(set(labels) - set(range(len(LABELS))))
    if bad:
        raise ValueError(f"{path}: unexpected label id(s) {bad}; expected 0..{len(LABELS) - 1}")
    return df[TEXT_COLUMN].astype(str).tolist(), labels.tolist()


def compute_class_weights(
    labels: Sequence[int], num_classes: int = len(LABELS), scheme: str = "sqrt_inverse"
) -> torch.Tensor:
    """Loss weights that counter class imbalance.

    ``sqrt_inverse`` gives w_c = sqrt(n_min / n_c): the rarest class gets weight 1 and
    frequent classes are down-weighted, but more gently than plain inverse frequency,
    which can push the model to over-predict the rare classes.
    """
    if scheme == "none":
        return torch.ones(num_classes)
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    counts[counts == 0] = np.nan  # unseen classes get weight 1 below
    ratio = np.nanmin(counts) / counts
    if scheme == "sqrt_inverse":
        weights = np.sqrt(ratio)
    elif scheme == "inverse":
        weights = ratio
    else:
        raise ValueError(f"unknown class weighting scheme: {scheme!r}")
    return torch.tensor(np.nan_to_num(weights, nan=1.0), dtype=torch.float)


class HateSpeechDataset(Dataset):
    """Tokenises tweets on the fly. ``labels`` may be omitted for inference."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None, tokenizer, max_len: int):
        if labels is not None and len(labels) != len(texts):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            # CrossEntropyLoss expects integer class indices.
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
