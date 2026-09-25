"""BERT encoder with a small classification head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from zh_hate_speech.config import LABELS


class BertHateSpeechClassifier(nn.Module):
    """[CLS] representation -> Linear -> ReLU -> Dropout -> Linear(num_labels).

    The head sits on the encoder's final hidden state (768-d for bert-base), not on
    masked-LM vocabulary logits.
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        num_labels: int = len(LABELS),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> BertHateSpeechClassifier:
        return cls(AutoModel.from_pretrained(name), **kwargs)

    @classmethod
    def from_config_name(cls, name: str, **kwargs) -> BertHateSpeechClassifier:
        """Build the architecture without downloading weights (used when loading a checkpoint)."""
        return cls(AutoModel.from_config(AutoConfig.from_pretrained(name)), **kwargs)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def reset_head(self) -> None:
        for m in self.head:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)
