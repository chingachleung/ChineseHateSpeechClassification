"""Label schema and training hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Label ids match the annotation guidelines used to build the dataset.
LABELS: tuple[str, ...] = ("Neither", "Abusive-only", "Hate-speech")
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
LABEL2ID: dict[str, int] = {name: i for i, name in ID2LABEL.items()}


@dataclass
class TrainConfig:
    """Hyperparameters for fine-tuning. Defaults reproduce the original experiments."""

    pretrained_model: str = "bert-base-chinese"
    max_len: int = 100  # covers the vast majority of tweets after character tokenisation
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.3
    patience: int = 2  # early stopping on validation macro-F1
    min_delta: float = 1e-3
    class_weighting: str = "sqrt_inverse"  # "sqrt_inverse" | "inverse" | "none"
    freeze_encoder: bool = False
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
