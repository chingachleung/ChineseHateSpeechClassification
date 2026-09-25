"""Training loop, evaluation, early stopping and checkpoint I/O."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from zh_hate_speech.config import ID2LABEL, TrainConfig
from zh_hate_speech.model import BertHateSpeechClassifier

log = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
ENCODER_CONFIG_DIR = "encoder_config"
TOKENIZER_DIR = "tokenizer"
TRAIN_CONFIG_FILE = "train_config.json"
HISTORY_FILE = "history.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted label ids, softmax probabilities)."""
    model.eval()
    all_probs = []
    for batch in loader:
        batch = _to_device(batch, device)
        batch.pop("labels", None)
        logits = model(**batch)
        all_probs.append(torch.softmax(logits, dim=-1).cpu())
    probs = torch.cat(all_probs).numpy()
    return probs.argmax(axis=1), probs


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        y = batch.pop("labels")
        logits = model(**batch)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@dataclass
class EarlyStopper:
    """Stops when validation macro-F1 hasn't improved by ``min_delta`` for ``patience`` epochs."""

    patience: int = 2
    min_delta: float = 1e-3
    best: float = -float("inf")
    bad_epochs: int = field(default=0, init=False)

    def step(self, score: float) -> bool:
        """Record ``score``; return True if it is a new best."""
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(out_dir: Path, model: BertHateSpeechClassifier, tokenizer, cfg: TrainConfig, extra: dict) -> None:
    """Write a self-contained model directory that can be reloaded fully offline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "id2label": ID2LABEL, **extra}, out_dir / MODEL_FILE)
    model.encoder.config.save_pretrained(out_dir / ENCODER_CONFIG_DIR)
    tokenizer.save_pretrained(out_dir / TOKENIZER_DIR)
    (out_dir / TRAIN_CONFIG_FILE).write_text(json.dumps(cfg.to_dict(), indent=2))


def load_checkpoint(
    model_dir: str | Path, device: torch.device | None = None
) -> tuple[BertHateSpeechClassifier, object, TrainConfig]:
    model_dir = Path(model_dir)
    cfg = TrainConfig(**json.loads((model_dir / TRAIN_CONFIG_FILE).read_text()))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir / ENCODER_CONFIG_DIR))
    model = BertHateSpeechClassifier(encoder, dropout=cfg.dropout)
    state = torch.load(model_dir / MODEL_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir / TOKENIZER_DIR)
    if device is not None:
        model.to(device)
    return model, tokenizer, cfg


def fit(
    model: BertHateSpeechClassifier,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: TrainConfig,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Fine-tune with AdamW + linear warmup, keep the best checkpoint by validation macro-F1."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    stopper = EarlyStopper(cfg.patience, cfg.min_delta)
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in bar:
            batch = _to_device(batch, device)
            y = batch.pop("labels")
            logits = model(**batch)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            bar.set_postfix(loss=f"{running / seen:.4f}", acc=f"{correct / seen:.3f}")

        val = evaluate(model, val_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": running / seen,
            "train_acc": correct / seen,
            "val_loss": val["loss"],
            "val_macro_f1": val["macro_f1"],
            "val_micro_f1": val["micro_f1"],
        }
        history.append(record)
        improved = stopper.step(val["macro_f1"])
        log.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_macro_f1 %.4f | val_micro_f1 %.4f%s",
            epoch,
            record["train_loss"],
            val["loss"],
            val["macro_f1"],
            val["micro_f1"],
            "  *best*" if improved else "",
        )
        if improved:
            save_checkpoint(out_dir, model, tokenizer, cfg, {"epoch": epoch, "val_macro_f1": val["macro_f1"]})
        if stopper.should_stop:
            log.info("early stopping after epoch %d (best val macro-F1 %.4f)", epoch, stopper.best)
            break

    (out_dir / HISTORY_FILE).write_text(json.dumps(history, indent=2))
    return history
