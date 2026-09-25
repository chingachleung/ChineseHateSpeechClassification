"""Fixtures that build a tiny BERT + tokenizer locally, so tests never hit the network."""

from __future__ import annotations

import pandas as pd
import pytest
from transformers import BertConfig, BertModel, BertTokenizerFast

from zh_hate_speech.model import BertHateSpeechClassifier

CHARS = list("你好我是他的人真這那個不了很笨豬滾回去吃飯天氣今日")
TEXTS = ["你好", "今天天氣很好", "我吃飯了", "你真笨", "滾回去", "那個人很笨", "他是豬", "滾回去吃飯", "這個不好"]
LABELS = [0, 0, 0, 1, 2, 1, 2, 2, 0]


@pytest.fixture
def tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *CHARS]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), tokenize_chinese_chars=True)


@pytest.fixture
def tiny_model(tokenizer):
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertHateSpeechClassifier(BertModel(cfg), dropout=0.1)


@pytest.fixture
def csv_files(tmp_path):
    df = pd.DataFrame({"Tweet": TEXTS, "Label": LABELS})
    train, val = tmp_path / "train.csv", tmp_path / "val.csv"
    df.to_csv(train, index=False, encoding="utf-8-sig")
    df.to_csv(val, index=False, encoding="utf-8-sig")
    return train, val
