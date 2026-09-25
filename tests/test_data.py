import math

import pandas as pd
import pytest
import torch

from zh_hate_speech.data import HateSpeechDataset, compute_class_weights, load_csv


def test_sqrt_inverse_weights_match_original_formula():
    # Original experiments used sqrt(17/61), 1, sqrt(17/22) for a 61/17/22 class split.
    labels = [0] * 61 + [1] * 17 + [2] * 22
    w = compute_class_weights(labels)
    assert w.tolist() == pytest.approx([math.sqrt(17 / 61), 1.0, math.sqrt(17 / 22)])


def test_weight_schemes():
    labels = [0, 0, 0, 0, 1, 1, 2]
    assert compute_class_weights(labels, scheme="none").tolist() == [1.0, 1.0, 1.0]
    assert compute_class_weights(labels, scheme="inverse").tolist() == pytest.approx([0.25, 0.5, 1.0])
    with pytest.raises(ValueError):
        compute_class_weights(labels, scheme="bogus")


def test_missing_class_gets_unit_weight():
    assert compute_class_weights([0, 0, 1]).tolist()[2] == 1.0


def test_load_csv_roundtrip(csv_files):
    texts, labels = load_csv(csv_files[0])
    assert len(texts) == len(labels) == 9
    assert all(isinstance(t, str) for t in texts)


def test_load_csv_rejects_bad_labels(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Tweet": ["a"], "Label": [5]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unexpected label"):
        load_csv(p)


def test_load_csv_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_csv(p)


def test_dataset_item_shapes_and_dtypes(tokenizer):
    ds = HateSpeechDataset(["你好", "你真笨"], [0, 1], tokenizer, max_len=8)
    item = ds[1]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].dtype == torch.long
    # Chinese is tokenised per character: [CLS] 你 真 笨 [SEP] + padding
    assert int(item["attention_mask"].sum()) == 5


def test_dataset_without_labels(tokenizer):
    ds = HateSpeechDataset(["你好"], None, tokenizer, max_len=8)
    assert "labels" not in ds[0]
