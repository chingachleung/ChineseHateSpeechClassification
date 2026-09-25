import json

import torch
from torch.utils.data import DataLoader

from zh_hate_speech.cli import main
from zh_hate_speech.config import TrainConfig
from zh_hate_speech.data import HateSpeechDataset, compute_class_weights, load_csv
from zh_hate_speech.engine import EarlyStopper, fit, load_checkpoint, predict, save_checkpoint


def test_forward_shape(tiny_model, tokenizer):
    enc = tokenizer(["你好", "你真笨"], padding=True, return_tensors="pt")
    assert tiny_model(**enc).shape == (2, 3)


def test_freeze_encoder_leaves_only_head_trainable(tiny_model):
    tiny_model.freeze_encoder()
    trainable = {n.split(".")[0] for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert trainable == {"head"}


def test_early_stopper():
    s = EarlyStopper(patience=2, min_delta=0.01)
    assert s.step(0.50) and not s.should_stop
    assert not s.step(0.505)  # below min_delta
    assert not s.should_stop
    assert not s.step(0.40)
    assert s.should_stop


def test_fit_learns_and_checkpoint_roundtrips(tiny_model, tokenizer, csv_files, tmp_path):
    torch.manual_seed(0)
    texts, labels = load_csv(csv_files[0])
    ds = HateSpeechDataset(texts, labels, tokenizer, max_len=12)
    cfg = TrainConfig(epochs=40, learning_rate=5e-3, batch_size=9, patience=100, dropout=0.0, warmup_ratio=0.0)
    out = tmp_path / "run"
    history = fit(
        tiny_model,
        tokenizer,
        DataLoader(ds, batch_size=9, shuffle=True),
        DataLoader(ds, batch_size=9),
        compute_class_weights(labels),
        cfg,
        out,
        torch.device("cpu"),
    )

    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["train_acc"] > history[0]["train_acc"]
    assert json.loads((out / "history.json").read_text())

    model, tok, loaded_cfg = load_checkpoint(out)
    preds, probs = predict(
        model, DataLoader(HateSpeechDataset(texts, None, tok, 12), batch_size=4), torch.device("cpu")
    )
    assert preds.shape == (9,) and probs.shape == (9, 3)
    assert loaded_cfg.learning_rate == cfg.learning_rate


def test_cli_evaluate_and_predict(tiny_model, tokenizer, csv_files, tmp_path, capsys):
    model_dir = tmp_path / "model"
    save_checkpoint(model_dir, tiny_model, tokenizer, TrainConfig(max_len=12), {"epoch": 0})

    main(["evaluate", "--model-dir", str(model_dir), "--test-file", str(csv_files[1])])
    assert "Hate-speech" in capsys.readouterr().out
    assert (model_dir / "confusion_matrix.png").exists()
    assert "macro avg" in json.loads((model_dir / "test_metrics.json").read_text())

    main(["predict", "--model-dir", str(model_dir), "--text", "你好", "滾回去"])
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [x["text"] for x in lines] == ["你好", "滾回去"]
    assert all(x["label"] in {"Neither", "Abusive-only", "Hate-speech"} for x in lines)


def test_cli_continued_finetuning(tiny_model, tokenizer, csv_files, tmp_path):
    base = tmp_path / "base"
    save_checkpoint(base, tiny_model, tokenizer, TrainConfig(max_len=12), {"epoch": 0})
    out = tmp_path / "adapted"
    main(
        [
            "train",
            "--train-file",
            str(csv_files[0]),
            "--val-file",
            str(csv_files[1]),
            "--output-dir",
            str(out),
            "--init-from",
            str(base),
            "--freeze-encoder",
            "--reset-head",
            "--epochs",
            "2",
            "--learning-rate",
            "1e-3",
        ]
    )
    assert (out / "model.pt").exists() and (out / "training_curves.png").exists()
    cfg = json.loads((out / "train_config.json").read_text())
    assert cfg["freeze_encoder"] is True and cfg["epochs"] == 2 and cfg["max_len"] == 12
