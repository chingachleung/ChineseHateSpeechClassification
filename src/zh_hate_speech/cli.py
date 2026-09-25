"""Command-line interface: ``zh-hate {train,evaluate,predict}``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from zh_hate_speech.config import ID2LABEL, TrainConfig
from zh_hate_speech.data import HateSpeechDataset, compute_class_weights, load_csv
from zh_hate_speech.engine import evaluate, fit, get_device, load_checkpoint, predict, set_seed
from zh_hate_speech.model import BertHateSpeechClassifier
from zh_hate_speech.reporting import dict_report, plot_confusion_matrix, plot_history, text_report

log = logging.getLogger("zh_hate_speech")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    """Expose every TrainConfig field as an optional --flag (defaults live in the dataclass)."""
    g = p.add_argument_group("hyperparameters (override TrainConfig defaults)")
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            g.add_argument(flag, action="store_true", default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(f.type), str)
            g.add_argument(flag, type=typ, default=None, help=f"default: {f.default}")


def _config_from_args(args: argparse.Namespace, base: TrainConfig | None = None) -> TrainConfig:
    cfg = dataclasses.replace(base) if base else TrainConfig()
    overrides = {f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)}
    return dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def cmd_train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    device = get_device()

    if args.init_from:
        # Continued fine-tuning (e.g. domain adaptation) from an existing checkpoint.
        model, tokenizer, base_cfg = load_checkpoint(args.init_from)
        cfg = _config_from_args(args, base_cfg)
        if args.reset_head:
            model.reset_head()
    else:
        cfg = _config_from_args(args)
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
        model = BertHateSpeechClassifier.from_pretrained(cfg.pretrained_model, dropout=cfg.dropout)
    if cfg.freeze_encoder:
        model.freeze_encoder()

    set_seed(cfg.seed)
    train_x, train_y = load_csv(args.train_file)
    val_x, val_y = load_csv(args.val_file)
    weights = compute_class_weights(train_y, scheme=cfg.class_weighting)
    log.info(
        "device=%s  train=%d  val=%d  class_weights=%s",
        device,
        len(train_x),
        len(val_x),
        [round(w, 3) for w in weights.tolist()],
    )

    train_loader = DataLoader(
        HateSpeechDataset(train_x, train_y, tokenizer, cfg.max_len), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(HateSpeechDataset(val_x, val_y, tokenizer, cfg.max_len), batch_size=cfg.eval_batch_size)

    history = fit(model, tokenizer, train_loader, val_loader, weights, cfg, out_dir, device)
    plot_history(history, out_dir / "training_curves.png")
    log.info("saved best model to %s", out_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    model, tokenizer, cfg = load_checkpoint(args.model_dir, device)
    texts, labels = load_csv(args.test_file)
    loader = DataLoader(HateSpeechDataset(texts, labels, tokenizer, cfg.max_len), batch_size=cfg.eval_batch_size)

    res = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
    print(text_report(res["y_true"], res["y_pred"]))

    out_dir = Path(args.output_dir or args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict_report(res["y_true"], res["y_pred"])
    report["micro_f1"] = res["micro_f1"]
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    plot_confusion_matrix(res["y_true"], res["y_pred"], out_dir / "confusion_matrix.png")
    log.info("wrote test_metrics.json and confusion_matrix.png to %s", out_dir)


def cmd_predict(args: argparse.Namespace) -> None:
    device = get_device()
    model, tokenizer, cfg = load_checkpoint(args.model_dir, device)
    if args.input_file:
        texts = [
            line.strip() for line in Path(args.input_file).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        texts = args.text
    loader = DataLoader(HateSpeechDataset(texts, None, tokenizer, cfg.max_len), batch_size=cfg.eval_batch_size)
    preds, probs = predict(model, loader, device)
    for text, p, pr in zip(texts, preds, probs):
        print(
            json.dumps(
                {"text": text, "label": ID2LABEL[int(p)], "confidence": round(float(pr[p]), 4)}, ensure_ascii=False
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zh-hate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune a classifier")
    t.add_argument("--train-file", required=True)
    t.add_argument("--val-file", required=True)
    t.add_argument("--output-dir", default="runs/bert-base-chinese")
    t.add_argument("--init-from", help="continue fine-tuning from an existing model directory")
    t.add_argument("--reset-head", action="store_true", help="re-initialise the classification head (with --init-from)")
    _add_config_args(t)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="score a model on a labelled test set")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--test-file", required=True)
    e.add_argument("--output-dir", help="where to write metrics/plots (default: model dir)")
    e.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="classify raw text")
    p.add_argument("--model-dir", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="one or more strings")
    src.add_argument("--input-file", help="UTF-8 file, one text per line")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
