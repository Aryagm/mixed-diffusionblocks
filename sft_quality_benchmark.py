from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from llm_dblocks.adapters import MLXLMAdapter, load_mlx_lm_adapter
from llm_dblocks.trainer import DBlockTrainer, DBlockTrainingConfig, masked_cross_entropy


DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-bf16"
DEFAULT_QLORA_MODEL = "mlx-community/Qwen2.5-0.5B-4bit"


@dataclass
class TokenizedSFTExample:
    tokens: list[int]
    loss_mask: list[float]


class TimingCallback:
    def __init__(self):
        self.train_reports: list[dict] = []
        self.val_reports: list[dict] = []

    def on_train_loss_report(self, train_info: dict):
        self.train_reports.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict):
        self.val_reports.append(dict(val_info))

    @property
    def val_seconds(self) -> float:
        return sum(float(report.get("val_time", 0.0)) for report in self.val_reports)

    @property
    def peak_gb(self) -> float | None:
        values = [
            float(report["peak_memory"])
            for report in self.train_reports
            if report.get("peak_memory") is not None
        ]
        return max(values) if values else None


def alpaca_prompt(row: dict[str, str]) -> str:
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def prepare_alpaca_sft(
    data_dir: Path,
    *,
    train_examples: int,
    valid_examples: int,
    test_examples: int,
    seed: int,
) -> dict[str, int]:
    from datasets import load_dataset

    total = train_examples + valid_examples + test_examples
    dataset = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=seed)
    if total > len(dataset):
        raise ValueError(f"Requested {total} rows, but Alpaca has {len(dataset)} rows")

    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in dataset.select(range(total)):
        prompt = alpaca_prompt(row)
        completion = row["output"].strip()
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
        )

    splits = {
        "train": rows[:train_examples],
        "valid": rows[train_examples : train_examples + valid_examples],
        "test": rows[train_examples + valid_examples :],
    }
    for split, split_rows in splits.items():
        with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {split: len(split_rows) for split, split_rows in splits.items()}


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_messages(row: dict) -> list[dict[str, str]]:
    if "messages" in row:
        return row["messages"]
    return [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], int]:
    tokens = tokenizer.apply_chat_template(messages)
    add_generation_prompt = messages[-1].get("role") == "assistant"
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=add_generation_prompt,
    )
    return list(tokens), len(prompt_tokens)


def tokenize_sft_rows(
    rows: Iterable[dict[str, str]],
    tokenizer,
    *,
    seq_len: int,
) -> list[TokenizedSFTExample]:
    examples = []
    for row in rows:
        tokens, response_offset = apply_chat_template(tokenizer, row_messages(row))
        tokens = tokens[: seq_len + 1]
        if len(tokens) < 2:
            continue
        loss_mask = [
            1.0 if token_idx >= response_offset else 0.0
            for token_idx in range(1, len(tokens))
        ]
        if not any(loss_mask):
            continue
        examples.append(TokenizedSFTExample(tokens=tokens, loss_mask=loss_mask))
    if not examples:
        raise ValueError("No usable SFT examples after tokenization/truncation")
    return examples


def sft_batch_iterator(
    examples: list[TokenizedSFTExample],
    *,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    seed: int,
):
    rng = random.Random(seed)
    order = list(range(len(examples)))
    position = 0

    while True:
        if position == 0 and shuffle:
            rng.shuffle(order)
        if position + batch_size > len(order):
            position = 0
            continue

        batch_indexes = order[position : position + batch_size]
        position += batch_size
        inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
        labels = np.zeros((batch_size, seq_len), dtype=np.int32)
        masks = np.zeros((batch_size, seq_len), dtype=np.float32)

        for row_idx, example_idx in enumerate(batch_indexes):
            example = examples[example_idx]
            usable = min(len(example.tokens) - 1, seq_len)
            inputs[row_idx, :usable] = example.tokens[:usable]
            labels[row_idx, :usable] = example.tokens[1 : usable + 1]
            masks[row_idx, :usable] = example.loss_mask[:usable]

        yield {
            "input_ids": mx.array(inputs, dtype=mx.int32),
            "labels": mx.array(labels, dtype=mx.int32),
            "loss_mask": mx.array(masks, dtype=mx.float32),
        }


def reset_peak_memory():
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def peak_memory_gb() -> float | None:
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / 1e9
    return None


def eval_response_ce(adapter, examples, args) -> float:
    adapter.model.eval()
    batches = sft_batch_iterator(
        examples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=False,
        seed=args.seed,
    )
    total = 0.0
    for _ in range(args.eval_batches):
        batch = next(batches)
        logits = adapter.model(batch["input_ids"])
        loss = masked_cross_entropy(
            logits,
            batch["labels"],
            adapter.vocab_size,
            batch["loss_mask"],
        )
        mx.eval(loss)
        total += float(loss.item())
    return total / args.eval_batches


def train_full(adapter, examples, args) -> float:
    adapter.model.unfreeze()
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    batches = sft_batch_iterator(
        examples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=True,
        seed=args.seed,
    )

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return masked_cross_entropy(
            logits,
            batch["labels"],
            adapter.vocab_size,
            batch["loss_mask"],
        )

    loss_and_grad = nn.value_and_grad(adapter.model, loss_fn)
    reset_peak_memory()
    start = time.perf_counter()
    for _ in range(args.steps):
        batch = next(batches)
        loss, grads = loss_and_grad(adapter.model, batch)
        if args.gradient_clip_norm > 0:
            grads, _ = optim.clip_grad_norm(grads, args.gradient_clip_norm)
        optimizer.update(adapter.model, grads)
        mx.eval(adapter.model.parameters(), optimizer.state, loss)
    return time.perf_counter() - start


def run_full(args, splits) -> dict:
    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    train_examples = tokenize_sft_rows(splits["train"], tokenizer, seq_len=args.seq_len)
    valid_examples = tokenize_sft_rows(splits["valid"], tokenizer, seq_len=args.seq_len)
    before = eval_response_ce(adapter, valid_examples, args)
    seconds = train_full(adapter, train_examples, args)
    after = eval_response_ce(adapter, valid_examples, args)
    peak_gb = peak_memory_gb()
    return {
        "ok": True,
        "method": "full_ar",
        "model": args.model,
        "response_ce_before": before,
        "response_ce_after": after,
        "response_ce_delta": before - after,
        "seconds": seconds,
        "peak_gb": peak_gb,
    }


def run_dblock(args, splits) -> dict:
    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    train_examples = tokenize_sft_rows(splits["train"], tokenizer, seq_len=args.seq_len)
    valid_examples = tokenize_sft_rows(splits["valid"], tokenizer, seq_len=args.seq_len)
    config = DBlockTrainingConfig(
        num_blocks=args.num_blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gamma=args.gamma,
        denoise_weight=args.denoise_weight,
        aux_lm_weight=args.aux_lm_weight,
        clean_lm_weight=args.clean_lm_weight,
        clean_lm_anchor_profile=args.clean_lm_anchor_profile,
        clean_lm_interval=args.clean_lm_interval,
        clean_lm_seq_len=args.clean_lm_seq_len,
        clean_lm_window_count=args.clean_lm_window_count,
        clean_lm_large_seq_len=args.clean_lm_large_seq_len,
        clean_lm_large_interval=args.clean_lm_large_interval,
        clean_lm_full_warmup_steps=args.clean_lm_full_warmup_steps,
        local_lm_weight=args.local_lm_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        objective=args.objective,
    )
    trainer = DBlockTrainer(adapter, config)
    batches = sft_batch_iterator(
        train_examples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=True,
        seed=args.seed,
    )
    before = eval_response_ce(adapter, valid_examples, args)
    reset_peak_memory()
    start = time.perf_counter()
    trainer.train(batches, iters=args.steps, log_every=args.log_every)
    seconds = time.perf_counter() - start
    after = eval_response_ce(adapter, valid_examples, args)
    peak_gb = peak_memory_gb()
    return {
        "ok": True,
        "method": "fast_windowed_blockwise",
        "model": args.model,
        "response_ce_before": before,
        "response_ce_after": after,
        "response_ce_delta": before - after,
        "seconds": seconds,
        "peak_gb": peak_gb,
        "objective": args.objective,
        "block_ranges": trainer.ranges,
        "denoise_weight": trainer.config.denoise_weight,
        "clean_lm_weight": trainer.config.clean_lm_weight,
        "clean_lm_anchor_profile": trainer.config.clean_lm_anchor_profile,
        "clean_lm_seq_len": trainer.config.clean_lm_seq_len,
        "clean_lm_large_seq_len": trainer.config.clean_lm_large_seq_len,
        "clean_lm_large_interval": trainer.config.clean_lm_large_interval,
        "clean_lm_full_warmup_steps": trainer.config.clean_lm_full_warmup_steps,
    }


def mlx_lora_args(args, *, mode: str) -> SimpleNamespace:
    from mlx_lm.lora import CONFIG_DEFAULTS

    values = dict(CONFIG_DEFAULTS)
    values.update(
        {
            "model": args.qlora_model if mode == "qlora" else args.model,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": args.lora_optimizer,
            "data": str(args.data_dir),
            "seed": args.seed,
            "num_layers": args.lora_num_layers,
            "batch_size": args.batch_size,
            "iters": args.steps,
            "val_batches": args.eval_batches,
            "learning_rate": args.qlora_lr if mode == "qlora" else args.lora_lr,
            "steps_per_report": args.steps,
            "steps_per_eval": args.steps,
            "adapter_path": str(args.adapter_root / mode),
            "save_every": args.steps + 1,
            "max_seq_length": args.seq_len,
            "grad_checkpoint": args.grad_checkpoint,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "mask_prompt": True,
            "report_to": None,
            "project_name": None,
        }
    )
    return SimpleNamespace(**values)


def run_lora_like(args, splits, *, mode: str) -> dict:
    from mlx_lm import load
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset as load_mlx_dataset

    lora_args = mlx_lora_args(args, mode=mode)
    model, tokenizer = load(lora_args.model, tokenizer_config={"trust_remote_code": True})
    adapter = MLXLMAdapter(model, tokenizer)
    valid_examples = tokenize_sft_rows(splits["valid"], tokenizer, seq_len=args.seq_len)
    before = eval_response_ce(adapter, valid_examples, args)
    train_set, valid_set, _ = load_mlx_dataset(lora_args, tokenizer)
    callback = TimingCallback()
    reset_peak_memory()
    start = time.perf_counter()
    train_model(lora_args, model, train_set, valid_set, callback)
    wall_seconds = time.perf_counter() - start
    seconds = max(wall_seconds - callback.val_seconds, 0.0)
    after = eval_response_ce(adapter, valid_examples, args)
    peak_gb = callback.peak_gb if callback.peak_gb is not None else peak_memory_gb()
    return {
        "ok": True,
        "method": mode,
        "model": lora_args.model,
        "base_model": args.model,
        "response_ce_before": before,
        "response_ce_after": after,
        "response_ce_delta": before - after,
        "seconds": seconds,
        "seconds_wall": wall_seconds,
        "val_seconds_inside_train": callback.val_seconds,
        "peak_gb": peak_gb,
        "lora_lr": lora_args.learning_rate,
        "lora_num_layers": lora_args.num_layers,
        "grad_checkpoint": lora_args.grad_checkpoint,
    }


def clear_mlx_state():
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def load_splits(data_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        split: load_jsonl(data_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }


def run_modes(args) -> dict:
    split_counts = prepare_alpaca_sft(
        args.data_dir,
        train_examples=args.train_examples,
        valid_examples=args.valid_examples,
        test_examples=args.test_examples,
        seed=args.seed,
    )
    splits = load_splits(args.data_dir)
    modes = ["full", "dblock", "lora", "qlora"] if args.mode == "all" else [args.mode]
    results = {
        "task": "alpaca_sft_response_ce",
        "dataset": "tatsu-lab/alpaca",
        "data_dir": str(args.data_dir),
        "model": args.model,
        "qlora_model": args.qlora_model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "eval_batches": args.eval_batches,
        "split_counts": split_counts,
        "modes": {},
    }

    for mode in modes:
        try:
            if mode == "full":
                result = run_full(args, splits)
            elif mode == "dblock":
                result = run_dblock(args, splits)
            elif mode in {"lora", "qlora"}:
                result = run_lora_like(args, splits, mode=mode)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            results["modes"][mode] = result
            print(
                f"{mode} response_ce_before={result['response_ce_before']:.4f} "
                f"response_ce_after={result['response_ce_after']:.4f} "
                f"delta={result['response_ce_delta']:.4f} "
                f"seconds={result['seconds']:.2f} "
                f"peak_gb={result.get('peak_gb')}"
            )
        except Exception as exc:
            results["modes"][mode] = {"ok": False, "error": repr(exc)}
            print(f"{mode} failed error={exc!r}")
        clear_mlx_state()

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca SFT response-loss benchmark for full, blockwise, LoRA, and QLoRA"
    )
    parser.add_argument("--mode", choices=["full", "dblock", "lora", "qlora", "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--qlora_model", default=DEFAULT_QLORA_MODEL)
    parser.add_argument("--data_dir", type=Path, default=Path("corpora/alpaca_sft_smoke"))
    parser.add_argument("--adapter_root", type=Path, default=Path("outputs/sft_benchmark"))
    parser.add_argument("--train_examples", type=int, default=1024)
    parser.add_argument("--valid_examples", type=int, default=128)
    parser.add_argument("--test_examples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--qlora_lr", type=float, default=1e-4)
    parser.add_argument("--lora_optimizer", choices=["adam", "adamw", "muon", "sgd", "adafactor"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_num_layers", type=int, default=-1)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--objective", choices=["paper_ar", "hidden"], default="paper_ar")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--denoise_weight", type=float, default=0.0)
    parser.add_argument("--aux_lm_weight", type=float, default=0.1)
    parser.add_argument("--clean_lm_weight", type=float, default=100.0)
    parser.add_argument(
        "--clean_lm_anchor_profile",
        choices=["manual", "auto_window"],
        default="auto_window",
    )
    parser.add_argument("--clean_lm_interval", type=int, default=1)
    parser.add_argument("--clean_lm_seq_len", type=int, default=0)
    parser.add_argument("--clean_lm_window_count", type=int, default=1)
    parser.add_argument("--clean_lm_large_seq_len", type=int, default=0)
    parser.add_argument("--clean_lm_large_interval", type=int, default=0)
    parser.add_argument("--clean_lm_full_warmup_steps", type=int, default=10)
    parser.add_argument("--local_lm_weight", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_json", default=None)
    return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    os.makedirs(args.adapter_root, exist_ok=True)
    results = run_modes(args)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main_cli(argv: list[str] | None = None):
    main(build_parser().parse_args(argv))


if __name__ == "__main__":
    main_cli()
