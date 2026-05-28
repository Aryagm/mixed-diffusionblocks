from __future__ import annotations

import argparse
import json
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from llm_dblocks.adapters import load_mlx_lm_adapter
from llm_dblocks.data import batch_iterator, load_text, tokenize_text
from llm_dblocks.trainer import DBlockTrainer, DBlockTrainingConfig


def scalar(x) -> float:
    return float(x.item())


def split_tokens(tokens: np.ndarray, val_fraction: float = 0.1):
    split = max(int(len(tokens) * (1.0 - val_fraction)), 1)
    return tokens[:split], tokens[split:]


def eval_full(adapter, batches, num_batches: int) -> float:
    adapter.model.eval()
    total = 0.0
    for _ in range(num_batches):
        batch = next(batches)
        logits = adapter.model(batch["input_ids"])
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, adapter.vocab_size),
            batch["labels"].reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        total += scalar(loss)
    return total / num_batches


def eval_next_token_ce(adapter, batches, num_batches: int) -> float:
    return eval_full(adapter, batches, num_batches)


def train_full(adapter, batches, steps: int, lr: float):
    adapter.model.unfreeze()
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return nn.losses.cross_entropy(
            logits.reshape(-1, adapter.vocab_size),
            batch["labels"].reshape(-1),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(adapter.model, loss_fn)
    start = time.perf_counter()
    for step in range(steps):
        batch = next(batches)
        loss, grads = loss_and_grad(adapter.model, batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(adapter.model, grads)
        mx.eval(adapter.model.parameters(), optimizer.state, loss)
    return time.perf_counter() - start


def eval_dblock_metrics(
    trainer: DBlockTrainer,
    batches,
    num_batches: int,
) -> dict[str, float]:
    trainer.adapter.model.eval()
    totals: dict[str, float] = {}
    for i in range(num_batches):
        batch = next(batches)
        block_idx = i % trainer.config.num_blocks
        _, metrics = trainer.loss(trainer.adapter.model, batch, block_idx=block_idx)
        metric_values = {
            key: value
            for key, value in metrics.items()
            if key in {"loss", "denoise_loss", "aux_lm_loss", "clean_lm_loss"}
        }
        mx.eval(*metric_values.values())
        for key, value in metric_values.items():
            totals[key] = totals.get(key, 0.0) + scalar(value)
    return {key: value / num_batches for key, value in totals.items()}


def eval_dblock(trainer: DBlockTrainer, batches, num_batches: int) -> float:
    return eval_dblock_metrics(trainer, batches, num_batches)["loss"]


def eval_dblock_diffusion(
    trainer: DBlockTrainer,
    batches,
    num_batches: int,
    num_steps: int,
) -> float:
    trainer.adapter.model.eval()
    total = 0.0
    for _ in range(num_batches):
        batch = next(batches)
        loss = trainer.diffusion_reconstruction_ce(batch, num_steps=num_steps)
        mx.eval(loss)
        total += scalar(loss)
    return total / num_batches


def train_dblock(trainer: DBlockTrainer, batches, steps: int, log_every: int):
    start = time.perf_counter()
    trainer.train(batches, iters=steps, log_every=log_every)
    return time.perf_counter() - start


def load_tokens(args, tokenizer):
    text = load_text(args.data, text_field=args.text_field)
    train_tokens = tokenize_text(tokenizer, text)
    if args.max_tokens and len(train_tokens) > args.max_tokens:
        train_tokens = train_tokens[: args.max_tokens]

    if args.val_data:
        val_text = load_text(args.val_data, text_field=args.text_field)
        val_tokens = tokenize_text(tokenizer, val_text)
        if args.max_tokens and len(val_tokens) > args.max_tokens:
            val_tokens = val_tokens[: args.max_tokens]
        return train_tokens, val_tokens

    return split_tokens(train_tokens, args.val_fraction)


def run_full(args):
    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    train_tokens, val_tokens = load_tokens(args, tokenizer)
    train_batches = batch_iterator(train_tokens, args.batch_size, args.seq_len)
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    before = eval_full(adapter, val_batches, args.eval_batches)
    seconds = train_full(adapter, train_batches, args.steps, args.lr)
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    after = eval_full(adapter, val_batches, args.eval_batches)
    return {
        "ok": True,
        "before": before,
        "after": after,
        "delta": before - after,
        "seconds": seconds,
    }


def run_dblock(args):
    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    train_tokens, val_tokens = load_tokens(args, tokenizer)
    config = DBlockTrainingConfig(
        num_blocks=args.num_blocks,
        lr=args.lr,
        weight_decay=0.01,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gamma=args.gamma,
        aux_lm_weight=args.aux_lm_weight,
        clean_lm_weight=args.clean_lm_weight,
        objective=args.objective,
    )
    trainer = DBlockTrainer(adapter, config)
    train_batches = batch_iterator(train_tokens, args.batch_size, args.seq_len)
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    before_metrics = eval_dblock_metrics(trainer, val_batches, args.eval_batches)
    before = before_metrics["loss"]
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    diffusion_before = eval_dblock_diffusion(
        trainer, val_batches, args.eval_batches, args.diffusion_eval_steps
    )
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    full_before = eval_next_token_ce(adapter, val_batches, args.eval_batches)
    seconds = train_dblock(trainer, train_batches, args.steps, args.log_every)
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    after_metrics = eval_dblock_metrics(trainer, val_batches, args.eval_batches)
    after = after_metrics["loss"]
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    diffusion_after = eval_dblock_diffusion(
        trainer, val_batches, args.eval_batches, args.diffusion_eval_steps
    )
    val_batches = batch_iterator(
        val_tokens, args.batch_size, args.seq_len, shuffle=False
    )
    full_after = eval_next_token_ce(adapter, val_batches, args.eval_batches)
    return {
        "ok": True,
        "objective": args.objective,
        "clean_lm_weight": args.clean_lm_weight,
        "before": before,
        "after": after,
        "delta": before - after,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "diffusion_ce_before": diffusion_before,
        "diffusion_ce_after": diffusion_after,
        "diffusion_ce_delta": diffusion_before - diffusion_after,
        "next_token_ce_before": full_before,
        "next_token_ce_after": full_after,
        "next_token_ce_delta": full_before - full_after,
        "seconds": seconds,
    }


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    results = {
        "model": args.model,
        "data": args.data,
        "val_data": args.val_data,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "eval_batches": args.eval_batches,
        "max_tokens": args.max_tokens,
        "modes": {},
    }

    if args.mode in {"full", "both"}:
        try:
            results["modes"]["full"] = run_full(args)
            r = results["modes"]["full"]
            print(
                f"full before={r['before']:.4f} after={r['after']:.4f} "
                f"delta={r['delta']:.4f} seconds={r['seconds']:.2f}"
            )
        except Exception as exc:
            results["modes"]["full"] = {"ok": False, "error": repr(exc)}
            print(f"full failed error={exc!r}")

    if args.mode in {"dblock", "both"}:
        try:
            results["modes"]["dblock"] = run_dblock(args)
            r = results["modes"]["dblock"]
            print(
                f"dblock objective={r['objective']} before={r['before']:.4f} "
                f"after={r['after']:.4f} delta={r['delta']:.4f} "
                f"denoise_after={r['metrics_after']['denoise_loss']:.4f} "
                f"clean_after={r['metrics_after']['clean_lm_loss']:.4f} "
                f"diff_before={r['diffusion_ce_before']:.4f} "
                f"diff_after={r['diffusion_ce_after']:.4f} "
                f"ntp_before={r['next_token_ce_before']:.4f} "
                f"ntp_after={r['next_token_ce_after']:.4f} "
                f"clean_lm_weight={r['clean_lm_weight']:.4f} "
                f"seconds={r['seconds']:.2f}"
            )
        except Exception as exc:
            results["modes"]["dblock"] = {"ok": False, "error": repr(exc)}
            print(f"dblock failed error={exc!r}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Small Qwen quality benchmark for full bf16 vs DiffusionBlocks"
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--mode", choices=["full", "dblock", "both"], default="both")
    parser.add_argument("--objective", choices=["paper_ar", "hidden"], default="paper_ar")
    parser.add_argument("--data", default="sample")
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--aux_lm_weight", type=float, default=0.1)
    parser.add_argument("--clean_lm_weight", type=float, default=0.0)
    parser.add_argument("--diffusion_eval_steps", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    main(parser.parse_args())
