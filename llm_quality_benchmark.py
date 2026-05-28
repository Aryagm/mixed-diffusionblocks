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

from llm_dblocks.adapters import TinyLMAdapter
from llm_dblocks.data import ByteTokenizer, batch_iterator, load_text, tokenize_text
from llm_dblocks.tiny_lm import TinyLMConfig
from llm_dblocks.trainer import DBlockTrainer, DBlockTrainingConfig


def scalar(x) -> float:
    return float(x.item())


def make_tiny_adapter(args):
    return TinyLMAdapter(
        TinyLMConfig(
            vocab_size=256,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            intermediate_size=args.intermediate_size,
            max_seq_len=args.seq_len,
        )
    )


def eval_full_lm(model, batches, num_batches: int) -> float:
    total = 0.0
    for _ in range(num_batches):
        batch = next(batches)
        logits = model(batch["input_ids"])
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, 256),
            batch["labels"].reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        total += scalar(loss)
    return total / num_batches


def train_full_lm(model, batches, steps: int, lr: float) -> float:
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return nn.losses.cross_entropy(
            logits.reshape(-1, 256),
            batch["labels"].reshape(-1),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    start = time.perf_counter()
    for _ in range(steps):
        batch = next(batches)
        loss, grads = loss_and_grad(model, batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    return time.perf_counter() - start


def eval_dblocks(trainer: DBlockTrainer, batches, num_batches: int) -> float:
    total = 0.0
    for i in range(num_batches):
        batch = next(batches)
        block_idx = i % trainer.config.num_blocks
        loss, _ = trainer.loss(trainer.adapter.model, batch, block_idx=block_idx)
        mx.eval(loss)
        total += scalar(loss)
    return total / num_batches


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    text = load_text(args.data, text_field=args.text_field)
    tokenizer = ByteTokenizer()
    tokens = tokenize_text(tokenizer, text)
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    results = {
        "backend": "tiny",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_blocks": args.num_blocks,
    }

    if args.mode in {"full", "both"}:
        full = make_tiny_adapter(args)
        train_batches = batch_iterator(train_tokens, args.batch_size, args.seq_len)
        val_batches = batch_iterator(val_tokens, args.batch_size, args.seq_len)
        before = eval_full_lm(full.model, val_batches, args.eval_batches)
        seconds = train_full_lm(full.model, train_batches, args.steps, args.lr)
        after = eval_full_lm(full.model, val_batches, args.eval_batches)
        results["full"] = {
            "before": before,
            "after": after,
            "delta": before - after,
            "seconds": seconds,
        }
        print(f"full before={before:.4f} after={after:.4f} delta={before-after:.4f}")

    if args.mode in {"dblock", "both"}:
        dblock = make_tiny_adapter(args)
        config = DBlockTrainingConfig(
            num_blocks=args.num_blocks,
            lr=args.lr,
            weight_decay=0.01,
            objective=args.objective,
            sigma_max=args.sigma_max,
            clean_lm_weight=args.clean_lm_weight,
        )
        trainer = DBlockTrainer(dblock, config)
        train_batches = batch_iterator(train_tokens, args.batch_size, args.seq_len)
        val_batches = batch_iterator(val_tokens, args.batch_size, args.seq_len)
        before = eval_dblocks(trainer, val_batches, args.eval_batches)
        start = time.perf_counter()
        trainer.train(train_batches, iters=args.steps, log_every=args.steps)
        seconds = time.perf_counter() - start
        after = eval_dblocks(trainer, val_batches, args.eval_batches)
        results["dblock"] = {
            "objective": args.objective,
            "clean_lm_weight": args.clean_lm_weight,
            "before": before,
            "after": after,
            "delta": before - after,
            "seconds": seconds,
        }
        print(
            f"dblock objective={args.objective} before={before:.4f} "
            f"after={after:.4f} delta={before-after:.4f}"
        )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "dblock", "both"], default="both")
    parser.add_argument("--objective", choices=["paper_ar", "hidden"], default="paper_ar")
    parser.add_argument("--data", default="sample")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--clean_lm_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--output_json", default=None)
    main(parser.parse_args())
