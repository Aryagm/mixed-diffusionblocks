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


def clean_anchor_config(config: DBlockTrainingConfig) -> dict[str, float | int | str]:
    return {
        "clean_lm_weight": config.clean_lm_weight,
        "denoise_weight": config.denoise_weight,
        "clean_lm_anchor_profile": config.clean_lm_anchor_profile,
        "clean_lm_interval": config.clean_lm_interval,
        "clean_lm_seq_len": config.clean_lm_seq_len,
        "clean_lm_window_count": config.clean_lm_window_count,
        "clean_lm_large_seq_len": config.clean_lm_large_seq_len,
        "clean_lm_large_interval": config.clean_lm_large_interval,
        "clean_lm_full_warmup_steps": config.clean_lm_full_warmup_steps,
        "local_lm_weight": config.local_lm_weight,
    }


def gb(nbytes: int | float) -> float:
    return float(nbytes) / (1024**3)


def scalar(x) -> float:
    return float(x.item())


def reset_memory():
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
        mx.reset_peak_memory()
    else:
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()


def memory_snapshot() -> dict[str, float]:
    if hasattr(mx, "get_active_memory"):
        return {
            "active_gb": gb(mx.get_active_memory()),
            "cache_gb": gb(mx.get_cache_memory()),
            "peak_gb": gb(mx.get_peak_memory()),
        }
    return {
        "active_gb": gb(mx.metal.get_active_memory()),
        "cache_gb": gb(mx.metal.get_cache_memory()),
        "peak_gb": gb(mx.metal.get_peak_memory()),
    }


def full_finetune_step(adapter, batch, lr: float, weight_decay: float):
    adapter.model.unfreeze()
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, adapter.vocab_size),
            batch["labels"].reshape(-1),
            reduction="mean",
        )
        return loss

    loss_and_grad = nn.value_and_grad(adapter.model, loss_fn)
    reset_memory()
    start = time.perf_counter()
    loss, grads = loss_and_grad(adapter.model, batch)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    optimizer.update(adapter.model, grads)
    mx.eval(adapter.model.parameters(), optimizer.state, loss)
    elapsed = time.perf_counter() - start
    return scalar(loss), elapsed, memory_snapshot()


def dblock_step(adapter, batch, args):
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
    block_idx = args.block_idx
    if block_idx is None:
        block_idx = random.randrange(args.num_blocks)
    trainer.set_trainable_block(block_idx)

    def loss_fn(model, batch):
        loss, _ = trainer.loss(
            model,
            batch,
            block_idx=block_idx,
            anchor_step=1,
        )
        return loss

    loss_and_grad = nn.value_and_grad(adapter.model, loss_fn)
    reset_memory()
    start = time.perf_counter()
    loss, grads = loss_and_grad(adapter.model, batch)
    grads, _ = optim.clip_grad_norm(grads, args.gradient_clip_norm)
    optimizer = trainer.optimizers[block_idx]
    optimizer.update(adapter.model, grads)
    mx.eval(adapter.model.parameters(), optimizer.state, loss)
    elapsed = time.perf_counter() - start
    return (
        scalar(loss),
        elapsed,
        memory_snapshot(),
        trainer.ranges[block_idx],
        clean_anchor_config(trainer.config),
    )


def load_adapter_and_batch(args):
    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    text = load_text(args.data, text_field=args.text_field)
    tokens = tokenize_text(tokenizer, text)
    batch = next(
        batch_iterator(
            tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            shuffle=False,
        )
    )
    mx.eval(batch["input_ids"], batch["labels"], adapter.model.parameters())
    return adapter, batch


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    adapter, batch = load_adapter_and_batch(args)

    results = {
        "model": args.model,
        "layers": adapter.num_layers,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_blocks": args.num_blocks,
        "objective": args.objective,
        "denoise_weight": args.denoise_weight,
        "clean_lm_weight": args.clean_lm_weight,
        "clean_lm_anchor_profile": args.clean_lm_anchor_profile,
        "clean_lm_interval": args.clean_lm_interval,
        "clean_lm_seq_len": args.clean_lm_seq_len,
        "clean_lm_window_count": args.clean_lm_window_count,
        "clean_lm_large_seq_len": args.clean_lm_large_seq_len,
        "clean_lm_large_interval": args.clean_lm_large_interval,
        "clean_lm_full_warmup_steps": args.clean_lm_full_warmup_steps,
        "local_lm_weight": args.local_lm_weight,
        "modes": {},
    }
    print(
        f"model={args.model} layers={adapter.num_layers} "
        f"batch={args.batch_size} seq_len={args.seq_len}"
    )

    if args.mode in {"full", "both"}:
        try:
            adapter, batch = load_adapter_and_batch(args)
            loss, elapsed, memory = full_finetune_step(
                adapter, batch, args.lr, args.weight_decay
            )
            results["modes"]["full"] = {
                "ok": True,
                "loss": loss,
                "seconds": elapsed,
                **memory,
            }
            print(
                f"full ok loss={loss:.4f} seconds={elapsed:.2f} "
                f"peak_gb={memory['peak_gb']:.2f}"
            )
        except Exception as exc:
            results["modes"]["full"] = {
                "ok": False,
                "error": repr(exc),
                **memory_snapshot(),
            }
            print(f"full failed error={exc!r}")

    if args.mode in {"dblock", "both"}:
        try:
            adapter, batch = load_adapter_and_batch(args)
            loss, elapsed, memory, block_range, anchor_config = dblock_step(
                adapter, batch, args
            )
            results["modes"]["dblock"] = {
                "ok": True,
                "loss": loss,
                "seconds": elapsed,
                "block_range": block_range,
                **anchor_config,
                **memory,
            }
            print(
                f"dblock ok block={block_range} loss={loss:.4f} seconds={elapsed:.2f} "
                f"peak_gb={memory['peak_gb']:.2f}"
            )
        except Exception as exc:
            results["modes"]["dblock"] = {
                "ok": False,
                "error": repr(exc),
                **memory_snapshot(),
            }
            print(f"dblock failed error={exc!r}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full fine-tuning vs DiffusionBlocks peak memory on mlx-lm models"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="sample")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--mode", choices=["full", "dblock", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--block_idx", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--denoise_weight", type=float, default=1.0)
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
    parser.add_argument("--clean_lm_full_warmup_steps", type=int, default=0)
    parser.add_argument("--local_lm_weight", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--objective", choices=["paper_ar", "hidden"], default="paper_ar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    return parser


def main_cli(argv: list[str] | None = None):
    main(build_parser().parse_args(argv))


if __name__ == "__main__":
    main_cli()
