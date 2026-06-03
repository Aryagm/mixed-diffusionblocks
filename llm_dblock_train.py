from __future__ import annotations

import argparse
import os
import random

import mlx.core as mx
import numpy as np

from llm_dblocks.adapters import TinyLMAdapter, load_mlx_lm_adapter
from llm_dblocks.data import ByteTokenizer, batch_iterator, load_text, tokenize_text
from llm_dblocks.tiny_lm import TinyLMConfig
from llm_dblocks.trainer import DBlockTrainer, DBlockTrainingConfig


def build_adapter_and_tokenizer(args):
    if args.backend == "tiny":
        config = TinyLMConfig(
            vocab_size=256,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            intermediate_size=args.intermediate_size,
            max_seq_len=args.seq_len,
        )
        return TinyLMAdapter(config), ByteTokenizer()

    load_kwargs = {}
    if args.adapter_path:
        load_kwargs["adapter_path"] = args.adapter_path
    return load_mlx_lm_adapter(args.model, **load_kwargs)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    adapter, tokenizer = build_adapter_and_tokenizer(args)
    text = load_text(args.data, text_field=args.text_field)
    tokens = tokenize_text(tokenizer, text)
    batches = batch_iterator(
        tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=True,
    )

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
    print(
        f"backend={args.backend} layers={adapter.num_layers} blocks={trainer.ranges} "
        f"tokens={len(tokens)} batch={args.batch_size} seq_len={args.seq_len} "
        f"clean_lm_anchor_profile={trainer.config.clean_lm_anchor_profile} "
        f"clean_lm_weight={trainer.config.clean_lm_weight} "
        f"denoise_weight={trainer.config.denoise_weight} "
        f"clean_lm_seq_len={trainer.config.clean_lm_seq_len} "
        f"clean_lm_large_seq_len={trainer.config.clean_lm_large_seq_len} "
        f"clean_lm_large_interval={trainer.config.clean_lm_large_interval}"
    )
    trainer.train(batches, iters=args.iters, log_every=args.log_every)
    if args.output_dir:
        path = trainer.save(args.output_dir)
        print(f"saved={path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DiffusionBlocks-style block-wise MLX trainer for causal LMs"
    )
    parser.add_argument("--backend", choices=["tiny", "mlx-lm"], default="tiny")
    parser.add_argument("--model", default=None, help="HF repo/local path for mlx-lm")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data", default="sample", help="'sample', .txt, or .jsonl")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--output_dir", default="llm_dblock_out")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    return parser


def main_cli(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    if args.backend == "mlx-lm" and args.model is None:
        raise ValueError("--model is required for --backend mlx-lm")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)


if __name__ == "__main__":
    main_cli()
