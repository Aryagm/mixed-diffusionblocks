from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    command: tuple[str, ...]
    output_json: str
    heavy: bool = False


def model_slug(model: str) -> str:
    name = model.rsplit("/", 1)[-1].lower()
    qwen_match = re.search(r"qwen2\.5-(\d+(?:\.\d+)?)b", name)
    if qwen_match:
        size = qwen_match.group(1).replace(".", "")
        return f"qwen25_{size}b"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _common_quality_args(
    *,
    model: str,
    data: str,
    val_data: str,
    steps: int,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    max_tokens: int,
    num_blocks: int,
    seed: int,
    output_json: str,
) -> list[str]:
    return [
        "mixed-dblocks-quality",
        "--model",
        model,
        "--data",
        data,
        "--val_data",
        val_data,
        "--steps",
        str(steps),
        "--eval_batches",
        str(eval_batches),
        "--batch_size",
        str(batch_size),
        "--seq_len",
        str(seq_len),
        "--num_blocks",
        str(num_blocks),
        "--max_tokens",
        str(max_tokens),
        "--seed",
        str(seed),
        "--output_json",
        output_json,
    ]


def publish_matrix(
    *,
    models: list[str],
    seeds: list[int],
    data: str,
    val_data: str,
    output_dir: str,
    include_memory: bool = False,
    steps: int = 100,
    seq_len: int = 1024,
    batch_size: int = 1,
    eval_batches: int = 4,
    max_tokens: int = 400_000,
    num_blocks: int = 8,
    memory_seq_len: int = 2048,
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    out = Path(output_dir)

    for model in models:
        slug = model_slug(model)
        for seed in seeds:
            base = {
                "model": model,
                "data": data,
                "val_data": val_data,
                "steps": steps,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "eval_batches": eval_batches,
                "max_tokens": max_tokens,
                "num_blocks": num_blocks,
                "seed": seed,
            }

            full_name = f"{slug}_full_seed{seed}"
            full_output = str(out / f"{full_name}.json")
            specs.append(
                ExperimentSpec(
                    name=full_name,
                    output_json=full_output,
                    command=tuple(
                        _common_quality_args(output_json=full_output, **base)
                        + ["--mode", "full"]
                    ),
                )
            )

            pure_name = f"{slug}_pure_dblocks_seed{seed}"
            pure_output = str(out / f"{pure_name}.json")
            specs.append(
                ExperimentSpec(
                    name=pure_name,
                    output_json=pure_output,
                    command=tuple(
                        _common_quality_args(output_json=pure_output, **base)
                        + [
                            "--mode",
                            "dblock",
                            "--clean_lm_anchor_profile",
                            "manual",
                            "--clean_lm_weight",
                            "0",
                            "--denoise_weight",
                            "1",
                        ]
                    ),
                )
            )

            full_anchor_name = f"{slug}_full_anchor_mixed_seed{seed}"
            full_anchor_output = str(out / f"{full_anchor_name}.json")
            specs.append(
                ExperimentSpec(
                    name=full_anchor_name,
                    output_json=full_anchor_output,
                    command=tuple(
                        _common_quality_args(output_json=full_anchor_output, **base)
                        + [
                            "--mode",
                            "dblock",
                            "--clean_lm_anchor_profile",
                            "manual",
                            "--clean_lm_weight",
                            "100",
                            "--clean_lm_seq_len",
                            "0",
                            "--denoise_weight",
                            "1",
                        ]
                    ),
                )
            )

            auto_name = f"{slug}_auto_window_seed{seed}"
            auto_output = str(out / f"{auto_name}.json")
            specs.append(
                ExperimentSpec(
                    name=auto_name,
                    output_json=auto_output,
                    command=tuple(
                        _common_quality_args(output_json=auto_output, **base)
                        + [
                            "--mode",
                            "dblock",
                            "--clean_lm_anchor_profile",
                            "auto_window",
                            "--clean_lm_weight",
                            "100",
                            "--denoise_weight",
                            "0",
                            "--clean_lm_full_warmup_steps",
                            "10",
                        ]
                    ),
                )
            )

        if include_memory:
            memory_name = f"{slug}_auto_window_memory_seq{memory_seq_len}"
            memory_output = str(out / f"{memory_name}.json")
            specs.append(
                ExperimentSpec(
                    name=memory_name,
                    output_json=memory_output,
                    heavy=True,
                    command=(
                        "mixed-dblocks-memory",
                        "--mode",
                        "dblock",
                        "--model",
                        model,
                        "--batch_size",
                        str(batch_size),
                        "--seq_len",
                        str(memory_seq_len),
                        "--num_blocks",
                        str(num_blocks),
                        "--objective",
                        "paper_ar",
                        "--clean_lm_anchor_profile",
                        "auto_window",
                        "--clean_lm_weight",
                        "100",
                        "--denoise_weight",
                        "0",
                        "--output_json",
                        memory_output,
                    ),
                )
            )

    return specs


def render_commands(specs: list[ExperimentSpec]) -> list[str]:
    return [shlex.join(spec.command) for spec in specs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or run the Mixed DiffusionBlocks publish matrix"
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Model repo/path. Can be passed multiple times.",
    )
    parser.add_argument("--seed", action="append", dest="seeds", type=int, default=[])
    parser.add_argument("--data", default="corpora/wikitext2/train.txt")
    parser.add_argument("--val_data", default="corpora/wikitext2/validation.txt")
    parser.add_argument("--output_dir", default="docs/results/publish")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=400_000)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--include_memory", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0)
    return parser


def main_cli(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    models = args.models or ["mlx-community/Qwen2.5-0.5B-Instruct-bf16"]
    seeds = args.seeds or [42]
    specs = publish_matrix(
        models=models,
        seeds=seeds,
        data=args.data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        include_memory=args.include_memory,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        max_tokens=args.max_tokens,
        num_blocks=args.num_blocks,
    )
    if args.max_runs > 0:
        specs = specs[: args.max_runs]

    commands = render_commands(specs)
    if not args.run:
        for command in commands:
            print(command)
        return

    for spec, command in zip(specs, commands, strict=True):
        print(command)
        subprocess.run(spec.command, check=True)


if __name__ == "__main__":
    main_cli()
