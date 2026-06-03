from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from llm_dblocks.adapters import load_mlx_lm_adapter
from llm_dblocks.data import load_text, tokenize_text
from llm_dblocks.speculative import (
    EarlyExitDraftModel,
    prefix_reuse_speculative_generate_step,
)


mlx_generate = importlib.import_module("mlx_lm.generate")


def parse_csv_ints(value: str) -> list[int]:
    values = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise ValueError(f"empty value in integer list: {value!r}")
        values.append(int(item))
    return values


def default_exit_layers(num_layers: int) -> list[int]:
    fractions = (1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 5 / 6)
    layers = {
        min(max(int(round(num_layers * fraction)), 1), num_layers - 1)
        for fraction in fractions
    }
    return sorted(layers)


def reset_memory():
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
        mx.reset_peak_memory()
    else:
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()


def peak_gb() -> float:
    if hasattr(mx, "get_peak_memory"):
        return float(mx.get_peak_memory()) / 1e9
    return float(mx.metal.get_peak_memory()) / 1e9


@dataclass
class RunStats:
    tokens: list[int]
    total_seconds: float
    first_token_seconds: float
    from_draft: int
    peak_gb: float

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / max(self.total_seconds, 1e-9)

    @property
    def draft_fraction(self) -> float:
        return self.from_draft / max(len(self.tokens), 1)

    def to_json(self) -> dict:
        return {
            "tokens": self.tokens,
            "num_tokens": len(self.tokens),
            "total_seconds": self.total_seconds,
            "first_token_seconds": self.first_token_seconds,
            "tokens_per_second": self.tokens_per_second,
            "from_draft": self.from_draft,
            "draft_fraction": self.draft_fraction,
            "peak_gb": self.peak_gb,
        }


def consume(generator: Iterable) -> RunStats:
    tokens: list[int] = []
    from_draft = 0
    first_token_seconds = 0.0
    start = time.perf_counter()
    for item in generator:
        now = time.perf_counter()
        if not tokens:
            first_token_seconds = now - start
        token = item[0]
        is_draft = bool(item[2]) if len(item) >= 3 else False
        tokens.append(int(token))
        from_draft += int(is_draft)
    total_seconds = time.perf_counter() - start
    return RunStats(
        tokens=tokens,
        total_seconds=total_seconds,
        first_token_seconds=first_token_seconds,
        from_draft=from_draft,
        peak_gb=peak_gb(),
    )


def best_run(
    factory: Callable[[], Iterable],
    *,
    repeats: int,
    warmup: int,
) -> RunStats:
    for _ in range(warmup):
        reset_memory()
        consume(factory())

    runs = []
    for _ in range(repeats):
        reset_memory()
        runs.append(consume(factory()))
    return min(runs, key=lambda run: run.total_seconds)


def prompt_tokens(args, tokenizer) -> mx.array:
    if args.prompt is not None:
        tokens = tokenizer.encode(args.prompt)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids
        values = np.asarray(tokens, dtype=np.int32)
    else:
        text = load_text(args.data, text_field=args.text_field)
        values = tokenize_text(tokenizer, text)

    if len(values) < args.prompt_len:
        repeats = int(np.ceil(args.prompt_len / max(len(values), 1)))
        values = np.tile(values, repeats)
    return mx.array(values[: args.prompt_len], dtype=mx.int32)


def full_generator(adapter, prompt: mx.array, args):
    return mlx_generate.generate_step(
        prompt,
        adapter.model,
        max_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
    )


def draft_generator(draft_model, prompt: mx.array, args):
    return mlx_generate.generate_step(
        prompt,
        draft_model,
        max_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
    )


def speculative_generator(adapter, draft_model, prompt: mx.array, draft_tokens: int, args):
    return mlx_generate.speculative_generate_step(
        prompt,
        adapter.model,
        draft_model,
        num_draft_tokens=draft_tokens,
        max_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
    )


def prefix_reuse_generator(adapter, prompt: mx.array, exit_layer: int, draft_tokens: int, args):
    return prefix_reuse_speculative_generate_step(
        prompt,
        adapter,
        exit_layer=exit_layer,
        num_draft_tokens=draft_tokens,
        max_tokens=args.max_new_tokens,
        prefill_step_size=args.prefill_step_size,
    )


def run(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    adapter, tokenizer = load_mlx_lm_adapter(args.model)
    adapter.model.eval()
    prompt = prompt_tokens(args, tokenizer)
    mx.eval(prompt, adapter.model.parameters())

    exit_layers = (
        parse_csv_ints(args.exit_layers)
        if args.exit_layers
        else default_exit_layers(adapter.num_layers)
    )
    draft_token_counts = parse_csv_ints(args.draft_tokens)

    results = {
        "model": args.model,
        "num_layers": adapter.num_layers,
        "prompt_len": int(prompt.shape[0]),
        "max_new_tokens": args.max_new_tokens,
        "prefill_step_size": args.prefill_step_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "exit_layers": exit_layers,
        "draft_tokens": draft_token_counts,
        "full": {},
        "draft_only": [],
        "speculative": [],
        "prefix_reuse_speculative": [],
    }

    full = best_run(
        lambda: full_generator(adapter, prompt, args),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results["full"] = full.to_json()
    print(
        f"full tps={full.tokens_per_second:.2f} "
        f"seconds={full.total_seconds:.3f} peak_gb={full.peak_gb:.2f}"
    )

    for exit_layer in exit_layers:
        draft_model = EarlyExitDraftModel(adapter, exit_layer=exit_layer)
        draft = best_run(
            lambda draft_model=draft_model: draft_generator(draft_model, prompt, args),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        draft_row = {
            "exit_layer": exit_layer,
            "layer_fraction": exit_layer / adapter.num_layers,
            **draft.to_json(),
            "speedup_vs_full": draft.tokens_per_second / full.tokens_per_second,
        }
        results["draft_only"].append(draft_row)
        print(
            f"draft exit={exit_layer}/{adapter.num_layers} "
            f"tps={draft.tokens_per_second:.2f} "
            f"speedup={draft_row['speedup_vs_full']:.2f}x"
        )

        for num_draft_tokens in draft_token_counts:
            try:
                spec = best_run(
                    lambda draft_model=draft_model, num_draft_tokens=num_draft_tokens: speculative_generator(
                        adapter,
                        draft_model,
                        prompt,
                        num_draft_tokens,
                        args,
                    ),
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                row = {
                    "exit_layer": exit_layer,
                    "layer_fraction": exit_layer / adapter.num_layers,
                    "num_draft_tokens": num_draft_tokens,
                    **spec.to_json(),
                    "speedup_vs_full": spec.tokens_per_second / full.tokens_per_second,
                }
            except Exception as exc:
                row = {
                    "exit_layer": exit_layer,
                    "layer_fraction": exit_layer / adapter.num_layers,
                    "num_draft_tokens": num_draft_tokens,
                    "ok": False,
                    "error": repr(exc),
                }
            results["speculative"].append(row)
            if row.get("ok") is False:
                print(
                    f"spec exit={exit_layer}/{adapter.num_layers} "
                    f"draft={num_draft_tokens} failed error={row['error']}"
                )
            else:
                print(
                    f"spec exit={exit_layer}/{adapter.num_layers} "
                    f"draft={num_draft_tokens} tps={row['tokens_per_second']:.2f} "
                    f"speedup={row['speedup_vs_full']:.2f}x "
                    f"accepted={row['draft_fraction']:.2%}"
                )

            try:
                reuse = best_run(
                    lambda exit_layer=exit_layer, num_draft_tokens=num_draft_tokens: prefix_reuse_generator(
                        adapter,
                        prompt,
                        exit_layer,
                        num_draft_tokens,
                        args,
                    ),
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                reuse_row = {
                    "exit_layer": exit_layer,
                    "layer_fraction": exit_layer / adapter.num_layers,
                    "num_draft_tokens": num_draft_tokens,
                    **reuse.to_json(),
                    "speedup_vs_full": reuse.tokens_per_second
                    / full.tokens_per_second,
                }
            except Exception as exc:
                reuse_row = {
                    "exit_layer": exit_layer,
                    "layer_fraction": exit_layer / adapter.num_layers,
                    "num_draft_tokens": num_draft_tokens,
                    "ok": False,
                    "error": repr(exc),
                }
            results["prefix_reuse_speculative"].append(reuse_row)
            if reuse_row.get("ok") is False:
                print(
                    f"reuse exit={exit_layer}/{adapter.num_layers} "
                    f"draft={num_draft_tokens} failed error={reuse_row['error']}"
                )
            else:
                print(
                    f"reuse exit={exit_layer}/{adapter.num_layers} "
                    f"draft={num_draft_tokens} "
                    f"tps={reuse_row['tokens_per_second']:.2f} "
                    f"speedup={reuse_row['speedup_vs_full']:.2f}x "
                    f"accepted={reuse_row['draft_fraction']:.2%}"
                )

    ok_specs = [row for row in results["speculative"] if row.get("ok", True)]
    if ok_specs:
        results["best_speculative"] = max(
            ok_specs,
            key=lambda row: row["tokens_per_second"],
        )
    ok_reuse_specs = [
        row for row in results["prefix_reuse_speculative"] if row.get("ok", True)
    ]
    if ok_reuse_specs:
        results["best_prefix_reuse_speculative"] = max(
            ok_reuse_specs,
            key=lambda row: row["tokens_per_second"],
        )
    return results


def main(args):
    results = run(args)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MLX early-exit self-speculative decoding"
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--data", default="sample")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prefill_step_size", type=int, default=512)
    parser.add_argument("--exit_layers", default=None)
    parser.add_argument("--draft_tokens", default="2,4,6,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    main(parser.parse_args())
