# Mixed DiffusionBlocks for Apple Silicon LLM Fine-Tuning

Mixed DiffusionBlocks is an experimental MLX training path for fine-tuning
large bf16 language-model blocks on Apple Silicon. The default LLM path now
uses **Windowed Mixed DiffusionBlocks**: exact autoregressive CE on deterministic
clean-token windows, while still training one full transformer block at a time.
For full fine-tuning, the default disables the denoising term after evidence
showed the AR anchor is the quality-critical part.

It extends SakanaAI's DiffusionBlocks idea with a practical LLM objective:

```text
loss = diffusionblocks_denoising_loss + clean_lm_weight * next_token_ce
```

Pure DiffusionBlocks gives the blockwise memory reduction, but in our LLM tests
it often hurts ordinary next-token CE. Mixed DiffusionBlocks adds standard
autoregressive CE while still training only the selected transformer block.
Windowed Mixed keeps that CE anchor local in sequence length so it is much
cheaper and close to pure DiffusionBlocks memory.

This is not the official SakanaAI repository. It is an MLX/LLM adaptation and
extension built on top of the public
[SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks) code and
paper.

## Why This Matters

On an M4 Max MacBook Pro with 36 GB unified memory:

| Setup | Full bf16 AR | Mixed DiffusionBlocks |
| --- | --- | --- |
| Qwen2.5-7B, seq512 | killed by macOS | 19.21 GB peak |
| Qwen2.5-7B, seq2048 | not viable after seq512 failure | 29.93 GB peak |

The value proposition is not "faster than full fine-tuning when full
fine-tuning fits." The value proposition is:

> Train actual bf16 transformer blocks locally when full bf16 backpropagation
> does not fit.

## Key Results

### Real Held-Out Data: WikiText-2

Qwen2.5-0.5B, WikiText-2 train/validation, batch 1, 100 steps:

| Context | Method | Validation CE before | Validation CE after | Time |
| ---: | --- | ---: | ---: | ---: |
| 128 | Full bf16 AR | 2.4844 | 2.5000 | 16.45 s |
| 128 | Pure DiffusionBlocks | 2.4844 | 3.0469 | 5.46 s |
| 128 | Mixed DiffusionBlocks | 2.4844 | 2.3750 | 11.35 s |
| 1024 | Full bf16 AR | 2.1094 | 1.9922 | 49.62 s |
| 1024 | Pure DiffusionBlocks | 2.1094 | 4.6875 | 41.91 s |
| 1024 | Mixed DiffusionBlocks | 2.1094 | 1.9453 | 77.49 s |

Qwen2.5-1.5B, WikiText-2, seq1024, batch 1, 100 steps:

| Method | Validation CE before | Validation CE after | Time |
| --- | ---: | ---: | ---: |
| Full bf16 AR | 1.7344 | 1.5781 | 140.11 s |
| Pure DiffusionBlocks | 1.7344 | 1.9609 | 103.56 s |
| Mixed DiffusionBlocks, full clean anchor | 1.7344 | 1.5859 | 197.90 s |
| Windowed Mixed, auto-window default | 1.7344 | 1.6016 | 115.97 s |
| Fast Windowed Blockwise, current default | 1.7344 | 1.5859 | 27.78 s |

Takeaway: pure DiffusionBlocks learns denoising but degrades ordinary LM CE.
Mixed DiffusionBlocks fixes that failure mode and preserves normal forward-pass
behavior. Windowed Mixed is the practical default when memory and wall-clock
matter; full-sequence anchoring remains the quality ablation.

### Memory

Same M4 Max 36 GB machine:

| Model / seq | Full bf16 AR | Pure DiffusionBlocks | Mixed DiffusionBlocks |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B seq128 | 4.39 GB | 1.56 GB | 1.57 GB |
| Qwen2.5-1.5B seq128 | 12.75 GB | 4.11 GB | 4.05 GB |
| Qwen2.5-3B seq128 | 24.59 GB | 8.47 GB | 8.47 GB |
| Qwen2.5-7B seq2048 | killed before seq512 full run finished | 22.35 GB | 29.93 GB |

With a 128-token windowed clean anchor at seq2048, Qwen2.5-7B used 22.31 GB
peak and 6.61 s for the one-step memory test, effectively matching pure
DiffusionBlocks memory while keeping an exact clean CE anchor.

With the current fast default (`denoise_weight=0`), Qwen2.5-7B seq2048 used
19.21 GB and 2.02 s for the windowed one-step memory test. The full-clean-anchor
warmup path used 29.91 GB and 16.56 s, still fitting where full AR was killed.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --extra llm
```

For Hugging Face gated/private models:

```bash
uv run --extra llm huggingface-cli login
```

The MLX path requires Apple Silicon. The original PyTorch/Lightning image
classification code is still available behind the `torch` extra:

```bash
uv sync --extra torch
```

## Quick Start

Tiny local smoke test:

```bash
uv run --extra llm mixed-dblocks-train \
  --backend tiny \
  --iters 20 \
  --batch_size 4 \
  --seq_len 64
```

Run Mixed DiffusionBlocks on an `mlx-lm` model:

```bash
uv run --extra llm mixed-dblocks-train \
  --backend mlx-lm \
  --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
  --data train.txt \
  --iters 100 \
  --batch_size 1 \
  --seq_len 1024 \
  --num_blocks 8
```

By default, the LLM CLIs use `--clean_lm_anchor_profile auto_window`,
`--clean_lm_weight 100`, `--denoise_weight 0`, and
`--clean_lm_full_warmup_steps 10`. The auto-window profile runs deterministic
128-token clean CE anchors after warmup, with a 512-token anchor every 8
updates. Use
`--clean_lm_anchor_profile manual --clean_lm_weight 0 --denoise_weight 1` for a
pure DiffusionBlocks ablation.

Installed commands:

```bash
uv run --extra llm mixed-dblocks-train --help
uv run --extra llm mixed-dblocks-quality --help
uv run --extra llm mixed-dblocks-memory --help
uv run --extra llm mixed-dblocks-matrix --help
uv run --extra llm mixed-dblocks-results --help
```

## Reproduce The WikiText-2 Runs

Prepare local WikiText-2 text files:

```bash
uv run --extra llm scripts/prepare_wikitext2.py
```

Full bf16 AR baseline:

```bash
uv run --extra llm mixed-dblocks-quality \
  --mode full \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --steps 100 \
  --eval_batches 4 \
  --batch_size 1 \
  --seq_len 1024 \
  --lr 1e-5 \
  --max_tokens 400000
```

Pure DiffusionBlocks:

```bash
uv run --extra llm mixed-dblocks-quality \
  --mode dblock \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --steps 100 \
  --eval_batches 4 \
  --batch_size 1 \
  --seq_len 1024 \
  --num_blocks 8 \
  --clean_lm_anchor_profile manual \
  --clean_lm_weight 0 \
  --denoise_weight 1 \
  --max_tokens 400000
```

Default Windowed Mixed DiffusionBlocks:

```bash
uv run --extra llm mixed-dblocks-quality \
  --mode dblock \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --steps 100 \
  --eval_batches 4 \
  --batch_size 1 \
  --seq_len 1024 \
  --num_blocks 8 \
  --max_tokens 400000
```

Full-sequence Mixed DiffusionBlocks ablation:

```bash
uv run --extra llm mixed-dblocks-quality \
  --mode dblock \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --steps 100 \
  --eval_batches 4 \
  --batch_size 1 \
  --seq_len 1024 \
  --num_blocks 8 \
  --clean_lm_anchor_profile manual \
  --clean_lm_weight 100 \
  --denoise_weight 1 \
  --max_tokens 400000
```

## Memory Boundary Test

```bash
uv run --extra llm mixed-dblocks-memory \
  --mode dblock \
  --model mlx-community/Qwen2.5-7B-Instruct-bf16 \
  --batch_size 1 \
  --seq_len 2048 \
  --num_blocks 8 \
  --objective paper_ar
```

Quantized 4-bit Qwen weights are not suitable for this full-block trainer
because MLX does not expose gradients for quantized matmul weights. Use
bf16/unquantized MLX weights for block training.

## Publish Matrix And Results

Generate the current quality ablation matrix without running it:

```bash
uv run --extra llm mixed-dblocks-matrix \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
  --seed 1 \
  --seed 2 \
  --seed 3 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --output_dir docs/results/publish
```

Run a bounded prefix:

```bash
uv run --extra llm mixed-dblocks-matrix \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --seed 1 \
  --data corpora/wikitext2/train.txt \
  --val_data corpora/wikitext2/validation.txt \
  --output_dir docs/results/publish \
  --max_runs 4 \
  --run
```

Summarize JSON results as a markdown table:

```bash
uv run --extra llm mixed-dblocks-results docs/results/*.json
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `llm_dblocks/` | Mixed DiffusionBlocks trainer, model adapters, tiny LM |
| `llm_dblock_train.py` | Main LLM training CLI |
| `qwen_quality_benchmark.py` | Full AR vs DiffusionBlocks quality benchmark |
| `qwen_memory_benchmark.py` | One-step peak-memory benchmark |
| `llm_dblocks/experiments.py` | Publish-matrix command generation |
| `llm_dblocks/results.py` | JSON result summarization |
| `mlx_*` | Apple MLX ViT/DiffusionBlocks image-classification port |
| `main.py`, `model.py`, `vit.py`, `dblock_modules.py` | Original upstream PyTorch implementation |
| `docs/` | Benchmark notes, method alignment, launch notes, saved JSON results |
| `scripts/prepare_wikitext2.py` | Reproducible WikiText-2 data preparation |

## Status

This is a research prototype. The current evidence supports:

- Mixed DiffusionBlocks fixes pure DiffusionBlocks' ordinary-LM CE degradation.
- The fix works on real WikiText-2 held-out validation.
- Windowed Mixed makes the quality-preserving anchor deterministic and cheap.
- The memory advantage is large enough to train models/contexts full bf16 AR
  cannot fit on an M4 Max 36 GB Mac.

Still needed:

- instruction-tuning datasets,
- generation-quality evaluation,
- LoRA/QLoRA baselines,
- longer 3B/7B quality curves,
- closing the remaining 1.5B seq1024 CE gap between auto-window and full AR.

See `docs/production_readiness.md` for the production contract and publish gate.

## Attribution

This repo builds on:

- [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)
- The DiffusionBlocks paper:
  [DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](https://arxiv.org/abs/2506.14202)
- [MLX](https://github.com/ml-explore/mlx) and
  [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)

If you use the original DiffusionBlocks method or upstream ViT code, cite the
SakanaAI paper:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author    = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```
