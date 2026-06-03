# Production Readiness

This repo is now organized as a usable training library plus reproducible
benchmark harness for Windowed Mixed DiffusionBlocks.

## Supported Use

Use this for full-block bf16 fine-tuning on Apple Silicon when full
autoregressive backpropagation over the whole model is too memory-heavy. The
default path updates one transformer block at a time and adds an exact
next-token CE anchor through the normal model forward pass. The default
fine-tuning objective sets `denoise_weight=0`; the diffusion objective remains
available as an explicit ablation or regularizer.

Supported production path:

- Apple Silicon with MLX and `mlx-lm`
- bf16 or otherwise trainable unquantized weights
- causal language models supported by `mlx-lm` adapters
- full-block updates through `DBlockTrainer`
- default anchor profile: `auto_window`

Not currently supported:

- quantized 4-bit full-block weight updates,
- non-MLX production training,
- a claim that this universally beats LoRA/QLoRA before running the publish
  matrix on instruction tuning data.

## Installed Commands

After `uv sync --extra llm`, the production-facing commands are:

```bash
uv run --extra llm mixed-dblocks-train --help
uv run --extra llm mixed-dblocks-quality --help
uv run --extra llm mixed-dblocks-memory --help
uv run --extra llm mixed-dblocks-matrix --help
uv run --extra llm mixed-dblocks-results --help
```

The default training command uses Windowed Mixed DiffusionBlocks:

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

Equivalent explicit default:

```bash
--clean_lm_anchor_profile auto_window --clean_lm_weight 100 --denoise_weight 0 --clean_lm_full_warmup_steps 10
```

Pure DiffusionBlocks remains available as an ablation:

```bash
--clean_lm_anchor_profile manual --clean_lm_weight 0 --denoise_weight 1
```

Full-sequence mixed anchoring remains available as a quality ablation:

```bash
--clean_lm_anchor_profile manual --clean_lm_weight 100 --clean_lm_seq_len 0 --denoise_weight 1
```

## Publish Matrix

Generate the current publish matrix without running it:

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

Run a bounded prefix of the matrix:

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

Add 7B memory boundary jobs only when you intentionally want heavy runs:

```bash
uv run --extra llm mixed-dblocks-matrix \
  --model mlx-community/Qwen2.5-7B-Instruct-bf16 \
  --seed 1 \
  --include_memory \
  --output_dir docs/results/publish
```

Summarize result files:

```bash
uv run --extra llm mixed-dblocks-results docs/results/*.json
```

## Readiness Gate

Treat the method as production-usable for memory-limited full-block MLX
fine-tuning when:

- the target model uses trainable bf16/unquantized weights,
- full AR does not fit or is impractically memory-heavy,
- the workload benefits from updating actual transformer blocks rather than
  low-rank adapters,
- the generated result table shows normal next-token CE improves rather than
  regresses.

Treat broad "default for everyone" claims as blocked until:

- the publish matrix has 3 seeds on 0.5B and 1.5B quality runs,
- the 7B memory boundary is reproduced with the current CLI,
- at least one instruction-tuning dataset is included,
- LoRA/QLoRA baselines are run with comparable token budgets,
- generation-quality evaluation is added beyond WikiText-2 CE.

The honest current claim is: Windowed Mixed DiffusionBlocks is a practical,
installable default for full-block bf16 fine-tuning under an Apple Silicon
memory cap, with reproducible commands and result summarization. On the current
Qwen2.5-1.5B seq1024 smoke test it matches full-anchor mixed CE while running
far faster than full AR, but it still does not beat full AR's best CE at the
same 100-step horizon.
