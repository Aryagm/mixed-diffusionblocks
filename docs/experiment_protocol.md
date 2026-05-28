# Experiment Protocol

This is the current gate for deciding whether the LLM DiffusionBlocks path is
useful enough to claim as fine-tuning, not just memory-bound training.

## Questions

1. Does pure `paper_ar` learn its own denoising objective?
2. Does the trained model improve ordinary next-token CE in the normal forward
   pass?
3. Does the quality-aware objective still preserve the Apple Silicon memory
   advantage on a model that full bf16 fine-tuning cannot fit?

## Fixed Setup

- Hardware: M4 Max MacBook Pro, 36 GB unified memory.
- Runtime: MLX through `mlx-lm`.
- Small quality proxy: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, built-in
  sample text, sequence length 128, batch size 1, 8 blocks, 8 eval batches.
- Large memory gate: `mlx-community/Qwen2.5-7B-Instruct-bf16`, batch size 1,
  8 blocks.
- Diffusion defaults: `sigma_min=0.002`, `sigma_max=80`, EDM preconditioning,
  sigma-weighted CE.

## Commands

```bash
uv run --extra llm qwen_quality_benchmark.py \
  --mode full \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --steps 100 \
  --seq_len 128 \
  --batch_size 1 \
  --output_json docs/results/qwen05_full_100.json

uv run --extra llm qwen_quality_benchmark.py \
  --mode dblock \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --steps 100 \
  --seq_len 128 \
  --batch_size 1 \
  --num_blocks 8 \
  --clean_lm_weight 100 \
  --output_json docs/results/qwen05_dblock_cw100_100.json

uv run --extra llm qwen_memory_benchmark.py \
  --mode dblock \
  --model mlx-community/Qwen2.5-7B-Instruct-bf16 \
  --seq_len 2048 \
  --batch_size 1 \
  --num_blocks 8 \
  --clean_lm_weight 100 \
  --output_json docs/results/qwen25_7b_memory_cw100_seq2048.json
```

## Decision Rule

- Pure `paper_ar` is a memory and denoising proof, not a standard LM
  fine-tuning claim.
- Mixed `paper_ar` + clean LM CE is the current practical path if ordinary
  next-token behavior matters.
- The current strong claim is quality-aware 7B bf16 block training at 2048
  tokens on M4 Max 36 GB, plus a small Qwen0.5B validation that ordinary CE can
  be improved rather than damaged.

