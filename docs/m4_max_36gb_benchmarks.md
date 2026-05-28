# M4 Max 36 GB Benchmarks

Hardware:

- MacBook Pro, Apple M4 Max
- 36 GB unified memory
- Batch size 1
- bf16 MLX weights
- One optimizer step
- 8 DiffusionBlocks
- `paper_ar` objective: noisy token embeddings, clean/noisy sequence
  concatenation, causal-consistency mask, CE denoising loss
- Mixed objective: `paper_ar` plus `--clean_lm_weight 100`, which adds standard
  next-token CE through the ordinary model forward pass while only the selected
  block is trainable

## Qwen2.5-7B-Instruct-bf16

Full bf16 fine-tuning was killed by the OS even at 512 tokens. DiffusionBlocks
completed useful contexts, including with the mixed objective needed for
ordinary next-token CE.

| Sequence length | Full bf16 fine-tune | DiffusionBlocks objective | Peak memory | Time | Notes |
| ---: | --- | --- | ---: | ---: | --- |
| 512 | killed by OS | mixed, `clean_lm_weight=100` | 19.21 GB | 5.54 s | Quality-fixing objective still fits |
| 1024 | not rerun after 512 kill | pure `paper_ar` | 19.21 GB | 1.91 s | Pure denoising reference |
| 2048 | not rerun after 512 kill | pure `paper_ar` | 22.35 GB | 4.06 s | Pure denoising reference |
| 2048 | not rerun after 512 kill | mixed, `clean_lm_weight=100` | 29.93 GB | 40.21 s | Best current quality-aware headline |
| 4096 | not rerun after 512 kill | pure `paper_ar` | 38.96 GB | 13.71 s | Over physical memory; edge run |

Suggested headline:

> Qwen2.5-7B bf16 block training on an M4 Max 36 GB Mac. Full bf16 training is
> killed even at 512 tokens; mixed DiffusionBlocks + next-token CE reaches
> 2048 tokens at 29.93 GB peak while still training actual bf16 transformer
> blocks.

## Qwen2.5-3B-Instruct-bf16

These measurements used the earlier hidden-state denoising objective and are
kept as an auxiliary memory boundary reference.

| Sequence length | Full bf16 fine-tune | DiffusionBlocks |
| ---: | ---: | ---: |
| 1024 | 24.59 GB peak, 12.74 s | 12.20 GB peak, 0.89 s |
| 4096 | 46.45 GB peak, 133.61 s | 26.27 GB peak, 5.53 s |
| 8192 | killed by OS | 53.04 GB peak, 80.17 s |

The 4096-token full run completed only by exceeding physical memory and becoming
very slow. DiffusionBlocks stayed materially lower and faster.

## Qwen2.5-1.5B-Instruct-bf16

These measurements used the earlier hidden-state denoising objective and are
kept as an auxiliary memory boundary reference.

| Sequence length | Full bf16 fine-tune | DiffusionBlocks |
| ---: | ---: | ---: |
| 128 | 12.75 GB peak, 0.58 s | 5.28 GB peak, 0.27 s |
| 1024 | 12.75 GB peak, 1.35 s | 7.33 GB peak, 0.39 s |
| 4096 | 29.20 GB peak, 10.36 s | 16.59 GB peak, 1.95 s |
| 8192 | killed by OS | 33.51 GB peak, 16.24 s |

## Caveats

- These are one-step memory boundary tests, not convergence results.
- Quantized 4-bit Qwen weights do not work for full-block training because MLX
  does not expose gradients for quantized matmul weights.
- LoRA and QLoRA are different baselines. This benchmark is about bf16/full-block
  training under a memory cap.
