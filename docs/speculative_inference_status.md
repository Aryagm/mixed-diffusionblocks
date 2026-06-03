# Speculative Inference Status

This branch tests whether the DiffusionBlocks/blockwise idea can speed up
inference by using early transformer blocks as an in-model draft path.

## Implemented

- `EarlyExitDraftModel`: wraps a loaded MLX LM as a cache-compatible draft model
  using only layers `0:exit_layer`.
- `prefix_reuse_speculative_generate_step`: greedy self-speculative decoding that
  reuses prefix-layer KV caches and hidden states, then verifies only suffix
  layers.
- `cached_greedy_generate_step`: lean cached greedy decoding without per-token
  logprob materialization.
- `speculative_inference_benchmark.py`: compares MLX cached greedy, lean cached
  greedy, MLX built-in speculative decoding, and prefix-reuse speculation.

All checked prefix-reuse rows produced the same token sequence as full greedy.

## Results

Single-machine smoke tests on cached MLX Qwen bf16 models, prompt length 128.

| Model | Best exact path observed | Tokens/s | Versus MLX greedy | Versus lean greedy |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | prefix reuse, full depth, draft 8 | 107.90 | 0.54x | 1.14x |
| Qwen2.5-1.5B | prefix reuse, full depth, draft 8 | 80.99 | 1.49x | 1.36x |
| Qwen2.5-7B | lean cached greedy | 19.64 | 1.67x | 1.00x |
| Qwen2.5-7B | prefix reuse, full depth, draft 8 | 19.59 | 1.66x | 1.00x |

Best non-full-depth prefix-reuse rows:

| Model | Exit | Draft | Acceptance | Tokens/s | Versus MLX greedy | Versus lean greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | 26 / 28 | 4 | 90.62% | 67.57 | 1.24x | 1.13x |
| Qwen2.5-7B | 24 / 28 | 1 | 64.58% | 18.77 | 1.59x | 0.96x |

## Current Verdict

This is not ready to claim as a universal default speculative decoder.

The strongest real finding is narrower:

- A lean cached greedy loop is materially faster than `mlx-lm`'s default
  logprob-producing greedy path on 7B for this setup.
- Prefix-reuse self-speculation is much better than naive self-speculation because
  it avoids rerunning draft prefix layers during verification.
- Exact self-speculation still has a hard ceiling: every accepted token still
  needs full-model verification. The method mainly wins through batching and lower
  Python/logprob overhead, not by eliminating full-model FLOPs.

The next path worth testing is training an explicit early-exit/draft head or
blockwise auxiliary objective so a cheaper exit, not the full-depth path, reaches
high acceptance. Without that, the safest default is an auto policy: use lean
cached greedy by default, and enable prefix-reuse speculation only after a short
calibration proves it beats lean greedy for the specific model, prompt length, and
decode settings.
