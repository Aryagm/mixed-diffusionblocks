# Budgeted Mixed DiffusionBlocks

Budgeted Mixed DiffusionBlocks is a follow-up experiment to Mixed
DiffusionBlocks. The current mixed objective fixes pure DiffusionBlocks'
ordinary next-token CE degradation by adding exact clean LM CE every update:

```text
loss = denoise_loss + clean_lm_weight * exact_clean_lm_ce
```

That exact anchor is reliable, but expensive. Budgeted Mixed DiffusionBlocks
runs the exact clean anchor every `N` block updates instead:

```text
loss_t = denoise_loss_t
       + I[t mod N == 0] * clean_lm_weight * exact_clean_lm_ce_t
```

This preserves a true autoregressive anchor while cutting the average number of
full clean CE backward passes.

## Implementation

The trainer now supports:

- `clean_lm_interval`: exact clean CE cadence; `1` is the original mixed method.
- `local_lm_weight`: an experimental local active-block LM-head anchor.

The local anchor was tested first because it is very cheap, but the current
result is negative: it remains fast but does not preserve normal next-token CE.

## Qwen2.5-0.5B WikiText-2 Seq128

Setup:

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
- Dataset: WikiText-2 train/validation
- Steps: 100
- Batch size: 1
- Eval batches: 16
- Sequence length: 128
- `clean_lm_weight=100`

| Method | Exact clean CE interval | Validation CE before | Validation CE after | Time |
| --- | ---: | ---: | ---: | ---: |
| Pure DiffusionBlocks | n/a | 2.4844 | 3.0469 | 5.20 s |
| Mixed DiffusionBlocks | 1 | 2.4844 | 2.3750 | 10.25 s |
| Budgeted Mixed DiffusionBlocks | 2 | 2.4844 | 2.3281 | 7.83 s |
| Budgeted Mixed DiffusionBlocks | 4 | 2.4844 | 2.4062 | 6.55 s |
| Budgeted Mixed DiffusionBlocks | 8 | 2.4844 | 2.5469 | 5.91 s |
| Local LM anchor only | n/a | 2.4844 | 3.3594 | 6.48 s |
| Local LM anchor only | n/a | 2.4844 | 3.4844 | 6.68 s |

Takeaway: interval 2 is the best setting in this smoke test. It improves normal
CE more than exact mixed while cutting training time by about 24%.

## Qwen2.5-0.5B WikiText-2 Seq1024

Setup:

- Steps: 100
- Batch size: 1
- Eval batches: 4
- Sequence length: 1024
- `clean_lm_weight=100`

| Method | Exact clean CE interval | Validation CE before | Validation CE after | Time |
| --- | ---: | ---: | ---: | ---: |
| Full bf16 AR | n/a | 2.1094 | 1.9922 | 49.62 s |
| Pure DiffusionBlocks | n/a | 2.1094 | 4.6875 | 41.91 s |
| Mixed DiffusionBlocks | 1 | 2.1094 | 1.9453 | 77.49 s |
| Budgeted Mixed DiffusionBlocks | 2 | 2.1094 | 1.9844 | 56.79 s |
| Budgeted Mixed DiffusionBlocks | 4 | 2.1094 | 2.0625 | 51.12 s |

Takeaway: interval 2 keeps the quality-preserving behavior and is much closer to
full AR wall-clock than exact mixed. It is 26.7% faster than exact mixed while
still beating full AR CE on this run.

## Current Verdict

Budgeted exact anchoring is a better research direction than the naive local LM
anchor. The local anchor is too weak and reproduces the pure-objective failure
mode. Exact clean CE every two updates appears to be the useful frontier:

- reliable enough to preserve ordinary next-token CE,
- cheaper than exact mixed every update,
- simple to smoke test,
- compatible with the existing 7B memory-feasibility story because the peak
  exact-anchor step is unchanged while average compute drops.

The next experiment should test interval 2 on Qwen2.5-1.5B seq1024 and one 7B
short quality curve.
