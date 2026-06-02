# Adaptive Framed DiffusionBlocks Experiments

This pass tested two extensions on top of Budgeted Mixed DiffusionBlocks:

1. **Windowed exact anchors:** run exact clean CE on a random clean-token window
   instead of the full sequence.
2. **Adaptive frames:** calibrate layer-wise activation drift, then form
   contiguous drift-balanced frames instead of equal layer-count blocks.

The goal was to reduce the clean-anchor compute and improve the core
DiffusionBlocks framing without losing the ordinary next-token CE preservation
from exact anchoring.

## Implementation

New trainer knobs:

- `clean_lm_seq_len`: if positive, exact clean CE uses a random subsequence of
  this length.
- `clean_lm_full_warmup_steps`: use full-sequence clean CE every step during an
  initial warmup, then fall back to the configured windowed anchor.
- `block_layer_boundaries`: optional explicit contiguous layer boundaries.

New benchmark knob:

- `frame_strategy=drift`: run a short activation-drift calibration pass and
  convert per-layer drift scores into balanced contiguous frames.

## Qwen2.5-0.5B WikiText-2 Seq1024

Setup unless noted:

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
- Steps: 100
- Batch size: 1
- Eval batches: 4
- Sequence length: 1024
- `clean_lm_weight=100`
- `clean_lm_interval=2`

| Method | Frames | Anchor window | Warmup | Validation CE before | Validation CE after | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full bf16 AR | n/a | n/a | n/a | 2.1094 | 1.9922 | 49.62 s |
| Pure DiffusionBlocks | 8 equal | n/a | n/a | 2.1094 | 4.6875 | 41.91 s |
| Budgeted Mixed | 8 equal | full | 0 | 2.1094 | 1.9844 | 56.79 s |
| Windowed Budgeted Mixed | 8 equal | 256 | 0 | 2.1094 | 2.0781 | 43.73 s |
| Windowed Budgeted Mixed | 8 equal | 512 | 0 | 2.1094 | 2.0000 | 47.12 s |
| Windowed Budgeted Mixed | 8 equal | 768 | 0 | 2.1094 | 2.0000 | 51.32 s |
| Windowed Budgeted Mixed | 8 equal | 512 | 10 | 2.1094 | 2.0000 | 51.85 s |
| Windowed Budgeted Mixed | 12 equal | 512 | 0 | 2.1094 | 2.0312 | 44.68 s |
| Drift-Framed Budgeted Mixed | drift | full | 0 | 2.1094 | 1.9844 | 58.72 s |
| Windowed Mixed | 8 equal | 512 | 0 | 2.1094 | 1.9531 | 58.43 s |
| Windowed Mixed | 8 equal | 256 | 0 | 2.1094 | 1.9766 | 46.92 s |
| Windowed Mixed | 8 equal | 128 | 0 | 2.1094 | 1.9688 | 45.25 s |
| Windowed Mixed | 8 equal | 64 | 0 | 2.1094 | 2.0000 | 42.81 s |

The drift calibration produced:

```text
[(0, 5), (5, 8), (8, 12), (12, 15), (15, 18), (18, 21), (21, 23), (23, 24)]
```

## Verdict

The first best tradeoff was **windowed budgeted exact anchoring with a 512-token
anchor window**:

- 17.0% faster than full-sequence budgeted anchoring,
- 39.2% faster than exact mixed every step,
- close to full AR wall-clock,
- normal CE after 100 steps is 2.0000, slightly worse than full AR's 1.9922 and
  full-sequence budgeted anchoring's 1.9844.

The better follow-up is **Windowed Mixed DiffusionBlocks**: run exact clean CE
every step, but only on a small random clean-token window. On this smoke test,
128-token windows are the best setting:

- Validation CE improves from 2.1094 to 1.9688.
- It beats the full AR 100-step CE of 1.9922.
- It runs in 45.25 s, faster than the 49.62 s full AR baseline.
- It keeps the clean anchor exact, but makes the exact anchor local in sequence
  length.

The adaptive frame experiments did not improve this smoke benchmark. Drift
frames preserved CE but were slower, and 12 equal frames were faster but lost
too much CE. This suggests the next adaptive-framing version should change more
than boundaries: it should also assign per-frame sigma caps, anchor intervals,
and sampling probabilities.

## Qwen2.5-1.5B WikiText-2 Seq1024

| Method | Anchor window | Steps | Validation CE before | Validation CE after | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full bf16 AR | n/a | 100 | 1.7344 | 1.5781 | 140.11 s |
| Pure DiffusionBlocks | n/a | 100 | 1.7344 | 1.9609 | 103.56 s |
| Mixed DiffusionBlocks | full | 100 | 1.7344 | 1.5859 | 197.90 s |
| Windowed Mixed | 128 | 100 | 1.7344 | 1.6094 | 109.89 s |
| Windowed Mixed | 256 | 100 | 1.7344 | 1.6250 | 121.99 s |
| Windowed Mixed | 128 | 150 | 1.7344 | 1.6094 | 167.86 s |

On 1.5B, 128-token Windowed Mixed is much faster than full AR and exact mixed,
but it does not yet match their final CE. Running longer to 150 steps did not
improve this setting, so the next improvement needs better frame/risk selection
or a different anchor schedule rather than more steps alone.

## Qwen2.5-7B Seq2048 Memory

| Method | Clean anchor | Peak memory | Time |
| --- | --- | ---: | ---: |
| Pure DiffusionBlocks | none | 22.35 GB | 4.06 s |
| Mixed DiffusionBlocks | full seq2048 | 29.93 GB | 40.21 s |
| Windowed Mixed | random seq128 | 22.31 GB | 6.61 s |

The 7B memory result is the strongest practical finding from this pass:
windowed exact anchoring keeps the quality-preserving AR anchor but removes the
large peak-memory penalty of full-sequence clean CE. Peak memory is effectively
the same as pure DiffusionBlocks while still applying an exact clean CE anchor.

## Next Candidate

The next method worth testing is **Risk-Adaptive Windowed DiffusionBlocks**:

```text
calibrate each frame risk = activation drift + clean CE sensitivity
high-risk frames: lower sigma_max, anchor interval 1-2, full/window 512 anchor
low-risk frames: higher sigma_max, anchor interval 4-8, window 256/no anchor
sample frames in proportion to risk and recent clean-CE drift
```

The quick result here is that windowed exact anchors are useful and highly
memory-efficient, but adaptive boundaries alone are not enough.
