# Windowed Mixed DiffusionBlocks Default

Windowed Mixed DiffusionBlocks is the current default candidate for full-block
LLM fine-tuning in this repo. It keeps the DiffusionBlocks memory profile by
training one transformer block at a time, but adds an exact clean next-token CE
anchor through the ordinary model forward pass.

The default profile is:

```text
clean_lm_anchor_profile = auto_window
clean_lm_weight = 100
clean_lm_interval = 1
clean_lm_seq_len = 128
clean_lm_window_count = 1
clean_lm_large_seq_len = 512
clean_lm_large_interval = 8
```

During training, anchor placement is deterministic from the update step and
block index. This removes the random-window variance from the original windowed
prototype while keeping the clean anchor local in sequence length.

## Why This Is The Default

The full-sequence clean anchor is reliable but expensive. Windowed Mixed keeps
an exact AR objective, but evaluates it on framed clean-token windows:

```text
loss = diffusionblocks_denoising_loss
     + aux_lm_weight * noisy_token_ce
     + clean_lm_weight * exact_clean_next_token_ce(window)
```

This is not LoRA and not adapter training. It updates the selected bf16
transformer block directly, so it targets cases where full-block adaptation is
needed but full bf16 backpropagation over the whole model does not fit.

## Current Smoke Evidence

Qwen2.5-1.5B, WikiText-2, seq1024, batch 1, 100 steps:

| Method | Validation CE before | Validation CE after | Time |
| --- | ---: | ---: | ---: |
| Full bf16 AR | 1.7344 | 1.5781 | 140.11 s |
| Pure DiffusionBlocks | 1.7344 | 1.9609 | 103.56 s |
| Full-anchor Mixed | 1.7344 | 1.5859 | 197.90 s |
| 128-token Windowed Mixed | 1.7344 | 1.6094 | 109.89 s |
| Auto-window Mixed | 1.7344 | 1.6016 | 115.97 s |

Qwen2.5-7B, seq2048, one-step memory boundary:

| Method | Peak memory | Time |
| --- | ---: | ---: |
| Pure DiffusionBlocks | 22.35 GB | 4.06 s |
| Full-anchor Mixed | 29.93 GB | 40.21 s |
| Windowed Mixed, seq128 anchor | 22.31 GB | 6.61 s |

## Interpretation

The useful claim is narrow and practical:

- It makes quality-aware DiffusionBlocks close to pure DiffusionBlocks memory.
- It removes the random anchor-window policy from the earlier prototype.
- It reduces the public knobs to one default profile.
- It is faster than full AR on the 1.5B seq1024 smoke run, but still has a CE
  gap to full AR at 100 steps.
- It should be evaluated against LoRA/QLoRA on real instruction tuning before
  claiming broad superiority.

The next research target is closing the remaining 1.5B CE gap without giving up
the 7B memory result.
