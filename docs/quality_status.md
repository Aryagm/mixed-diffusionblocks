# Quality Status

The current quality evidence is a sanity check, not a final model-quality claim.

## Tiny Local Model

On the built-in byte-level tiny transformer, the paper-style AR DiffusionBlocks
objective learns:

| Method | Eval objective before | Eval objective after | Steps |
| --- | ---: | ---: | ---: |
| Full next-token CE | 5.6479 | 2.5964 | 30 |
| DiffusionBlocks `paper_ar` CE | 5.6056 | 3.5696 | 30 |

## Qwen2.5-0.5B-Instruct-bf16

On the built-in sample text, sequence length 128, batch size 1. These results
use the repo-matched DiffusionBlocks defaults: `sigma_max=80`, EDM
preconditioning, and sigma-weighted CE.

| Method | Objective | Before | After | Steps |
| --- | --- | ---: | ---: | ---: |
| Full bf16 | standard next-token CE | 2.3613 | 0.0280 | 100 |
| DiffusionBlocks `paper_ar` | denoising CE | 87.7998 | 25.5391 | 100 |
| DiffusionBlocks `paper_ar` | denoising CE | 81.8438 | 5.1035 | 1000 |
| DiffusionBlocks `hidden` | hidden denoising + aux CE | 247.6444 | 20.3816 | 1000 |

Teacher-forced diffusion reconstruction CE also improves:

| Method | Diffusion reconstruction CE before | After | Steps |
| --- | ---: | ---: | ---: |
| DiffusionBlocks `paper_ar` | 135.8750 | 98.9375 | 100 |
| DiffusionBlocks `paper_ar` | 116.7500 | 30.4062 | 1000 |

Pure `paper_ar` learns the denoising task but still damages vanilla full-model
next-token CE after applying the trained blocks to the ordinary forward pass:

| Method | Standard next-token CE before | Standard next-token CE after | Steps |
| --- | ---: | ---: | ---: |
| Full bf16 | 2.3613 | 0.0280 | 100 |
| DiffusionBlocks `paper_ar` | 2.3672 | 3.9473 | 100 |
| DiffusionBlocks `paper_ar` | 2.3984 | 7.0000 | 1000 |
| DiffusionBlocks `hidden` | 2.4062 | 14.8750 | 1000 |

This means the paper objective alone is not a standard next-token fine-tuning
replacement. Running longer is not enough by itself: with pure `paper_ar`, 10x
more steps improved denoising CE but made standard CE worse.

## Mixed DiffusionBlocks + Clean LM Objective

To make the trained blocks useful in the ordinary forward pass, the trainer now
supports `--clean_lm_weight`. This adds a normal next-token CE term through the
full model while only the selected DiffusionBlock is trainable.

Same Qwen2.5-0.5B setup, 100 training steps unless noted. For mixed rows, the
eval objective includes both the denoising term and weighted clean CE.

| Method | `clean_lm_weight` | Eval objective after | Diffusion reconstruction CE after | Standard CE after | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full bf16 | n/a | n/a | n/a | 0.0280 | 15.11 |
| DiffusionBlocks | 0 | 25.5391 | 98.9375 | 3.9473 | 5.28 |
| DiffusionBlocks | 25 | 26.2979 | 99.6875 | 0.0896 | 10.48 |
| DiffusionBlocks | 50 | 28.4170 | 103.0625 | 0.0725 | 10.90 |
| DiffusionBlocks | 100 | 33.4746 | 106.0625 | 0.0645 | 11.21 |
| DiffusionBlocks | 100 | 23.8828 | 99.7500 | 0.0587 | 22.18, 200 steps |
| DiffusionBlocks | 100 | 4.5654 | 21.9844 | 0.0124 | 114.31, 1000 steps |
| DiffusionBlocks | 500 | 64.4805 | 116.4375 | 0.2972 | 10.84 |

For the main `clean_lm_weight=100`, 100-step run, the split eval metrics are
`denoise_loss=28.4312` and `clean_lm_loss=0.0502`.

The useful band is `clean_lm_weight=25` to `100`. It turns DiffusionBlocks from
a denoising-only objective into a blockwise fine-tuning path that improves
ordinary next-token CE. At 1000 blockwise steps, the mixed objective beats the
100-step full bf16 CE target on this small sample, but it takes about 7.6x more
wall-clock than the 100-step full run.

A separate target-curve run evaluated normal next-token CE every 100 mixed
DiffusionBlocks steps. It crossed the 100-step full bf16 CE target between 300
and 400 steps:

| Mixed DiffusionBlocks steps | Standard CE | Cumulative train time |
| ---: | ---: | ---: |
| 300 | 0.0359 | 34.67 s |
| 400 | 0.0167 | 46.21 s |

With 100-step eval granularity, the measured time-to-target is 400 steps and
46.21 s, about 3.1x the 15.11 s full bf16 training time. Linear interpolation
between the 300-step and 400-step checkpoints puts the crossing near 340 steps
and about 39-40 s, about 2.6x full bf16 time.

Memory on the same Qwen2.5-0.5B setup:

| Method | Peak memory | Relative to full |
| --- | ---: | ---: |
| Full bf16 | 4.39 GB | 1.00x |
| DiffusionBlocks `paper_ar` + `clean_lm_weight=100` | 1.57 GB | 0.36x |

So this setup uses about 64% less peak memory on Qwen0.5B while improving
ordinary CE with the mixed objective.

## Real Dataset: WikiText-2

To move beyond the built-in repeated sample text, we prepared real train and
validation files from `Salesforce/wikitext`, config `wikitext-2-raw-v1`.

Setup:

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
- Train data: `corpora/wikitext2/train.txt`
- Validation data: `corpora/wikitext2/validation.txt`
- Sequence length 128, batch size 1, 16 deterministic validation batches
- 100 training steps, `max_tokens=200000`

Results:

| Method | Validation CE before | Validation CE after | Time | Notes |
| --- | ---: | ---: | ---: | --- |
| Full bf16 AR | 2.4844 | 2.5000 | 16.45 s | No improvement at this LR/step budget |
| Pure DiffusionBlocks | 2.4844 | 3.0469 | 5.46 s | Denoising improves, ordinary CE worsens |
| Mixed DiffusionBlocks | 2.4844 | 2.3750 | 11.35 s | Ordinary CE improves |

This confirms the key failure/fix on a real held-out dataset: pure
DiffusionBlocks is not enough for normal LM behavior, while the mixed objective
improves standard next-token CE.

### Longer Context: WikiText-2 Seq1024

The seq1024 runs were executed sequentially, not in parallel, to avoid timing and
memory contention.

Qwen2.5-0.5B:

| Method | Validation CE before | Validation CE after | Time | Notes |
| --- | ---: | ---: | ---: | --- |
| Full bf16 AR | 2.1094 | 1.9922 | 49.62 s | Improves CE |
| Pure DiffusionBlocks | 2.1094 | 4.6875 | 41.91 s | Denoising improves, ordinary CE worsens |
| Mixed DiffusionBlocks | 2.1094 | 1.9453 | 77.49 s | Beats full AR CE, slower |

Qwen2.5-1.5B:

| Method | Validation CE before | Validation CE after | Time | Notes |
| --- | ---: | ---: | ---: | --- |
| Full bf16 AR | 1.7344 | 1.5781 | 140.11 s | Best CE by a small margin |
| Pure DiffusionBlocks | 1.7344 | 1.9609 | 103.56 s | Denoising improves, ordinary CE worsens |
| Mixed DiffusionBlocks | 1.7344 | 1.5859 | 197.90 s | Essentially matches full AR CE, slower |

At longer context, mixed DiffusionBlocks still fixes the pure objective's
ordinary-CE failure mode. It is no longer faster than full AR at 100 steps, but
it remains the quality-preserving blockwise path.

## Still Needed

Before claiming quality parity or practical fine-tuning quality, run:

- a real instruction or domain dataset, not the built-in sample text,
- held-out validation over the same objective,
- generation samples before/after,
- teacher-model perplexity or MAUVE-style generation metrics, matching the
  DiffusionBlocks paper's AR evaluation style,
- larger held-out datasets for the mixed objective,
- a denoising free-generation path for `paper_ar`, not just teacher-forced
  reconstruction,
- LoRA and full bf16 baselines where they fit.

## Qwen2.5-1.5B Scale-Up

Same built-in sample text, sequence length 128, batch size 1, 8 blocks.

Full bf16 100-step target:

| Method | Steps | Standard CE before | Standard CE after | Train time |
| --- | ---: | ---: | ---: | ---: |
| Full bf16 | 100 | 2.3457 | 0.0306 | 48.35 s |

Mixed DiffusionBlocks target curve, `clean_lm_weight=100`:

| Mixed DiffusionBlocks steps | Standard CE | Cumulative train time | Hit target |
| ---: | ---: | ---: | --- |
| 100 | 0.0417 | 29.98 s | no |
| 200 | 0.0588 | 59.62 s | no |
| 300 | 0.0462 | 90.32 s | no |
| 400 | 0.0402 | 120.44 s | no |
| 500 | 0.0404 | 151.96 s | no |
| 600 | 0.0250 | 183.71 s | yes |

This confirms the 0.5B result scales to a larger model: mixed DiffusionBlocks
matches and beats the 100-step full bf16 CE target. The cost is about 6x more
steps and 3.8x more wall-clock time on this setup.

Memory at the same Qwen2.5-1.5B seq128 setup:

| Method | Peak memory | Relative to full |
| --- | ---: | ---: |
| Full bf16 | 12.75 GB | 1.00x |
| Pure DiffusionBlocks | 4.11 GB | 0.32x |
| Mixed DiffusionBlocks | 4.05 GB | 0.32x |

So the mixed objective uses about 68% less memory while reaching the AR target
with longer training.

## Qwen2.5-3B Memory Check

The Qwen2.5-3B quality run was not completed in this pass because the 100-step
full bf16 baseline exceeded the quick experiment budget. The memory pattern does
hold at 3B:

| Method | Peak memory | Relative to full |
| --- | ---: | ---: |
| Full bf16 | 24.59 GB | 1.00x |
| Pure DiffusionBlocks | 8.47 GB | 0.34x |
| Mixed DiffusionBlocks | 8.47 GB | 0.34x |

The next scale-up quality target should be an overnight-style 3B run with
checkpointed CE every 100 mixed steps.
