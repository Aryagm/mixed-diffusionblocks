# Method Alignment With SakanaAI DiffusionBlocks

This repo is an MLX port/adaptation of SakanaAI's DiffusionBlocks idea, not a
new training method.

## Paper Components Implemented

For autoregressive language models, the paper states that:

- noise is added after the embedding layer,
- the denoiser recovers clean token embeddings from noisy token embeddings
  conditioned on previous clean token embeddings,
- cross-entropy is used instead of L2 for AR text,
- noisy and clean sequences are concatenated with a modified causal mask so
  noisy tokens can attend to their clean past without clean future leakage.

The `paper_ar` objective implements those mechanics:

- `clean = normalize(embed(input_ids))`
- `noisy = clean + sigma * epsilon`
- `hidden = concat(clean, noisy)`
- custom causal-consistency mask from `ar_concat_mask`
- active block range only is trained
- CE is computed on noisy-token positions

The implementation also ports the public repository's ViT denoising math:

- upstream sigma range defaults, `sigma_min=0.002`, `sigma_max=80.0`
- EDM preconditioning with `c_in`, `c_skip`, and `c_out`
- sigma-weighted CE using the same EDM weight formula
- Euler-style diffusion reconstruction for teacher-forced eval

The optional mixed objective adds:

- standard next-token CE through the ordinary full-model forward pass,
- gradients only for the selected block, preserving the blockwise update rule,
- the `--clean_lm_weight` knob for trading denoising quality against ordinary
  next-token CE.

## Deliberate Differences

- The implementation targets pretrained `mlx-lm` causal LMs instead of training
  the 12-layer Llama-style models from scratch used in the paper experiments.
- Large-model benchmarks are one-step memory boundary tests, not convergence
  results.
- The mixed objective is an engineering extension for practical LM fine-tuning;
  pure `paper_ar` is the closer paper-aligned objective.
- The older `hidden` objective is kept only as an ablation path.
- The public GitHub repo does not include AR/Llama source code, so AR masking and
  evaluation are implemented from the paper text rather than translated from
  upstream code.

## Claims We Can Make

- We port the DiffusionBlocks training transformation to MLX.
- We provide a paper-style AR objective for causal LMs.
- We demonstrate bf16 Qwen block training on Apple Silicon where ordinary full
  bf16 fine-tuning is killed.

## Claims We Should Not Make Yet

- We should not claim final model quality on Qwen until running real fine-tuning
  and generation evaluations.
- We should not claim this beats LoRA/QLoRA by quality or parameter count.
- We should not claim quantized full-block training; MLX quantized matmul
  weights do not expose gradients.
