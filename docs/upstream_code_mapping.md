# Upstream Code Mapping

The public `SakanaAI/DiffusionBlocks` repository currently has a single branch,
`main`, and contains the ViT image-classification implementation only. There is
no public CUDA/Llama/AR training source in that repository to translate line by
line.

Verified upstream files:

- `model.py`
- `vit.py`
- `dblock_modules.py`
- `main.py`
- `data.py`

## Ported Directly From The Public Repo

- log-normal block sigma partitioning from `get_block_sigmas`
- discrete sampling sigmas from `get_discrete_sigmas`
- one-block-at-a-time target selection
- gamma-expanded sigma interval per block
- normalized class/token embeddings
- EDM preconditioning:
  - `c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)`
  - `c_out = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)`
  - `c_in = 1 / sqrt(sigma^2 + sigma_data^2)`
- EDM loss weight:
  - `(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`
- Euler denoising loop pattern for diffusion reconstruction

## Not Present Upstream

- Qwen/LLM adapters
- MLX implementation
- AR clean/noisy concatenation mask code
- AR generation/evaluation code

Those pieces are implemented from the paper description and adapted to `mlx-lm`.

## Current Gap

The public repo's ViT path produces ordinary classification logits through its
DiffusionBlocks sampler. For AR LLMs, the paper evaluates generated text from a
diffusion-style sampler rather than ordinary next-token CE from a vanilla model
forward. This repo now includes a teacher-forced diffusion reconstruction CE, but
still needs a full free-running generation evaluator before claiming final AR
quality parity.
