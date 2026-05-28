# DiffusionBlocks (ICLR 2026)

<div align="center">
<img alt="overview" src="./overview.jpg" title="overview">
</div>

> We propose ***DiffusionBlocks***, a principled framework that partitions transformers into independently trainable blocks, reducing memory requirements proportionally while maintaining competitive performance across diverse architectures and tasks.

This is an official implementation of *[DiffusionBlocks](https://arxiv.org/abs/2506.14202)* on image classification using Vision Transformers (ViT).

## Installation

Please install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then, run:

```bash
# Install Apple MLX dependencies
uv sync

# Optional: install the original PyTorch/Lightning stack
uv sync --extra torch

# Make sure to login to Hugging Face before training
uv run huggingface-cli login

# Optional, only needed for the PyTorch/Lightning entry point
uv run wandb login
```

We conducted our experiments in the following environment: Python Version 3.12 and CUDA Version 12.2 H100.

The MLX port targets Apple Silicon Macs. It keeps the original PyTorch implementation in
`main.py`, `model.py`, and `vit.py`, and adds a parallel MLX implementation in
`mlx_main.py`, `mlx_model.py`, `mlx_vit.py`, `mlx_data.py`, and
`mlx_dblock_modules.py`.

## Training

The model checkpoints are saved in `logs` folder.

**Baseline (ViT):**

```bash
uv run --extra torch main.py train cifar100 --model_type vit
```

**DiffusionBlocks:**

```bash
uv run --extra torch main.py train cifar100 --model_type dblock
```

### Training on Apple MLX

**Baseline (ViT):**

```bash
uv run mlx_main.py train cifar100 --model_type vit
```

**DiffusionBlocks:**

```bash
uv run mlx_main.py train cifar100 --model_type dblock
```

Use `--debug` for a short smoke run:

```bash
uv run mlx_main.py train cifar100 --model_type dblock --debug --batch_size 8
```

* **NOTE:** the total epochs in DiffusionBlocks is multiplied by the number of blocks to align the total number of iterations with the baseline as one step in DiffusionBlocks corresponds to training for one block.

<details>

In the base setting, we don't reply on techniques such as heavy data augmentation. In case you want to see the performance with heavy data augmentation and learning rate scheduler, run as follows:

**Baseline (ViT):**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=3900
MODEL_TYPE="dblock"
srun uv run --extra torch main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

**DiffusionBlocks:**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=$((3900 * 3)) # 3 indicates the number of blocks
MODEL_TYPE="dblock"
srun uv run --extra torch main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

</details>

## Evaluation

**Baseline (ViT):**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run --extra torch main.py test cifar100 --model_type vit --ckpt_path $CKPT
```

**DiffusionBlocks:**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run --extra torch main.py test cifar100 --model_type dblock --ckpt_path $CKPT
```

### Evaluation on Apple MLX

MLX checkpoints are saved as `.npz` files:

```bash
CKPT_PATH="logs/path-to-last/last.npz"
uv run mlx_main.py test cifar100 --model_type dblock --ckpt_path $CKPT_PATH
```

## LLM DiffusionBlocks on MLX

The repo also includes an experimental causal-LM DiffusionBlocks trainer for
Apple Silicon. The default LLM path uses the paper-style autoregressive
DiffusionBlocks objective, so each step only updates a slice of the transformer
stack.

Smoke test with the built-in tiny byte-level transformer:

```bash
uv run llm_dblock_train.py --backend tiny --iters 20 --batch_size 4 --seq_len 64
```

Use an `mlx-lm` model by installing the optional LLM dependencies:

```bash
uv sync --extra llm
uv run --extra llm llm_dblock_train.py \
    --backend mlx-lm \
    --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
    --data train.txt \
    --iters 100 \
    --batch_size 1 \
    --seq_len 1024 \
    --num_blocks 8
```

This path is intended for memory-bound fine-tuning experiments. The tiny backend
is verified locally; real `mlx-lm` model families may need adapter shims in
`llm_dblocks/adapters.py` if their layer attribute names differ.

The default LLM objective is `paper_ar`, which follows the AR setup described in
the DiffusionBlocks paper: add noise after token embeddings, concatenate clean
and noisy sequences, apply a causal-consistency mask, and train with
cross-entropy denoising on noisy token positions. The older hidden-state
denoising objective remains available as `--objective hidden` for ablations.
For ordinary next-token fine-tuning behavior, add `--clean_lm_weight`; this keeps
the blockwise trainable-parameter path but adds standard LM CE through the normal
forward pass.

To test the "too large for normal full fine-tuning" claim, use the memory
benchmark harness. Run full backprop and DiffusionBlocks in separate invocations
for the cleanest peak-memory numbers:

```bash
uv run --extra llm qwen_memory_benchmark.py \
    --model mlx-community/Qwen2.5-7B-Instruct-bf16 \
    --mode full \
    --batch_size 1 \
    --seq_len 512 \
    --output_json results/qwen-full.json

uv run --extra llm qwen_memory_benchmark.py \
    --model mlx-community/Qwen2.5-7B-Instruct-bf16 \
    --mode dblock \
    --objective paper_ar \
    --clean_lm_weight 100 \
    --batch_size 1 \
    --seq_len 2048 \
    --num_blocks 8 \
    --output_json results/qwen-dblock.json
```

Increase `--seq_len`, `--batch_size`, or model size until `--mode full` fails or
exceeds available memory while `--mode dblock` still completes.

On an M4 Max MacBook Pro with 36 GB unified memory, the strongest current result
uses `mlx-community/Qwen2.5-7B-Instruct-bf16`, batch size 1, and 8 blocks:

| Sequence length | Full bf16 fine-tune | DiffusionBlocks objective | Peak memory | Result |
| --- | ---: | --- | ---: | --- |
| 512 | killed by OS | mixed `paper_ar` + clean CE | 19.21 GB | completes |
| 2048 | not rerun after 512 kill | pure `paper_ar` | 22.35 GB | completes |
| 2048 | not rerun after 512 kill | mixed `paper_ar` + clean CE | 29.93 GB | quality-aware path completes |
| 4096 | not rerun after 512 kill | pure `paper_ar` | 38.96 GB | edge run; above physical memory |

On Qwen2.5-0.5B with the built-in sample text, pure `paper_ar` improves the
denoising objective but worsens ordinary next-token CE. The mixed objective fixes
that failure mode: `--clean_lm_weight 100` improves standard CE from `2.3672` to
`0.0645` in 100 blockwise steps, versus `0.0280` for full bf16 fine-tuning.

The smaller `mlx-community/Qwen2.5-1.5B-Instruct-bf16` benchmark found a context
boundary:

| Sequence length | Full fine-tune peak | DiffusionBlocks peak | Result |
| --- | ---: | ---: | --- |
| 128 | 12.75 GB | 5.28 GB | both complete |
| 1024 | 12.75 GB | 7.33 GB | both complete |
| 4096 | 29.20 GB | 16.59 GB | both complete |
| 8192 | killed by OS | 33.51 GB | DiffusionBlocks completes |

Quantized 4-bit Qwen weights are not suitable for this full-block trainer because
MLX does not expose gradients for quantized matmul weights. Use bf16/unquantized
MLX weights for block training; use LoRA/QLoRA for quantized adapter training.

See [docs/m4_max_36gb_benchmarks.md](docs/m4_max_36gb_benchmarks.md) for the
full benchmark table, [docs/method_alignment.md](docs/method_alignment.md) for
paper-alignment notes, [docs/upstream_code_mapping.md](docs/upstream_code_mapping.md)
for what was directly translated from the public SakanaAI repo,
[docs/experiment_protocol.md](docs/experiment_protocol.md) for the current
experiment gate, [docs/quality_status.md](docs/quality_status.md) for the
current quality evidence, and [docs/launch_notes.md](docs/launch_notes.md) for
the repo/Twitter positioning notes.

## Acknowledgement

The implementation of Vision Transformer in [vit.py](./vit.py) is based on [HuggingFace Transformers](https://github.com/huggingface/transformers). And, the implementation of EDM is based on [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models).  
We are grateful for their work.

## Citation

To cite our work, please use the following BibTeX:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author.   = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```
