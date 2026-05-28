# Launch Notes

## Repository Positioning

This repo is not "another MLX LoRA trainer." The claim is narrower and more
interesting:

> Train bf16 transformer blocks on Apple Silicon when normal full bf16
> fine-tuning does not fit.

The current best demo is Qwen2.5-7B-Instruct-bf16 on an M4 Max 36 GB Mac:

- Full bf16 fine-tuning is killed by the OS at 512 tokens.
- DiffusionBlocks with the pure paper-style AR objective completes 2048-token
  block training at 22.35 GB peak.
- Mixed DiffusionBlocks + standard next-token CE completes 2048-token block
  training at 29.93 GB peak.
- The mixed objective improves ordinary next-token CE on Qwen2.5-0.5B
  (`2.3672 -> 0.0645` in 100 steps), while pure `paper_ar` worsens ordinary CE
  (`2.3672 -> 3.9473`).

## Twitter/X Draft

I got bf16 Qwen2.5-7B block training running on a 36 GB M4 Max MacBook.

Normal full bf16 fine-tuning was killed by macOS even at 512 tokens.

DiffusionBlocks on MLX:

- 1024 ctx: 19.2 GB
- 2048 ctx, pure paper objective: 22.4 GB
- 2048 ctx, paper objective + clean CE: 29.9 GB

No CUDA. No LoRA. Training actual bf16 transformer blocks.

## GitHub README Headline

Train bigger bf16 LLM blocks on Apple Silicon with DiffusionBlocks + MLX.

On an M4 Max 36 GB MacBook Pro, normal full bf16 training of
Qwen2.5-7B-Instruct-bf16 is killed even at 512 tokens. This repo's
mixed DiffusionBlocks trainer completes 2048-token block training at 29.93 GB
peak while adding a standard next-token CE term for ordinary LM behavior.

## What To Avoid Claiming

- Do not claim final model quality yet. We have a small Qwen0.5B quality sweep
  and 7B memory-boundary proof, not a full instruction-tuning benchmark.
- Do not claim this beats LoRA/QLoRA on parameter count. It is a different
  training mode.
- Do not claim 4-bit full-block training. MLX quantized matmul weights do not
  expose gradients.
