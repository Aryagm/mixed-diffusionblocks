# Alpaca SFT Benchmark

This benchmark tests actual supervised fine-tuning rather than next-token
continued pretraining. It uses `tatsu-lab/alpaca`, formats rows as chat
messages, and evaluates response-only validation cross entropy.

## Commands

```bash
uv run --extra llm mixed-dblocks-sft-quality \
  --mode full \
  --model mlx-community/Qwen2.5-0.5B-bf16 \
  --train_examples 2048 \
  --valid_examples 256 \
  --test_examples 256 \
  --steps 100 \
  --eval_batches 16 \
  --seq_len 512 \
  --batch_size 1 \
  --lr 1e-5 \
  --output_json docs/results/alpaca_qwen05_base_sft_full_lr1e5_100.json
```

The checked-in comparison was run as separate one-method processes with
method-specific learning rates to avoid cross-run memory retention. The
blockwise rows use `--lr 3e-6`; LoRA and QLoRA use `--lora_lr 1e-5` and
`--qlora_lr 1e-5`. The separate JSON outputs were consolidated into
`docs/results/alpaca_qwen05_base_sft_comparison.json`.

## Result

| Method | Steps | Response CE before | Response CE after | Time | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full bf16 AR | 100 | 2.6217 | 1.6144 | 30.96 s | 5.13 GB |
| Windowed blockwise | 100 | 2.6217 | 2.0890 | 8.44 s | 4.13 GB |
| Windowed blockwise, equal time | 360 | 2.6217 | 1.9377 | 31.20 s | 4.21 GB |
| Full-anchor blockwise | 100 | 2.6217 | 1.8492 | 19.04 s | 4.21 GB |
| Full-anchor blockwise, equal time | 160 | 2.6217 | 1.8349 | 28.30 s | 4.21 GB |
| LoRA | 100 | 2.6217 | 1.7279 | 5.58 s | 2.22 GB |
| QLoRA, 4-bit base | 100 | 2.8224 | 1.7889 | 5.31 s | 1.61 GB |

## Interpretation

This is not evidence that the current blockwise method should replace LoRA or
tuned full fine-tuning for small-model SFT. LoRA is both faster and better on
this benchmark, and tuned full AR reaches the best response CE.

The useful signal is narrower:

- blockwise training updates real bf16 transformer blocks;
- it uses less peak memory than full bf16 AR on the same model and sequence;
- full-anchor blockwise is the stronger SFT variant;
- the windowed default is a speed/memory mode, not the SFT-quality mode.

For a paper, this should be framed as a production benchmark and a current
limitation. The next technique work should target response-aware anchor windows
or adaptive block schedules before claiming default status for SFT.
