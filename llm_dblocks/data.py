from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np


SAMPLE_TEXT = """
DiffusionBlocks trains a transformer one block at a time. The goal is to reduce
activation memory while keeping the model useful for downstream generation.
Apple Silicon has unified memory, so a block-wise MLX trainer can target models
and context lengths that are awkward for ordinary full backpropagation.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size


def load_text(path: str | None, text_field: str = "text") -> str:
    if path is None or path == "sample":
        return SAMPLE_TEXT * 256

    data_path = Path(path)
    if data_path.suffix == ".jsonl":
        rows = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line)[text_field])
        return "\n".join(rows)

    return data_path.read_text(encoding="utf-8")


def tokenize_text(tokenizer, text: str) -> np.ndarray:
    tokens = tokenizer.encode(text)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    return np.asarray(tokens, dtype=np.int32)


def batch_iterator(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    *,
    shuffle: bool = True,
) -> Iterator[dict[str, mx.array]]:
    if len(tokens) < seq_len + 2:
        raise ValueError("Text is too short for the requested sequence length")

    while True:
        max_start = len(tokens) - seq_len - 1
        if shuffle:
            starts = [random.randint(0, max_start) for _ in range(batch_size)]
        else:
            starts = list(range(0, batch_size))

        inputs = np.stack([tokens[s : s + seq_len] for s in starts])
        labels = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts])
        yield {
            "input_ids": mx.array(inputs, dtype=mx.int32),
            "labels": mx.array(labels, dtype=mx.int32),
        }
