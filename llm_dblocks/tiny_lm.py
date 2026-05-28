from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TinyLMConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    num_layers: int = 6
    num_heads: int = 4
    intermediate_size: int = 512
    max_seq_len: int = 256


def causal_mask(seq_len: int) -> mx.array:
    rows = mx.arange(seq_len)[:, None]
    cols = mx.arange(seq_len)[None, :]
    return cols <= rows


class TinySelfAttention(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | str | None = None) -> mx.array:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            mask = causal_mask(tokens)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask[None, None, :, :], scores, -1e9)
        else:
            scores = scores + mask
        probs = mx.softmax(scores, axis=-1)
        out = probs @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, tokens, hidden)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = TinySelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | str | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        h = self.fc2(nn.gelu(self.fc1(self.mlp_norm(x))))
        return x + h


class TinyCausalLM(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = [TinyBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def embed(self, input_ids: mx.array) -> mx.array:
        _, seq_len = input_ids.shape
        positions = mx.arange(seq_len)[None, :]
        return self.embed_tokens(input_ids) + self.position_embeddings(positions)

    def run_layers(
        self,
        hidden_states: mx.array,
        start: int,
        end: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        for layer in self.layers[start:end]:
            hidden_states = layer(hidden_states, mask=mask)
        return hidden_states

    def logits_from_hidden(self, hidden_states: mx.array) -> mx.array:
        return self.lm_head(self.norm(hidden_states))

    def __call__(self, input_ids: mx.array) -> mx.array:
        hidden_states = self.embed(input_ids)
        hidden_states = self.run_layers(hidden_states, 0, len(self.layers))
        return self.logits_from_hidden(hidden_states)
