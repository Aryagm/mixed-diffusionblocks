from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ViTConfig:
    image_size: int
    num_labels: int
    patch_size: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int | None = None
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pooling_type: str = "cls"
    time_conditioning: bool = False
    cond_hidden_size: int | None = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.cond_hidden_size is None:
            self.cond_hidden_size = self.hidden_size // 6
        if self.pooling_type not in {"cls", "mean"}:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")


@dataclass
class ViTOutput:
    last_hidden_state: mx.array
    conditioning: Optional[mx.array] = None


def _normal(shape, std: float = 0.02):
    return mx.random.normal(shape, dtype=mx.float32) * std


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + mx.expand_dims(scale, 1)) + mx.expand_dims(shift, 1)


class PatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        patch_dim = config.patch_size * config.patch_size * 3
        self.projection = nn.Linear(patch_dim, config.hidden_size)
        self.patch_size = config.patch_size

    def __call__(self, pixel_values: mx.array) -> mx.array:
        batch, height, width, channels = pixel_values.shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError("Image dimensions must be divisible by the patch size")
        patches = pixel_values.reshape(
            batch, height // patch, patch, width // patch, patch, channels
        )
        patches = mx.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = patches.reshape(batch, -1, patch * patch * channels)
        return self.projection(patches)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period)
            * mx.arange(0, half, dtype=mx.float32)
            / mx.array(half, dtype=mx.float32)
        )
        args = mx.expand_dims(t, -1) * mx.expand_dims(freqs, 0)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.linear2(nn.silu(self.linear1(t_freq)))


class AdaLN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.linear(x))


class MLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, x: mx.array) -> mx.array:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        probs = mx.softmax(scores, axis=-1)
        probs = self.attn_dropout(probs)
        out = probs @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, tokens, hidden)
        return self.proj_dropout(self.proj(out))


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        if config.time_conditioning:
            self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 6 * config.hidden_size
            )

    def __call__(
        self, hidden_states: mx.array, conditioning: Optional[mx.array] = None
    ) -> mx.array:
        residual = hidden_states
        if self.time_conditioning:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
                self.adaLN_modulation(conditioning), 6, axis=1
            )
        hidden_states = self.norm1(hidden_states)
        if self.time_conditioning:
            hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        attention_output = self.attention(hidden_states)
        if self.time_conditioning:
            attention_output = mx.expand_dims(gate_msa, 1) * attention_output
        hidden_states = residual + attention_output

        layer_output = self.norm2(hidden_states)
        if self.time_conditioning:
            layer_output = modulate(layer_output, shift_mlp, scale_mlp)
        layer_output = self.mlp(layer_output)
        if self.time_conditioning:
            layer_output = mx.expand_dims(gate_mlp, 1) * layer_output
        return hidden_states + layer_output


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.time_conditioning = config.time_conditioning
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = _normal(
            (1, num_patches + 1, config.hidden_size),
            std=0.02,
        )
        self.cls_token = None
        if not config.time_conditioning:
            self.cls_token = _normal((1, 1, config.hidden_size), std=0.02)
        if config.time_conditioning:
            self.label_embeddings = nn.Embedding(config.num_labels, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self, pixel_values: mx.array, noisy_embeds: Optional[mx.array] = None
    ) -> mx.array:
        batch = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        if self.time_conditioning and noisy_embeds is not None:
            cls_tokens = mx.expand_dims(noisy_embeds, 1)
        else:
            cls_tokens = mx.broadcast_to(
                self.cls_token, (batch, 1, self.cls_token.shape[-1])
            )
        embeddings = mx.concatenate([cls_tokens, embeddings], axis=1)
        return self.dropout(embeddings + self.position_embeddings)


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.layers = [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.time_conditioning:
            self.time_embedder = TimestepEmbedder(config.cond_hidden_size)

    def get_input_embeddings(self):
        if self.config.time_conditioning:
            return self.embeddings.label_embeddings
        return self.embeddings.patch_embeddings

    def __call__(
        self,
        pixel_values: mx.array,
        timesteps: Optional[mx.array] = None,
        noisy_embeds: Optional[mx.array] = None,
        layer_indices: Optional[list[int]] = None,
    ) -> ViTOutput:
        hidden_states = self.embeddings(pixel_values, noisy_embeds=noisy_embeds)
        conditioning = None
        if self.config.time_conditioning:
            conditioning = nn.silu(self.time_embedder(timesteps))

        active_layers = set(layer_indices) if layer_indices is not None else None
        for i, layer in enumerate(self.layers):
            if active_layers is not None and i not in active_layers:
                continue
            hidden_states = layer(hidden_states, conditioning=conditioning)

        hidden_states = self.layernorm(hidden_states)
        return ViTOutput(last_hidden_state=hidden_states, conditioning=conditioning)


class ViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.vit = ViT(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if config.time_conditioning:
            self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 2 * config.hidden_size
            )
            self._init_dit()

    def _init_dit(self):
        self.vit.embeddings.label_embeddings.weight = _normal(
            self.vit.embeddings.label_embeddings.weight.shape, std=0.02
        )
        for block in self.vit.layers:
            block.adaLN_modulation.linear.weight = mx.zeros_like(
                block.adaLN_modulation.linear.weight
            )
            block.adaLN_modulation.linear.bias = mx.zeros_like(
                block.adaLN_modulation.linear.bias
            )
        self.adaLN_modulation.linear.weight = mx.zeros_like(
            self.adaLN_modulation.linear.weight
        )
        self.adaLN_modulation.linear.bias = mx.zeros_like(
            self.adaLN_modulation.linear.bias
        )
        self.classifier.weight = mx.zeros_like(self.classifier.weight)
        self.classifier.bias = mx.zeros_like(self.classifier.bias)

    def get_input_embeddings(self):
        return self.vit.get_input_embeddings()

    def forward_block(
        self,
        layer_indices: list[int],
        pixel_values: mx.array,
        timesteps: mx.array,
        noisy_embeds: mx.array,
    ) -> ViTOutput:
        outputs = self.vit(
            pixel_values=pixel_values,
            timesteps=timesteps,
            noisy_embeds=noisy_embeds,
            layer_indices=layer_indices,
        )
        if self.config.pooling_type == "cls":
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
        else:
            last_hidden_state = mx.mean(outputs.last_hidden_state[:, 1:, :], axis=1)
        return ViTOutput(
            last_hidden_state=last_hidden_state,
            conditioning=outputs.conditioning,
        )

    def forward_output_embeddings(
        self, hidden_states: mx.array, conditioning: mx.array
    ) -> mx.array:
        if self.config.time_conditioning:
            shift, scale = mx.split(self.adaLN_modulation(conditioning), 2, axis=1)
            hidden_states = modulate(hidden_states, shift, scale)
        return self.classifier(hidden_states[:, 0, :])

    def __call__(self, pixel_values: mx.array) -> mx.array:
        outputs = self.vit(pixel_values=pixel_values)
        if self.config.pooling_type == "cls":
            sequence_output = outputs.last_hidden_state[:, 0, :]
        else:
            sequence_output = mx.mean(outputs.last_hidden_state[:, 1:, :], axis=1)
        return self.classifier(sequence_output)


def load_vit(image_size: int, num_labels: int, is_dblock: bool = False, **kwargs):
    if image_size == 32:
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_hidden_layers", 12)
        kwargs.setdefault("hidden_size", 128)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
    elif image_size == 64:
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_hidden_layers", 12)
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("num_attention_heads", 12)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
    else:
        raise ValueError(f"Invalid image size: {image_size}")

    config = ViTConfig(
        image_size=image_size,
        num_labels=num_labels,
        time_conditioning=is_dblock,
        **kwargs,
    )
    return ViTForImageClassification(config)
