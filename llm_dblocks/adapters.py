from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.base import create_attention_mask
except ImportError:
    create_attention_mask = None

from llm_dblocks.tiny_lm import TinyCausalLM, TinyLMConfig


def _resolve_attr(obj: Any, paths: list[str]):
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise AttributeError(f"Could not resolve any of: {paths}")


@dataclass
class ModelAdapter:
    model: nn.Module
    layers: list[nn.Module]
    vocab_size: int

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def embed(self, input_ids: mx.array) -> mx.array:
        raise NotImplementedError

    def run_layers(
        self,
        hidden_states: mx.array,
        start: int,
        end: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        raise NotImplementedError

    def logits_from_hidden(self, hidden_states: mx.array) -> mx.array:
        raise NotImplementedError

    def embedding_weight(self) -> mx.array:
        raise NotImplementedError

    def freeze_for_dblocks(self):
        # Default: every transformer block can be selected by the block-wise loss.
        self.model.freeze()
        for layer in self.layers:
            layer.unfreeze()

    def save_weights(self, path: str):
        self.model.save_weights(path)


class TinyLMAdapter(ModelAdapter):
    def __init__(self, config: TinyLMConfig):
        model = TinyCausalLM(config)
        super().__init__(model=model, layers=model.layers, vocab_size=config.vocab_size)

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.model.embed(input_ids)

    def run_layers(
        self,
        hidden_states: mx.array,
        start: int,
        end: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        return self.model.run_layers(hidden_states, start, end, mask=mask)

    def logits_from_hidden(self, hidden_states: mx.array) -> mx.array:
        return self.model.logits_from_hidden(hidden_states)

    def embedding_weight(self) -> mx.array:
        return self.model.embed_tokens.weight


class MLXLMAdapter(ModelAdapter):
    """Adapter for common mlx-lm causal LM layouts.

    Llama, Mistral, Qwen, and Phi-style mlx-lm models generally expose token
    embeddings, transformer layers, final norm, and lm_head under one of these
    attribute paths. Keeping the logic here makes model-family fixes local.
    """

    def __init__(self, model: nn.Module, tokenizer: Any | None = None):
        self.inner = _resolve_attr(model, ["model", "language_model", "transformer"])
        layers = list(_resolve_attr(self.inner, ["layers", "h"]))
        self.embed_tokens = _resolve_attr(
            self.inner, ["embed_tokens", "wte", "tok_embeddings"]
        )
        self.norm = _resolve_attr(self.inner, ["norm", "ln_f", "final_layernorm"])
        self.lm_head = None
        try:
            self.lm_head = _resolve_attr(model, ["lm_head", "embed_out", "output"])
        except AttributeError:
            # Cohere-style mlx-lm models tie output projection to token embeddings.
            if not hasattr(self.embed_tokens, "as_linear"):
                raise
            self.logit_scale = getattr(getattr(self.inner, "args", None), "logit_scale", 1.0)
        vocab_size = getattr(getattr(model, "args", None), "vocab_size", None)
        if vocab_size is None and tokenizer is not None:
            vocab_size = len(tokenizer)
        if vocab_size is None and hasattr(self.lm_head, "weight"):
            vocab_size = self.lm_head.weight.shape[0]
        if vocab_size is None:
            raise ValueError("Could not infer vocabulary size for mlx-lm model")
        super().__init__(model=model, layers=layers, vocab_size=int(vocab_size))

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids)

    def run_layers(
        self,
        hidden_states: mx.array,
        start: int,
        end: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        if mask is None:
            mask = create_attention_mask(hidden_states) if create_attention_mask else None
        for layer in self.layers[start:end]:
            try:
                out = layer(hidden_states, mask, None)
            except TypeError:
                try:
                    out = layer(hidden_states, mask)
                except TypeError:
                    out = layer(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    def logits_from_hidden(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return self.embed_tokens.as_linear(hidden_states) * self.logit_scale

    def embedding_weight(self) -> mx.array:
        return self.embed_tokens.weight


def load_mlx_lm_adapter(model_name: str, **load_kwargs) -> tuple[MLXLMAdapter, Any]:
    from mlx_lm import load

    model, tokenizer = load(model_name, **load_kwargs)
    return MLXLMAdapter(model, tokenizer), tokenizer
