from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from llm_dblocks.adapters import ModelAdapter, create_attention_mask


@dataclass
class DecodeResult:
    tokens: mx.array
    generated_tokens: int
    draft_tokens: int = 0
    accepted_tokens: int = 0
    verify_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens


class EarlyExitDraftModel:
    """Cache-compatible draft model that reuses a prefix of a loaded LM."""

    def __init__(self, adapter: ModelAdapter, *, exit_layer: int):
        if exit_layer < 0 or exit_layer > adapter.num_layers:
            raise ValueError(
                f"exit_layer must be between 0 and {adapter.num_layers}, got {exit_layer}"
            )
        self.adapter = adapter
        self.exit_layer = exit_layer
        self.layers = adapter.layers[:exit_layer]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = self.adapter.embed(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            hidden = self.adapter.run_layers(hidden, 0, self.exit_layer)
        else:
            mask = create_attention_mask(hidden, cache[0]) if create_attention_mask else None
            for layer, layer_cache in zip(self.layers, cache):
                try:
                    out = layer(hidden, mask, layer_cache)
                except TypeError:
                    out = layer(hidden, mask)
                hidden = out[0] if isinstance(out, tuple) else out
        return self.adapter.logits_from_hidden(hidden)


def early_exit_logits(
    adapter: ModelAdapter,
    input_ids: mx.array,
    *,
    exit_layer: int,
) -> mx.array:
    if exit_layer < 0 or exit_layer > adapter.num_layers:
        raise ValueError(
            f"exit_layer must be between 0 and {adapter.num_layers}, got {exit_layer}"
        )
    hidden = adapter.embed(input_ids)
    hidden = adapter.run_layers(hidden, 0, exit_layer)
    return adapter.logits_from_hidden(hidden)


def _argmax_next_token(logits: mx.array) -> mx.array:
    return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)[:, None]


def _append_token(input_ids: mx.array, token: mx.array) -> mx.array:
    return mx.concatenate([input_ids, token], axis=1)


def full_greedy_decode(
    adapter: ModelAdapter,
    prompt: mx.array,
    *,
    max_new_tokens: int,
) -> DecodeResult:
    tokens = prompt
    for _ in range(max_new_tokens):
        logits = adapter.model(tokens)
        next_token = _argmax_next_token(logits)
        tokens = _append_token(tokens, next_token)
        mx.eval(tokens)
    return DecodeResult(tokens=tokens, generated_tokens=max_new_tokens)


def _draft_greedy(
    adapter: ModelAdapter,
    prefix: mx.array,
    *,
    max_draft_tokens: int,
    exit_layer: int,
) -> tuple[mx.array, list[int]]:
    tokens = prefix
    draft: list[int] = []
    for _ in range(max_draft_tokens):
        logits = early_exit_logits(adapter, tokens, exit_layer=exit_layer)
        next_token = _argmax_next_token(logits)
        mx.eval(next_token)
        draft.append(int(next_token.item()))
        tokens = _append_token(tokens, next_token)
    return tokens, draft


def self_speculative_decode(
    adapter: ModelAdapter,
    prompt: mx.array,
    *,
    max_new_tokens: int,
    draft_exit_layer: int,
    draft_tokens: int,
) -> DecodeResult:
    if prompt.shape[0] != 1:
        raise ValueError("self_speculative_decode currently supports batch size 1")
    if draft_tokens < 1:
        raise ValueError("draft_tokens must be >= 1")

    tokens = prompt
    generated = 0
    proposed = 0
    accepted = 0
    verify_calls = 0

    while generated < max_new_tokens:
        prefix_len = tokens.shape[1]
        remaining = max_new_tokens - generated
        proposal_len = min(draft_tokens, remaining)
        candidate, draft = _draft_greedy(
            adapter,
            tokens,
            max_draft_tokens=proposal_len,
            exit_layer=draft_exit_layer,
        )
        proposed += len(draft)

        logits = adapter.model(candidate)
        verify_calls += 1
        verify = mx.argmax(
            logits[:, prefix_len - 1 : prefix_len - 1 + len(draft), :],
            axis=-1,
        )
        mx.eval(verify)
        verified = [int(token) for token in verify[0].tolist()]

        accepted_this_round = 0
        for draft_token, verified_token in zip(draft, verified):
            if draft_token != verified_token:
                break
            accepted_this_round += 1

        if accepted_this_round:
            tokens = candidate[:, : prefix_len + accepted_this_round]
            generated += accepted_this_round
            accepted += accepted_this_round
            mx.eval(tokens)

        if accepted_this_round < len(draft) and generated < max_new_tokens:
            replacement = mx.array(
                [[verified[accepted_this_round]]],
                dtype=mx.int32,
            )
            tokens = _append_token(tokens, replacement)
            generated += 1
            mx.eval(tokens)

    return DecodeResult(
        tokens=tokens,
        generated_tokens=generated,
        draft_tokens=proposed,
        accepted_tokens=accepted,
        verify_calls=verify_calls,
    )
