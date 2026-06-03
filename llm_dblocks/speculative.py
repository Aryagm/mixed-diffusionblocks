from __future__ import annotations

import importlib
from dataclasses import dataclass

import mlx.core as mx

from llm_dblocks.adapters import ModelAdapter, create_attention_mask

mlx_cache = importlib.import_module("mlx_lm.models.cache")


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


@dataclass
class _LayerView:
    layers: list


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


def _run_cached_layers(hidden: mx.array, layers: list, cache: list) -> mx.array:
    if not layers:
        return hidden
    mask = create_attention_mask(hidden, cache[0]) if create_attention_mask else None
    for layer, layer_cache in zip(layers, cache):
        try:
            out = layer(hidden, mask, layer_cache)
        except TypeError:
            out = layer(hidden, mask)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


def _prefix_hidden(
    adapter: ModelAdapter,
    input_ids: mx.array,
    *,
    exit_layer: int,
    cache: list,
) -> mx.array:
    hidden = adapter.embed(input_ids)
    return _run_cached_layers(hidden, adapter.layers[:exit_layer], cache)


def _suffix_logits(
    adapter: ModelAdapter,
    hidden: mx.array,
    *,
    exit_layer: int,
    cache: list,
) -> mx.array:
    hidden = _run_cached_layers(hidden, adapter.layers[exit_layer:], cache)
    return adapter.logits_from_hidden(hidden)


def _trim(cache: list, num_tokens: int):
    if num_tokens > 0 and cache:
        mlx_cache.trim_prompt_cache(cache, num_tokens)


def cached_greedy_generate_step(
    prompt: mx.array,
    adapter: ModelAdapter,
    *,
    max_tokens: int,
    prefill_step_size: int = 512,
):
    """Greedy cached generation without per-token logprob materialization."""
    if prompt.ndim != 1:
        raise ValueError("prompt must be a 1D token array")
    if prompt.size < 1:
        raise ValueError("prompt must contain at least one token")
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")

    model_view = _LayerView(adapter.layers)
    model_cache = mlx_cache.make_prompt_cache(model_view)

    prefill = prompt[:-1]
    while prefill.size > 0:
        n_to_process = min(prefill_step_size, prefill.size)
        chunk = prefill[:n_to_process][None]
        try:
            adapter.model(chunk, cache=model_cache)
        except TypeError:
            adapter.model(chunk)
        mx.eval([c.state for c in model_cache])
        prefill = prefill[n_to_process:]
        mx.clear_cache()

    current = prompt[-1:].astype(mx.int32)
    for _ in range(max_tokens):
        try:
            logits = adapter.model(current[None], cache=model_cache)
        except TypeError:
            logits = adapter.model(current[None])
        current = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(current)
        yield int(current.item()), None, False


def prefix_reuse_speculative_generate_step(
    prompt: mx.array,
    adapter: ModelAdapter,
    *,
    exit_layer: int,
    num_draft_tokens: int,
    max_tokens: int,
    prefill_step_size: int = 512,
):
    """Greedy self-speculation that reuses draft prefix-layer work.

    This generator preserves full-model greedy outputs while avoiding the main
    overhead in ordinary self-speculation: rerunning the draft prefix layers
    during verification. It supports batch-1 MLX causal LMs with trimmable KV
    caches.
    """
    if prompt.ndim != 1:
        raise ValueError("prompt must be a 1D token array")
    if num_draft_tokens < 1:
        raise ValueError("num_draft_tokens must be >= 1")
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if exit_layer < 0 or exit_layer > adapter.num_layers:
        raise ValueError(
            f"exit_layer must be between 0 and {adapter.num_layers}, got {exit_layer}"
        )
    if prompt.size < 1:
        raise ValueError("prompt must contain at least one token")

    prefix_model = _LayerView(adapter.layers[:exit_layer])
    suffix_model = _LayerView(adapter.layers[exit_layer:])
    prefix_cache = mlx_cache.make_prompt_cache(prefix_model)
    suffix_cache = mlx_cache.make_prompt_cache(suffix_model)
    if not mlx_cache.can_trim_prompt_cache(prefix_cache):
        raise ValueError("prefix cache is not trimmable")
    if not mlx_cache.can_trim_prompt_cache(suffix_cache):
        raise ValueError("suffix cache is not trimmable")

    # Prefill all prompt tokens except the current generation input token.
    prefill = prompt[:-1]
    while prefill.size > 0:
        n_to_process = min(prefill_step_size, prefill.size)
        chunk = prefill[:n_to_process][None]
        hidden = _prefix_hidden(
            adapter,
            chunk,
            exit_layer=exit_layer,
            cache=prefix_cache,
        )
        _suffix_logits(adapter, hidden, exit_layer=exit_layer, cache=suffix_cache)
        mx.eval([c.state for c in prefix_cache], [c.state for c in suffix_cache])
        prefill = prefill[n_to_process:]
        mx.clear_cache()

    current = prompt[-1:].astype(mx.int32)
    generated = 0

    while generated < max_tokens:
        remaining = max_tokens - generated
        draft_count = min(num_draft_tokens, remaining)
        draft_tokens: list[int] = []
        draft_inputs = []
        hidden_steps = []
        y = current

        for _ in range(draft_count):
            hidden = _prefix_hidden(
                adapter,
                y[None],
                exit_layer=exit_layer,
                cache=prefix_cache,
            )
            hidden_steps.append(hidden)
            draft_inputs.append(int(y.item()))
            logits = adapter.logits_from_hidden(hidden)
            y = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(y)
            draft_tokens.append(int(y.item()))

        verify_hidden = mx.concatenate(hidden_steps, axis=1)
        verify_logits = _suffix_logits(
            adapter,
            verify_hidden,
            exit_layer=exit_layer,
            cache=suffix_cache,
        )
        verify_tokens = mx.argmax(verify_logits, axis=-1)
        logprobs = verify_logits - mx.logsumexp(verify_logits, axis=-1, keepdims=True)
        mx.eval(verify_tokens, logprobs)
        verified = [int(token) for token in verify_tokens[0].tolist()]

        accepted = 0
        for draft_token, verified_token in zip(draft_tokens, verified):
            if draft_token != verified_token:
                break
            accepted += 1
            generated += 1
            yield draft_token, logprobs[0, accepted - 1], True
            if generated == max_tokens:
                break

        if generated == max_tokens:
            break

        if accepted == draft_count:
            current = mx.array([draft_tokens[-1]], dtype=mx.int32)
            continue

        replacement = verified[accepted]
        generated += 1
        current = mx.array([replacement], dtype=mx.int32)
        yield replacement, logprobs[0, accepted], False

        # Drafting processed current + draft tokens up to the token before the
        # last proposal. Keep only the cache entries that are true context for
        # the next current token.
        trim_count = draft_count - (accepted + 1)
        _trim(prefix_cache, trim_count)
        _trim(suffix_cache, trim_count)


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
