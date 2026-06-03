from __future__ import annotations

import mlx.core as mx

from llm_dblocks.adapters import TinyLMAdapter
from llm_dblocks.speculative import (
    early_exit_logits,
    full_greedy_decode,
    self_speculative_decode,
)
from llm_dblocks.tiny_lm import TinyLMConfig


def tiny_adapter() -> TinyLMAdapter:
    return TinyLMAdapter(
        TinyLMConfig(
            vocab_size=32,
            hidden_size=32,
            num_layers=4,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=32,
        )
    )


def test_full_depth_early_exit_matches_normal_forward():
    adapter = tiny_adapter()
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    partial = early_exit_logits(adapter, input_ids, exit_layer=adapter.num_layers)
    full = adapter.model(input_ids)
    mx.eval(partial, full)

    assert partial.shape == full.shape
    assert bool(mx.allclose(partial, full, atol=1e-5).item())


def test_self_speculative_full_depth_matches_greedy_decode():
    adapter = tiny_adapter()
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    greedy = full_greedy_decode(adapter, prompt, max_new_tokens=6)
    speculative = self_speculative_decode(
        adapter,
        prompt,
        max_new_tokens=6,
        draft_exit_layer=adapter.num_layers,
        draft_tokens=3,
    )
    mx.eval(greedy.tokens, speculative.tokens)

    assert speculative.tokens.tolist() == greedy.tokens.tolist()
    assert speculative.accepted_tokens == 6
    assert speculative.acceptance_rate == 1.0
