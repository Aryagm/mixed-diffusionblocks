import mlx.core as mx

from llm_dblocks.adapters import TinyLMAdapter
from llm_dblocks.tiny_lm import TinyLMConfig
from llm_dblocks.trainer import (
    DBlockTrainer,
    DBlockTrainingConfig,
    balanced_block_ranges,
    block_ranges_from_boundaries,
    scalar,
)


def tiny_batch():
    return {
        "input_ids": mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        "labels": mx.array([[2, 3, 4, 5]], dtype=mx.int32),
    }


def long_tiny_batch():
    return {
        "input_ids": mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32),
        "labels": mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32),
    }


def tiny_adapter():
    return TinyLMAdapter(
        TinyLMConfig(
            vocab_size=16,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            intermediate_size=32,
            max_seq_len=8,
        )
    )


def test_paper_ar_loss_reports_local_anchor_metric():
    trainer = DBlockTrainer(
        tiny_adapter(),
        DBlockTrainingConfig(num_blocks=2, local_lm_weight=10.0),
    )

    loss, metrics = trainer.loss(trainer.adapter.model, tiny_batch(), block_idx=0)
    mx.eval(loss, metrics["local_lm_loss"])

    assert "local_lm_loss" in metrics
    assert scalar(metrics["local_lm_loss"]) > 0.0


def test_local_anchor_does_not_call_exact_clean_lm_loss():
    trainer = DBlockTrainer(
        tiny_adapter(),
        DBlockTrainingConfig(
            num_blocks=2,
            clean_lm_weight=0.0,
            local_lm_weight=10.0,
        ),
    )

    def fail_if_called(model, batch):
        raise AssertionError("local anchor should not run exact clean LM CE")

    trainer.clean_next_token_loss = fail_if_called

    loss, metrics = trainer.loss(trainer.adapter.model, tiny_batch(), block_idx=0)
    mx.eval(loss, metrics["local_lm_loss"])

    assert scalar(metrics["local_lm_loss"]) > 0.0


def test_exact_clean_anchor_can_be_skipped_for_budgeted_updates():
    trainer = DBlockTrainer(
        tiny_adapter(),
        DBlockTrainingConfig(
            num_blocks=2,
            clean_lm_weight=10.0,
        ),
    )

    def fail_if_called(model, batch):
        raise AssertionError("budgeted update should skip exact clean LM CE")

    trainer.clean_next_token_loss = fail_if_called

    loss, metrics = trainer.loss(
        trainer.adapter.model,
        tiny_batch(),
        block_idx=0,
        use_clean_lm=False,
    )
    mx.eval(loss, metrics["clean_lm_loss"])

    assert scalar(metrics["clean_lm_loss"]) == 0.0


def test_clean_anchor_batch_can_use_short_random_window():
    trainer = DBlockTrainer(
        tiny_adapter(),
        DBlockTrainingConfig(num_blocks=2, clean_lm_seq_len=3),
    )

    cropped = trainer.clean_anchor_batch(long_tiny_batch())

    assert cropped["input_ids"].shape == (1, 3)
    assert cropped["labels"].shape == (1, 3)


def test_clean_anchor_batch_can_force_full_sequence_for_warmup():
    trainer = DBlockTrainer(
        tiny_adapter(),
        DBlockTrainingConfig(num_blocks=2, clean_lm_seq_len=3),
    )

    full = trainer.clean_anchor_batch(long_tiny_batch(), force_full=True)

    assert full["input_ids"].shape == (1, 6)
    assert full["labels"].shape == (1, 6)


def test_custom_block_boundaries_create_contiguous_ranges():
    assert block_ranges_from_boundaries(6, (1, 4)) == [(0, 1), (1, 4), (4, 6)]


def test_balanced_block_ranges_uses_layer_scores():
    ranges = balanced_block_ranges([9.0, 1.0, 1.0, 1.0], 2)

    assert ranges == [(0, 1), (1, 4)]
