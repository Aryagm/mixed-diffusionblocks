from __future__ import annotations

from speculative_inference_benchmark import default_exit_layers, parse_csv_ints


def test_parse_csv_ints_rejects_empty_values():
    assert parse_csv_ints("1, 2,4") == [1, 2, 4]

    try:
        parse_csv_ints("1,,4")
    except ValueError as exc:
        assert "empty value" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_default_exit_layers_are_unique_and_ordered():
    assert default_exit_layers(4) == [1, 2, 3]
    assert default_exit_layers(24) == [4, 6, 8, 12, 16, 18, 20]
