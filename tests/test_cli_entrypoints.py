import llm_dblock_train
import llm_quality_benchmark
import qwen_memory_benchmark
import qwen_quality_benchmark


def test_train_parser_defaults_to_auto_window():
    parser = llm_dblock_train.build_parser()
    args = parser.parse_args([])

    assert args.clean_lm_anchor_profile == "auto_window"
    assert args.clean_lm_weight == 100.0
    assert args.denoise_weight == 1.0


def test_qwen_quality_parser_defaults_to_auto_window():
    parser = qwen_quality_benchmark.build_parser()
    args = parser.parse_args([])

    assert args.clean_lm_anchor_profile == "auto_window"
    assert args.clean_lm_weight == 100.0
    assert args.denoise_weight == 1.0


def test_memory_parser_can_select_pure_dblocks():
    parser = qwen_memory_benchmark.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "--clean_lm_anchor_profile",
            "manual",
            "--clean_lm_weight",
            "0",
        ]
    )

    assert args.clean_lm_anchor_profile == "manual"
    assert args.clean_lm_weight == 0.0


def test_tiny_quality_parser_has_auto_window_default():
    parser = llm_quality_benchmark.build_parser()
    args = parser.parse_args([])

    assert args.clean_lm_anchor_profile == "auto_window"
    assert args.clean_lm_weight == 100.0
