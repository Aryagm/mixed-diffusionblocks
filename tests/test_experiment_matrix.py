from llm_dblocks.experiments import publish_matrix, render_commands


def test_publish_matrix_includes_required_quality_ablations():
    specs = publish_matrix(
        models=["mlx-community/Qwen2.5-0.5B-Instruct-bf16"],
        seeds=[1, 2],
        data="corpora/wikitext2/train.txt",
        val_data="corpora/wikitext2/validation.txt",
        output_dir="docs/results/publish",
        include_memory=False,
    )

    names = {spec.name for spec in specs}

    assert "qwen25_05b_full_seed1" in names
    assert "qwen25_05b_pure_dblocks_seed1" in names
    assert "qwen25_05b_full_anchor_mixed_seed1" in names
    assert "qwen25_05b_auto_window_seed1" in names
    assert "qwen25_05b_auto_window_seed2" in names


def test_render_commands_contains_safe_auto_window_default():
    specs = publish_matrix(
        models=["mlx-community/Qwen2.5-0.5B-Instruct-bf16"],
        seeds=[42],
        data="train.txt",
        val_data="valid.txt",
        output_dir="out",
        include_memory=False,
    )

    commands = render_commands(specs)

    assert any("--clean_lm_anchor_profile auto_window" in cmd for cmd in commands)
    assert any(
        "--clean_lm_anchor_profile manual --clean_lm_weight 0" in cmd
        for cmd in commands
    )


def test_memory_specs_are_opt_in_and_marked_heavy():
    without_memory = publish_matrix(
        models=["mlx-community/Qwen2.5-7B-Instruct-bf16"],
        seeds=[42],
        data="train.txt",
        val_data="valid.txt",
        output_dir="out",
        include_memory=False,
    )
    with_memory = publish_matrix(
        models=["mlx-community/Qwen2.5-7B-Instruct-bf16"],
        seeds=[42],
        data="train.txt",
        val_data="valid.txt",
        output_dir="out",
        include_memory=True,
    )

    assert not any(spec.name.endswith("_memory_seq2048") for spec in without_memory)
    assert any(
        spec.name.endswith("_memory_seq2048") and spec.heavy for spec in with_memory
    )
