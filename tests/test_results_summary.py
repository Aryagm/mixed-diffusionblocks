from pathlib import Path

from llm_dblocks.results import load_result_rows, markdown_table


def test_load_result_rows_extracts_nested_quality_metrics(tmp_path: Path):
    path = tmp_path / "quality.json"
    path.write_text(
        """
{
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
  "seq_len": 1024,
  "modes": {
    "dblock": {
      "ok": true,
      "clean_lm_anchor_profile": "auto_window",
      "next_token_ce_before": 2.0,
      "next_token_ce_after": 1.5,
      "seconds": 12.0
    }
  }
}
""",
        encoding="utf-8",
    )

    rows = load_result_rows([path])

    assert rows == [
        {
            "file": "quality.json",
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "seq_len": 1024,
            "mode": "dblock",
            "profile": "auto_window",
            "before": 2.0,
            "after": 1.5,
            "seconds": 12.0,
            "peak_gb": None,
        }
    ]


def test_load_result_rows_extracts_memory_peak(tmp_path: Path):
    path = tmp_path / "memory.json"
    path.write_text(
        """
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-bf16",
  "seq_len": 2048,
  "modes": {
    "dblock": {
      "ok": true,
      "clean_lm_anchor_profile": "auto_window",
      "seconds": 6.61,
      "peak_gb": 22.31
    }
  }
}
""",
        encoding="utf-8",
    )

    rows = load_result_rows([path])

    assert rows[0]["peak_gb"] == 22.31
    assert rows[0]["before"] is None
    assert rows[0]["after"] is None


def test_markdown_table_renders_rows():
    table = markdown_table(
        [
            {
                "file": "x.json",
                "model": "m",
                "seq_len": 128,
                "mode": "dblock",
                "profile": "auto_window",
                "before": 2.0,
                "after": 1.5,
                "seconds": 12.0,
                "peak_gb": None,
            }
        ]
    )

    assert (
        "| File | Model | Seq | Mode | Profile | Before | After | Seconds | Peak GB |"
        in table
    )
    assert "| x.json | m | 128 | dblock | auto_window | 2.0000 | 1.5000 | 12.00 |  |" in table
