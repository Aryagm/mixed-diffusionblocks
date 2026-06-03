# Production Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Windowed Mixed DiffusionBlocks prototype into a usable Python library and reproducible benchmark package for full-block LLM fine-tuning.

**Architecture:** Keep `llm_dblocks.trainer` as the core training API. Add small importable modules for CLI defaults, publish-matrix generation, and result summarization, while preserving the existing script entry points as runnable examples.

**Tech Stack:** Python 3.12, MLX, mlx-lm, argparse, pytest, JSON result artifacts.

---

### Task 1: Production CLI Defaults And Entry Points

**Files:**
- Modify: `llm_dblock_train.py`
- Modify: `qwen_quality_benchmark.py`
- Modify: `qwen_memory_benchmark.py`
- Modify: `llm_quality_benchmark.py`
- Modify: `pyproject.toml`
- Create: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli_entrypoints.py`:

```python
import llm_dblock_train
import llm_quality_benchmark
import qwen_memory_benchmark
import qwen_quality_benchmark


def test_train_parser_defaults_to_auto_window():
    parser = llm_dblock_train.build_parser()
    args = parser.parse_args([])

    assert args.clean_lm_anchor_profile == "auto_window"
    assert args.clean_lm_weight == 100.0


def test_qwen_quality_parser_defaults_to_auto_window():
    parser = qwen_quality_benchmark.build_parser()
    args = parser.parse_args([])

    assert args.clean_lm_anchor_profile == "auto_window"
    assert args.clean_lm_weight == 100.0


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
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli_entrypoints.py -q
```

Expected: failure because the scripts do not yet expose `build_parser()`.

- [ ] **Step 3: Implement parser builders and entry points**

For each script, move the existing `argparse.ArgumentParser` construction into
`build_parser()`, add:

```python
def main_cli():
    main(build_parser().parse_args())
```

and replace the `if __name__ == "__main__"` block with:

```python
if __name__ == "__main__":
    main_cli()
```

Add `pyproject.toml` console scripts:

```toml
[project.scripts]
mixed-dblocks-train = "llm_dblock_train:main_cli"
mixed-dblocks-quality = "qwen_quality_benchmark:main_cli"
mixed-dblocks-memory = "qwen_memory_benchmark:main_cli"
mixed-dblocks-tiny-quality = "llm_quality_benchmark:main_cli"
```

- [ ] **Step 4: Verify CLI tests pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli_entrypoints.py -q
```

Expected: all tests pass.

### Task 2: Publish Matrix Generator

**Files:**
- Create: `llm_dblocks/experiments.py`
- Modify: `llm_dblocks/__init__.py`
- Create: `tests/test_experiment_matrix.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing matrix tests**

Create `tests/test_experiment_matrix.py`:

```python
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
    assert any("--clean_lm_anchor_profile manual --clean_lm_weight 0" in cmd for cmd in commands)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_experiment_matrix.py -q
```

Expected: import failure for `llm_dblocks.experiments`.

- [ ] **Step 3: Implement matrix generation**

Create a lightweight module with:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    command: tuple[str, ...]
    output_json: str
    heavy: bool = False


def model_slug(model: str) -> str:
    ...


def publish_matrix(...):
    ...


def render_commands(specs):
    return [" ".join(spec.command) for spec in specs]
```

The generated quality ablations must include full AR, pure DBlocks,
full-sequence mixed, and auto-window mixed for each model/seed. Memory specs are
included only when `include_memory=True`.

- [ ] **Step 4: Verify matrix tests pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_experiment_matrix.py -q
```

Expected: all tests pass.

### Task 3: Result Summarization

**Files:**
- Create: `llm_dblocks/results.py`
- Create: `tests/test_results_summary.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write failing summary tests**

Create `tests/test_results_summary.py`:

```python
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

    assert "| File | Model | Seq | Mode | Profile | Before | After | Seconds | Peak GB |" in table
    assert "| x.json | m | 128 | dblock | auto_window | 2.0000 | 1.5000 | 12.00 |  |" in table
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_results_summary.py -q
```

Expected: import failure for `llm_dblocks.results`.

- [ ] **Step 3: Implement result summarization**

Implement `load_result_rows(paths)` and `markdown_table(rows)`. Add console
script:

```toml
mixed-dblocks-results = "llm_dblocks.results:main_cli"
```

- [ ] **Step 4: Verify summary tests pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_results_summary.py -q
```

Expected: all tests pass.

### Task 4: Production Readiness Documentation

**Files:**
- Create: `docs/production_readiness.md`
- Modify: `README.md`

- [ ] **Step 1: Document the production contract**

Create a document that states:

- supported platform: Apple Silicon + MLX for full-block bf16 training,
- default method: Windowed Mixed DiffusionBlocks with auto-window anchors,
- explicit ablations: full AR, pure DBlocks, full-anchor mixed,
- current caveat: the method is production-usable as a training path, but broad
  superiority over full AR/LoRA still requires the publish matrix,
- commands for dry-run matrix generation and result summarization.

- [ ] **Step 2: Verify docs reference shipped commands**

Run:

```bash
rg -n "mixed-dblocks-train|mixed-dblocks-quality|mixed-dblocks-memory|mixed-dblocks-results" README.md docs/production_readiness.md
```

Expected: all production commands are documented.

### Task 5: Final Verification And Push

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli_entrypoints.py tests/test_experiment_matrix.py tests/test_results_summary.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run lightweight CLI smoke checks**

Run:

```bash
.venv/bin/python -m llm_dblocks.results docs/results/wikitext2_qwen15_seq1024_dblock_auto_window_100.json
```

Expected: prints a markdown table with the auto-window row.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add .
git commit -m "Add production library workflow"
git push
```

Expected: branch is clean and pushed.
