from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ResultRow = dict[str, Any]


def _metric(result: dict[str, Any], preferred: str, fallback: str) -> float | None:
    value = result.get(preferred)
    if value is None:
        value = result.get(fallback)
    return value


def _row_from_mode(path: Path, payload: dict[str, Any], mode: str, result: dict[str, Any]):
    return {
        "file": path.name,
        "model": payload.get("model"),
        "seq_len": payload.get("seq_len"),
        "mode": mode,
        "profile": result.get("clean_lm_anchor_profile")
        or payload.get("clean_lm_anchor_profile")
        or "",
        "before": _metric(
            result,
            "response_ce_before",
            "next_token_ce_before",
        )
        or result.get("before"),
        "after": _metric(
            result,
            "response_ce_after",
            "next_token_ce_after",
        )
        or result.get("after"),
        "seconds": result.get("seconds"),
        "peak_gb": result.get("peak_gb") or payload.get("peak_gb"),
    }


def load_result_rows(paths: Iterable[str | Path]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        modes = payload.get("modes")
        if isinstance(modes, dict):
            for mode, result in modes.items():
                if isinstance(result, dict) and result.get("ok", True):
                    rows.append(_row_from_mode(path, payload, mode, result))
            continue
        rows.append(_row_from_mode(path, payload, payload.get("mode", ""), payload))
    return rows


def _format_float(value: Any, precision: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def markdown_table(rows: list[ResultRow]) -> str:
    lines = [
        "| File | Model | Seq | Mode | Profile | Before | After | Seconds | Peak GB |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| {row.get('file', '')}",
                    str(row.get("model") or ""),
                    str(row.get("seq_len") or ""),
                    str(row.get("mode") or ""),
                    str(row.get("profile") or ""),
                    _format_float(row.get("before"), 4),
                    _format_float(row.get("after"), 4),
                    _format_float(row.get("seconds"), 2),
                    f"{_format_float(row.get('peak_gb'), 2)} |",
                ]
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Mixed DiffusionBlocks JSON results as markdown"
    )
    parser.add_argument("paths", nargs="+")
    return parser


def main_cli(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    print(markdown_table(load_result_rows(args.paths)))


if __name__ == "__main__":
    main_cli()
