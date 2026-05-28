from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def write_split(out_dir: Path, split: str, filename: str) -> None:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    lines = [row["text"] for row in dataset if row["text"].strip()]
    path = out_dir / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path} chars={path.stat().st_size} lines={len(lines)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local WikiText-2 train/validation text files."
    )
    parser.add_argument("--out_dir", default="corpora/wikitext2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(out_dir, "train", "train.txt")
    write_split(out_dir, "validation", "validation.txt")


if __name__ == "__main__":
    main()
