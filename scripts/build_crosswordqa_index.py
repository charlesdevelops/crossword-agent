#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec>=2025.7",
#   "huggingface-hub>=0.34,<2",
#   "polars>=1.33,<2",
# ]
# ///

"""Build a compact clue-answer index from the human CrosswordQA corpus."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

DATASET_ROOT = "hf://datasets/albertxu/CrosswordQA"
SPLITS = {"train": "train.csv", "validation": "valid.csv"}
DEFAULT_LIMIT = 500_000
WHITESPACE = re.compile(r"\s+")


def main() -> None:
    args = _parse_args()
    source = args.source or f"{DATASET_ROOT}/{SPLITS[args.split]}"
    output = args.output or Path(
        f"data/crosswordqa/{args.split}-clue-index.json"
    )

    import polars as pl

    print(f"Reading {source}", flush=True)
    frame = pl.read_csv(source)
    required = {"clue", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"CrosswordQA is missing required columns {sorted(missing)}; "
            f"found {frame.columns}"
        )

    frame = (
        frame.select(
            pl.col("clue").cast(pl.Utf8).alias("clue"),
            pl.col("answer").cast(pl.Utf8).alias("answer"),
        )
        .with_columns(
            pl.col("clue")
            .str.replace_all(WHITESPACE.pattern, " ")
            .str.strip_chars()
            .alias("clue"),
            pl.col("answer").str.strip_chars().alias("answer"),
        )
        .filter(
            (pl.col("clue").str.len_chars() > 0)
            & (pl.col("answer").str.len_chars() > 0)
        )
        .unique(maintain_order=True)
    )
    unique_count = frame.height
    if args.limit > 0 and frame.height > args.limit:
        frame = frame.sample(n=args.limit, seed=args.seed)

    examples = [
        {"clue": clue, "answer": answer}
        for clue, answer in frame.iter_rows()
    ]
    examples.sort(key=lambda item: (item["answer"], item["clue"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source": source,
                "allowed_splits": [args.split],
                "source_rows": unique_count,
                "indexed_examples": len(examples),
                "sampling_limit": args.limit,
                "sampling_seed": args.seed,
                "examples": examples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": source,
                "unique_examples": unique_count,
                "indexed_examples": len(examples),
                "output": str(output),
                "seed": args.seed,
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25 clue-answer index from CrosswordQA."
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLITS),
        default="train",
        help="Use train for retrieval or validation for a held-out diagnostic index.",
    )
    parser.add_argument("--source", help="Local CSV path or hf:// URI.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum unique clue-answer pairs to index; 0 keeps all pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


if __name__ == "__main__":
    main()
