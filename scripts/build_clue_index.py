#!/usr/bin/env python3
"""Build a local clue-answer retrieval index from approved training splits."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from crossword_agent.puzzles import load_normalized_records

DEFAULT_SPLITS = ("development", "dev", "train", "training")


def main() -> None:
    args = _parse_args()
    records = load_normalized_records(args.dataset)
    allowed_splits = {value.casefold() for value in args.allow_split}
    excluded_ids: set[str] = set()
    excluded_pairs: set[tuple[str, str]] = set()
    for path in args.exclude_dataset:
        for record in load_normalized_records(path):
            excluded_ids.add(record.puzzle.id)
            excluded_pairs.update(_examples_for_record(record))

    eligible = [
        record
        for record in records
        if record.split.casefold() in allowed_splits
    ]
    if not eligible:
        available = sorted({record.split for record in records})
        raise SystemExit(
            "No records use an approved training split. "
            f"Allowed: {sorted(allowed_splits)}; found: {available}. "
            "Prepare a separate development artifact with "
            "`scripts/prepare_crosswordbench.py --split development`."
        )

    examples: set[tuple[str, str]] = set()
    skipped_overlap = 0
    for record in eligible:
        if record.puzzle.id in excluded_ids:
            skipped_overlap += len(record.puzzle.entries)
            continue
        for clue, answer in _examples_for_record(record):
            if (clue, answer) in excluded_pairs:
                skipped_overlap += 1
                continue
            examples.add((clue, answer))
    if not examples:
        raise SystemExit("No non-overlapping clue-answer examples remain for indexing.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(args.dataset.resolve()),
        "allowed_splits": sorted(allowed_splits),
        "source_puzzles": len(eligible),
        "excluded_datasets": [str(path.resolve()) for path in args.exclude_dataset],
        "skipped_overlap": skipped_overlap,
        "examples": [
            {"clue": clue, "answer": answer}
            for clue, answer in sorted(examples, key=lambda item: (item[1], item[0]))
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "puzzles": len(eligible),
                "examples": len(examples),
                "skipped_overlap": skipped_overlap,
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a BM25 clue-answer index. Demo/evaluation records are rejected by default "
            "to prevent benchmark leakage."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Normalized training/development JSON produced by the preparation script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crosswordbench/clue-index.json"),
    )
    parser.add_argument(
        "--allow-split",
        action="append",
        default=None,
        help="Allowed source split; repeat as needed. Defaults to train/development aliases.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        type=Path,
        default=[],
        help="Evaluation JSON whose puzzle IDs and exact clue-answer pairs must be excluded.",
    )
    args = parser.parse_args()
    args.allow_split = args.allow_split or list(DEFAULT_SPLITS)
    for path in [args.dataset, *args.exclude_dataset]:
        if not path.is_file():
            parser.error(f"Dataset does not exist: {path}")
    return args


def _examples_for_record(record) -> set[tuple[str, str]]:
    if record.solution is None:
        return set()
    return {
        (
            " ".join(entry.clue.split()),
            "".join(record.solution[row][col] for row, col in entry.cells),
        )
        for entry in record.puzzle.entries
    }


if __name__ == "__main__":
    main()
