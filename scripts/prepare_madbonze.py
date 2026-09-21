#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.33,<2",
# ]
# ///

"""Normalize MadBonze/puzzles crossword rows for the crossword agent."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DATASET_ROOT = "hf://datasets/MadBonze/puzzles"
SPLITS = {"train": "train.csv", "test": "test.csv"}
GRID_MARKERS = re.compile(r"[■_]")
CLUE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(\d+)([ad])[A-Za-z]*\s+\((across|down),\s*(\d+)\):\s*"
    r"(.*?)\s*\[Row\s+(\d+),\s*Col\s+(\d+)\]"
)
ANSWER_KEY = re.compile(r"^(\d+)([ad])", re.IGNORECASE)


class ConversionError(ValueError):
    pass


def main() -> None:
    args = _parse_args()
    source = args.source or f"{DATASET_ROOT}/{SPLITS[args.split]}"
    output = args.output or Path(f"data/madbonze/{args.split}.json")

    import polars as pl

    print(f"Reading {source}", flush=True)
    frame = pl.read_csv(source)
    required = {"puzzle", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"MadBonze is missing required columns {sorted(missing)}; "
            f"found {frame.columns}"
        )
    puzzle_text = pl.col("puzzle").cast(pl.String)
    has_grid_sections = puzzle_text.str.contains(
        "Crossword Grid :", literal=True
    ) & puzzle_text.str.contains(
        "Word Positions and Clues:", literal=True
    )
    named_as_crossword = (
        pl.col("name").cast(pl.String).str.to_lowercase() == "crossword"
        if "name" in frame.columns
        else pl.lit(False)
    )
    crossword_rows = frame.filter(named_as_crossword | has_grid_sections)
    print(
        f"Loaded {frame.height} rows; {crossword_rows.height} crossword rows",
        flush=True,
    )

    converted: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, row in enumerate(crossword_rows.iter_rows(named=True)):
        try:
            converted.append(
                normalize_row(
                    row,
                    source_index=index,
                    split=args.split,
                )
            )
        except ConversionError as error:
            failures.append(f"row {index}: {error}")
            if len(failures) <= 5:
                print(f"Skipped row {index}: {error}", file=sys.stderr)
        if (index + 1) % 1000 == 0 or index + 1 == crossword_rows.height:
            print(
                f"Processed {index + 1}/{crossword_rows.height}: "
                f"{len(converted)} converted, {len(failures)} skipped",
                flush=True,
            )

    if args.limit > 0:
        converted = converted[: args.limit]
    if not converted:
        raise SystemExit("No MadBonze crossword rows could be normalized.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source": source,
                "rows": frame.height,
                "crossword_rows": crossword_rows.height,
                "exported": len(converted),
                "skipped": len(failures),
                "output": str(output),
            },
            indent=2,
        )
    )
    if failures:
        print("First skipped rows:", file=sys.stderr)
        print("\n".join(failures[:5]), file=sys.stderr)


def normalize_row(
    row: Mapping[str, Any],
    *,
    source_index: int,
    split: str,
) -> dict[str, Any]:
    puzzle_text = str(row.get("puzzle") or "")
    if not puzzle_text:
        raise ConversionError("missing puzzle text")
    grid = _parse_grid(puzzle_text)
    clues = _parse_clues(puzzle_text)
    answers = _parse_answers(row.get("answer"))
    if not clues:
        raise ConversionError("no positioned clues found")
    if not answers:
        raise ConversionError("no answer map found")

    solution = [list(line) for line in grid]
    entries: list[dict[str, Any]] = []
    occupied: dict[tuple[int, int], tuple[str, str]] = {}
    for number, direction, length, clue, start_row, start_col in clues:
        key = f"{number}{direction}"
        answer = _answer_for(answers, number, direction)
        if answer is None:
            raise ConversionError(f"missing answer for {key}")
        cleaned = _clean_answer(answer)
        if len(cleaned) != length:
            raise ConversionError(
                f"{key} answer length {len(cleaned)} does not match clue length {length}"
            )
        dr, dc = (0, 1) if direction == "a" else (1, 0)
        cells = [
            (start_row + offset * dr, start_col + offset * dc)
            for offset in range(length)
        ]
        for row_index, col_index, letter in zip(
            (cell[0] for cell in cells),
            (cell[1] for cell in cells),
            cleaned,
            strict=True,
        ):
            if not (
                0 <= row_index < len(solution)
                and 0 <= col_index < len(solution[0])
            ):
                raise ConversionError(f"{key} extends beyond the grid")
            if solution[row_index][col_index] == "#":
                raise ConversionError(f"{key} crosses a block")
            existing = solution[row_index][col_index]
            if existing not in {".", letter}:
                owner, owner_letter = occupied[(row_index, col_index)]
                raise ConversionError(
                    f"{key} conflicts at {(row_index, col_index)}: "
                    f"{owner} has {owner_letter}, {key} has {letter}"
                )
            solution[row_index][col_index] = letter
            occupied[(row_index, col_index)] = (key, letter)
        entries.append(
            {
                "id": key.upper(),
                "number": number,
                "direction": "across" if direction == "a" else "down",
                "clue": clue,
                "answer": cleaned,
                "start": [start_row, start_col],
            }
        )

    if any(cell == "." for line in solution for cell in line):
        raise ConversionError("some open grid cells were not covered by clues")
    title = str(row.get("name") or f"MadBonze crossword {source_index}")
    row_id = str(row.get("Unnamed: 0") or source_index)
    digest = hashlib.sha256(
        json.dumps(
            {"solution": solution, "entries": entries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:12]
    return {
        "id": f"madbonze-{row_id}-{digest}",
        "title": title,
        "split": split,
        "solution": ["".join(line) for line in solution],
        "entries": entries,
    }


def _parse_grid(text: str) -> list[str]:
    start = text.find("Crossword Grid :")
    end = text.find("Word Positions and Clues:")
    if start < 0 or end < 0 or end <= start:
        raise ConversionError("missing crossword grid or clue section")
    markers = GRID_MARKERS.findall(text[start:end])
    side = math.isqrt(len(markers))
    if side * side != len(markers):
        raise ConversionError(f"grid has {len(markers)} cells, not a square")
    return [
        "".join("#" if marker == "■" else "." for marker in markers[offset : offset + side])
        for offset in range(0, len(markers), side)
    ]


def _parse_clues(text: str) -> list[tuple[int, str, int, str, int, int]]:
    clues = []
    for match in CLUE.finditer(text):
        number, direction, _direction_name, length, clue, row, col = match.groups()
        cleaned_clue = " ".join(clue.split())
        if not cleaned_clue:
            continue
        clues.append(
            (
                int(number),
                direction.lower(),
                int(length),
                cleaned_clue,
                int(row),
                int(col),
            )
        )
    return clues


def _parse_answers(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key).lower(): str(answer) for key, answer in value.items()}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        raise ConversionError("answer column is not a dictionary") from None
    if not isinstance(parsed, Mapping):
        raise ConversionError("answer column is not a dictionary")
    return {str(key).lower(): str(answer) for key, answer in parsed.items()}


def _answer_for(answers: Mapping[str, str], number: int, direction: str) -> str | None:
    exact = answers.get(f"{number}{direction}")
    if exact is not None:
        return exact
    for key, answer in answers.items():
        match = ANSWER_KEY.match(key)
        if match and int(match.group(1)) == number and match.group(2).lower() == direction:
            return answer
    return None


def _clean_answer(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.upper())
    return "".join(
        char
        for char in decomposed
        if char.isalpha() and unicodedata.category(char) != "Mn"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MadBonze human crossword grids for the agent UI."
    )
    parser.add_argument("--split", choices=tuple(SPLITS), default="train")
    parser.add_argument("--source", help="Local CSV path or hf:// URI.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to data/madbonze/<split>.json.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


if __name__ == "__main__":
    main()
