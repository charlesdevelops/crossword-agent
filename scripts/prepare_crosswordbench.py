#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec>=2025.7",
#   "huggingface-hub>=0.34,<2",
#   "polars>=1.33,<2",
# ]
# ///

"""Download and normalize CrossWordBench without depending on the agent package."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET = "HINT-lab/CrossWordBench"
VARIANTS = {
    "english": {
        "7x7": "english/7x7-00000-of-00001.parquet",
        "14x14": "english/14x14-00000-of-00001.parquet",
    },
    "english_simple": {
        "7x7": "english_simple/7x7-00000-of-00001.parquet",
    },
}
ENTRY_KEY = re.compile(r"^(across|down)[ _-]?(\d+)$", re.IGNORECASE)
NUMBERED_CLUE = re.compile(r"^\s*(\d+)[.): -]+\s*(.+?)\s*$")
INLINE_CLUE = re.compile(r"^\s*(across|down)\s+(\d+)[.): -]+\s*(.+?)\s*$", re.IGNORECASE)
NUMBER_DIRECTION_CLUE = re.compile(
    r"^\s*(\d+)\s+(across|down)[.): -]+\s*(.+?)\s*$",
    re.IGNORECASE,
)
LETTER = re.compile(r"^[A-Za-z]$")
STARTED_AT = time.monotonic()


class ConversionError(ValueError):
    pass


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED_AT
    print(f"[+{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic local CrossWordBench JSON dataset.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="english_simple",
        help="CrossWordBench variant to prepare.",
    )
    parser.add_argument("--size", choices=("7x7", "14x14"), default="7x7")
    parser.add_argument(
        "--source",
        help="Local parquet path or hf:// URI. Defaults to the authenticated HF dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Normalized JSON destination. Defaults under data/crosswordbench/.",
    )
    parser.add_argument("--limit", type=int, default=20, help="0 exports every unique puzzle")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many puzzles after seeded shuffling to create disjoint artifacts.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--split", default="demo")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print parquet schema and one row without writing output.",
    )
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.size not in VARIANTS[args.variant]:
        parser.error(
            f"{args.variant} does not provide {args.size}; "
            f"available sizes: {', '.join(sorted(VARIANTS[args.variant]))}"
        )

    source = args.source or f"hf://datasets/{DATASET}/{VARIANTS[args.variant][args.size]}"
    output = args.output or Path(
        f"data/crosswordbench/{args.variant}-{args.size}.json"
    )
    log(f"Reading CrossWordBench {args.variant} {args.size} parquet")
    log(f"Source: {source}")
    frame = _read_parquet(source)
    log(f"Loaded {frame.height} rows and {len(frame.columns)} columns")
    log(f"Columns: {', '.join(frame.columns)}")
    if args.inspect:
        log("Printing schema and sanitized row 0")
        print(frame.schema)
        print(json.dumps(_json_safe(frame.row(0, named=True)), indent=2)[:20_000])
        return

    log("Normalizing rows")
    converted: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: set[str] = set()
    unreferenced_wordlist_entries = 0
    for index, row in enumerate(frame.iter_rows(named=True)):
        try:
            puzzle = normalize_row(row, source_index=index, size=args.size, split=args.split)
        except ConversionError as error:
            failures.append(f"row {index}: {error}")
            if len(failures) <= 3:
                log(f"Skipped row {index}: {error}")
        else:
            fingerprint = _puzzle_fingerprint(puzzle)
            if fingerprint not in seen:
                seen.add(fingerprint)
                converted.append(puzzle)
                unreferenced_wordlist_entries += int(
                    puzzle.get("source_unreferenced_wordlist_entries", 0)
                )
        if (index + 1) % 10 == 0 or index + 1 == frame.height:
            log(
                f"Processed {index + 1}/{frame.height}: "
                f"{len(converted)} unique, {len(failures)} skipped"
            )

    if not converted:
        schema = ", ".join(frame.columns)
        detail = "\n".join(failures[:3])
        diagnostic = output.with_suffix(".diagnostic.json")
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(
            json.dumps(
                {
                    "source": source,
                    "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
                    "row_0": _json_safe(frame.row(0, named=True)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"Wrote diagnostic sample to {diagnostic}")
        raise SystemExit(
            "No CrossWordBench rows could be normalized.\n"
            f"Columns: {schema}\n"
            f"{detail}\n"
            f"Inspect {diagnostic} or run again with --inspect."
        )

    converted.sort(key=lambda item: item["id"])
    random.Random(args.seed).shuffle(converted)
    converted = converted[args.offset :]
    if args.limit > 0:
        converted = converted[: args.limit]
    converted.sort(key=lambda item: item["id"])

    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing {len(converted)} puzzles to {output}")
    output.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source": source,
                "rows": frame.height,
                "unique_puzzles": len(seen),
                "exported": len(converted),
                "skipped_rows": len(failures),
                "source_unreferenced_wordlist_entries": unreferenced_wordlist_entries,
                "output": str(output),
                "seed": args.seed,
                "offset": args.offset,
            },
            indent=2,
        )
    )
    if failures:
        print("First skipped rows:", file=sys.stderr)
        print("\n".join(failures[:3]), file=sys.stderr)
    log("Preparation complete")


def _read_parquet(source: str):
    import polars as pl

    return pl.read_parquet(source)


def normalize_row(
    raw_row: Mapping[str, Any],
    *,
    source_index: int,
    size: str,
    split: str,
) -> dict[str, Any]:
    row = _decode(dict(raw_row))
    explicit = _normalize_crosswordbench_record(
        row,
        source_index=source_index,
        size=size,
        split=split,
    )
    if explicit is not None:
        return explicit

    clues, answers = _find_entries(row)
    grid = _find_grid(row)
    if grid is None:
        raise ConversionError("no grid-like value found")
    template, completed = _normalize_grid(grid)
    expected_dimension = int(size.split("x", 1)[0])
    if len(template) != expected_dimension or any(
        len(row) != expected_dimension for row in template
    ):
        raise ConversionError(
            f"expected a {size} grid, found {len(template)}x"
            f"{len(template[0]) if template else 0}"
        )
    if completed is None:
        if not answers:
            raise ConversionError(
                "grid has no solution letters and no reference answers were found"
            )
        completed = _fill_solution(template, answers)
    derived_numbers = _entry_numbers(template)
    if not derived_numbers:
        raise ConversionError(
            "grid contains no Across or Down entries; check the source block encoding"
        )
    if not any(direction == "across" for direction, _number in derived_numbers):
        raise ConversionError("grid contains no Across entries")
    if not any(direction == "down" for direction, _number in derived_numbers):
        raise ConversionError("grid contains no Down entries")
    _validate_reference_answers(template, completed, derived_numbers, answers)
    missing_clues = [
        f"{direction} {number}"
        for direction, number in derived_numbers
        if (direction, number) not in clues
    ]
    if missing_clues:
        raise ConversionError(f"missing clues for {', '.join(missing_clues[:5])}")

    across = {
        str(number): clues[("across", number)]
        for direction, number in derived_numbers
        if direction == "across"
    }
    down = {
        str(number): clues[("down", number)]
        for direction, number in derived_numbers
        if direction == "down"
    }
    canonical = {
        "solution": completed,
        "across": across,
        "down": down,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    source_id = row.get("id", source_index + 1)
    return {
        "id": f"crosswordbench-{size}-{digest}",
        "title": f"CrossWordBench {size} #{source_id}",
        "split": split,
        **canonical,
    }


def _normalize_crosswordbench_record(
    row: Mapping[str, Any],
    *,
    source_index: int,
    size: str,
    split: str,
) -> dict[str, Any] | None:
    puzzle_state = row.get("puzzle_state")
    reference_answer = row.get("reference_answer")
    if not isinstance(puzzle_state, Mapping) or not isinstance(
        reference_answer, Sequence
    ):
        return None
    grid_value = puzzle_state.get("grid")
    wordlist = puzzle_state.get("wordlist")
    if grid_value is None:
        raise ConversionError("puzzle_state is missing grid")
    if not isinstance(wordlist, Sequence):
        raise ConversionError("puzzle_state is missing wordlist")
    template, completed = _normalize_grid(grid_value)
    if completed is None:
        raise ConversionError("puzzle_state grid is not fully solved")
    expected_dimension = int(size.split("x", 1)[0])
    if len(template) != expected_dimension or any(
        len(grid_row) != expected_dimension for grid_row in template
    ):
        raise ConversionError(
            f"expected a {size} grid, found {len(template)}x"
            f"{len(template[0]) if template else 0}"
        )

    locations = [_parse_wordlist_item(item, index) for index, item in enumerate(wordlist)]
    unused_locations = set(range(len(locations)))
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(reference_answer):
        if not isinstance(entry, Mapping):
            raise ConversionError(f"reference_answer item {index} is not an object")
        direction_value = entry.get("direction")
        match = ENTRY_KEY.fullmatch(str(direction_value).strip())
        if not match:
            raise ConversionError(
                f"reference_answer item {index} has invalid direction {direction_value!r}"
            )
        key = (match.group(1).lower(), int(match.group(2)))
        clue = entry.get("clue")
        answer = entry.get("answer")
        if not isinstance(clue, str) or not clue.strip():
            raise ConversionError(f"reference_answer item {index} has no clue")
        if not isinstance(answer, str) or not answer.strip():
            raise ConversionError(f"reference_answer item {index} has no answer")
        cleaned_answer = _clean_answer(answer)
        direction = key[0]
        location_index = _match_wordlist_location(
            locations,
            unused_locations,
            answer=cleaned_answer,
            clue=clue,
            direction=direction,
        )
        unused_locations.remove(location_index)
        location = locations[location_index]
        cells = _cells_for_entry(
            row=location["row"],
            col=location["col"],
            direction=direction,
            length=len(cleaned_answer),
        )
        if any(
            cell_row < 0
            or cell_row >= len(completed)
            or cell_col < 0
            or cell_col >= len(completed[0])
            for cell_row, cell_col in cells
        ):
            raise ConversionError(f"{direction} {key[1]} extends beyond the grid")
        grid_answer = "".join(completed[cell_row][cell_col] for cell_row, cell_col in cells)
        if grid_answer != cleaned_answer:
            raise ConversionError(
                f"{direction} {key[1]} is {grid_answer!r} in puzzle_state "
                f"but {cleaned_answer!r} in reference_answer"
            )
        entries.append(
            {
                "id": f"{key[1]}{'A' if direction == 'across' else 'D'}",
                "number": key[1],
                "direction": direction,
                "clue": clue.strip(),
                "answer": cleaned_answer,
                "start": [location["row"], location["col"]],
            }
        )

    entries.sort(key=lambda item: (int(item["number"]), str(item["direction"])))
    canonical = {"solution": completed, "entries": entries}
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    source_id = row.get("id", source_index + 1)
    return {
        "id": f"crosswordbench-{size}-{digest}",
        "title": f"CrossWordBench {size} #{source_id}",
        "split": split,
        "source_unreferenced_wordlist_entries": len(unused_locations),
        **canonical,
    }


def _parse_wordlist_item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 5:
        raise ConversionError(f"puzzle_state wordlist item {index} is malformed")
    answer, clue, row, col, direction_value = item[:5]
    if direction_value in (0, "0", "across"):
        direction = "across"
    elif direction_value in (1, "1", "down"):
        direction = "down"
    else:
        raise ConversionError(
            f"puzzle_state wordlist item {index} has invalid direction {direction_value!r}"
        )
    try:
        parsed_row = int(row)
        parsed_col = int(col)
    except (TypeError, ValueError) as error:
        raise ConversionError(
            f"puzzle_state wordlist item {index} has an invalid start"
        ) from error
    return {
        "answer": _clean_answer(str(answer)),
        "clue": str(clue).strip(),
        "row": parsed_row,
        "col": parsed_col,
        "direction": direction,
    }


def _match_wordlist_location(
    locations: Sequence[Mapping[str, Any]],
    unused: set[int],
    *,
    answer: str,
    clue: str,
    direction: str,
) -> int:
    exact = [
        index
        for index in unused
        if locations[index]["answer"] == answer
        and locations[index]["direction"] == direction
        and _text_key(str(locations[index]["clue"])) == _text_key(clue)
    ]
    if len(exact) == 1:
        return exact[0]
    loose = [
        index
        for index in unused
        if locations[index]["answer"] == answer
        and locations[index]["direction"] == direction
    ]
    if len(loose) == 1:
        return loose[0]
    if not exact and not loose:
        raise ConversionError(
            f"no puzzle_state location found for {direction} answer {answer!r}"
        )
    raise ConversionError(
        f"ambiguous puzzle_state locations for {direction} answer {answer!r}"
    )


def _cells_for_entry(
    *,
    row: int,
    col: int,
    direction: str,
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = (0, 1) if direction == "across" else (1, 0)
    return [(row + offset * dr, col + offset * dc) for offset in range(length)]


def _text_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_reference_answers(
    template: Sequence[str],
    completed: Sequence[str],
    derived_numbers: Sequence[tuple[str, int]],
    answers: Mapping[tuple[str, int], str],
) -> None:
    if not answers:
        return
    cells_by_entry = _entry_cells(template)
    derived = set(derived_numbers)
    missing_answers = sorted(derived - set(answers))
    if missing_answers:
        direction, number = missing_answers[0]
        raise ConversionError(f"missing reference answer for {direction} {number}")
    extra_answers = sorted(set(answers) - derived)
    if extra_answers:
        direction, number = extra_answers[0]
        raise ConversionError(f"reference answer does not map to grid: {direction} {number}")
    for key, cells in cells_by_entry.items():
        grid_answer = "".join(completed[row][col] for row, col in cells)
        if grid_answer != answers[key]:
            raise ConversionError(
                f"{key[0]} {key[1]} is {grid_answer!r} in puzzle_state "
                f"but {answers[key]!r} in reference_answer"
            )


def _find_entries(
    row: Mapping[str, Any],
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    clues: dict[tuple[str, int], str] = {}
    answers: dict[tuple[str, int], str] = {}
    for path, value in _walk(row):
        path_key = ".".join(path).lower()
        direction_hint = next(
            (direction for direction in ("across", "down") if direction in path_key),
            None,
        )
        if isinstance(value, Mapping):
            direct_direction = _first_string(value, ("direction", "orientation"))
            direct_number = value.get("number", value.get("index"))
            if (
                direct_direction
                and direct_direction.lower() in {"across", "down"}
                and str(direct_number).isdigit()
            ):
                _record_entry(
                    clues,
                    answers,
                    direct_direction.lower(),
                    int(str(direct_number)),
                    value,
                    path_key,
                )
            for key, entry_value in value.items():
                match = ENTRY_KEY.fullmatch(str(key).strip())
                if match:
                    _record_entry(
                        clues,
                        answers,
                        match.group(1).lower(),
                        int(match.group(2)),
                        entry_value,
                        path_key,
                    )
                    continue
                if direction_hint and str(key).strip().isdigit():
                    _record_entry(
                        clues,
                        answers,
                        direction_hint,
                        int(str(key).strip()),
                        entry_value,
                        path_key,
                    )
        if isinstance(value, str):
            if any(
                marker in path_key
                for marker in ("prompt", "question", "clue", "text", "state")
            ):
                clues.update(_parse_prompt_clues(value))
            if any(
                marker in path_key
                for marker in ("answer", "solution", "gold", "reference")
            ):
                answers.update(_parse_reference_answers(value))
    return clues, answers


def _record_entry(
    clues: dict[tuple[str, int], str],
    answers: dict[tuple[str, int], str],
    direction: str,
    number: int,
    value: Any,
    context: str,
) -> None:
    key = (direction, number)
    if isinstance(value, Mapping):
        clue = _first_string(value, ("clue", "question", "definition", "hint"))
        answer = _first_string(value, ("answer", "word", "solution", "gold", "reference"))
        if clue:
            clues.setdefault(key, clue)
        if answer:
            answers.setdefault(key, _clean_answer(answer))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        strings = [item for item in value if isinstance(item, str)]
        if strings:
            answer_candidates = [item for item in strings if _looks_like_answer(item)]
            clue_candidates = [item for item in strings if item not in answer_candidates]
            if answer_candidates:
                answers.setdefault(key, _clean_answer(answer_candidates[-1]))
            if clue_candidates:
                clues.setdefault(key, clue_candidates[0].strip())
        return
    if not isinstance(value, str):
        return
    if any(marker in context for marker in ("answer", "solution", "gold", "reference")):
        answers.setdefault(key, _clean_answer(value))
    elif any(marker in context for marker in ("clue", "question", "prompt")):
        clues.setdefault(key, value.strip())
    elif _looks_like_answer(value):
        answers.setdefault(key, _clean_answer(value))
    else:
        clues.setdefault(key, value.strip())


def _parse_prompt_clues(text: str) -> dict[tuple[str, int], str]:
    clues: dict[tuple[str, int], str] = {}
    direction: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = line.rstrip(":").lower()
        if heading in {"across", "across clues"}:
            direction = "across"
            continue
        if heading in {"down", "down clues"}:
            direction = "down"
            continue
        inline = INLINE_CLUE.match(line)
        if inline:
            clues[(inline.group(1).lower(), int(inline.group(2)))] = inline.group(3)
            continue
        number_direction = NUMBER_DIRECTION_CLUE.match(line)
        if number_direction:
            clues[
                (number_direction.group(2).lower(), int(number_direction.group(1)))
            ] = number_direction.group(3)
            continue
        numbered = NUMBERED_CLUE.match(line)
        if direction and numbered:
            clues[(direction, int(numbered.group(1)))] = numbered.group(2)
    return clues


def _parse_reference_answers(text: str) -> dict[tuple[str, int], str]:
    answers: dict[tuple[str, int], str] = {}
    direction: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = line.rstrip(":").lower()
        if heading in {"across", "across answers"}:
            direction = "across"
            continue
        if heading in {"down", "down answers"}:
            direction = "down"
            continue
        inline = INLINE_CLUE.match(line)
        if inline and _looks_like_answer(inline.group(3)):
            answers[
                (inline.group(1).lower(), int(inline.group(2)))
            ] = _clean_answer(inline.group(3))
            continue
        number_direction = NUMBER_DIRECTION_CLUE.match(line)
        if number_direction and _looks_like_answer(number_direction.group(3)):
            answers[
                (number_direction.group(2).lower(), int(number_direction.group(1)))
            ] = _clean_answer(number_direction.group(3))
            continue
        numbered = NUMBERED_CLUE.match(line)
        if direction and numbered and _looks_like_answer(numbered.group(2)):
            answers[(direction, int(numbered.group(1)))] = _clean_answer(
                numbered.group(2)
            )
    return answers


def _find_grid(row: Mapping[str, Any]) -> Any | None:
    preferred = (
        "solution_grid",
        "answer_grid",
        "completed_grid",
        "filled_grid",
        "grid",
        "crossword",
        "board",
    )
    candidates = list(_walk(row))
    for name in preferred:
        for path, value in candidates:
            if path and path[-1].lower() == name and _is_grid_like(value):
                return value
    for _path, value in candidates:
        if _is_grid_like(value):
            return value
    return None


def _normalize_grid(value: Any) -> tuple[list[str], list[str] | None]:
    rows = _coerce_rows(value)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ConversionError("grid rows are missing or have inconsistent widths")
    normalized: list[str] = []
    has_letters = False
    for row in rows:
        output = ""
        for cell in row:
            char = _cell_character(cell)
            output += char
            has_letters = has_letters or char.isalpha()
        normalized.append(output)
    template = ["".join("#" if char == "#" else "." for char in row) for row in normalized]
    if has_letters and all(char == "#" or char.isalpha() for row in normalized for char in row):
        return template, normalized
    return template, None


def _coerce_rows(value: Any) -> list[list[Any]]:
    if isinstance(value, str):
        decoded = _maybe_json(value)
        if decoded is not value:
            return _coerce_rows(decoded)
        lines = [
            line.replace(" ", "")
            for line in value.splitlines()
            if re.fullmatch(r"\s*[A-Za-z.#?01_-]+\s*", line)
        ]
        return [list(line) for line in lines]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    if value and all(isinstance(row, str) for row in value):
        return [list(str(row).replace(" ", "")) for row in value]
    if value and all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray))
        for row in value
    ):
        return [list(row) for row in value]
    return []


def _cell_character(cell: Any) -> str:
    if isinstance(cell, bool):
        return "." if cell else "#"
    if isinstance(cell, int | float):
        return "." if int(cell) == 1 else "#"
    if isinstance(cell, Mapping):
        if cell.get("block") or cell.get("blocked"):
            return "#"
        for key in ("letter", "char", "answer", "value"):
            if key in cell:
                return _cell_character(cell[key])
        return "."
    if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes, bytearray)):
        letters = [item for item in cell if isinstance(item, str) and LETTER.fullmatch(item)]
        if letters:
            return letters[-1].upper()
        binary = [
            item
            for item in cell
            if isinstance(item, (bool, int, float)) and int(item) in {0, 1}
        ]
        return _cell_character(binary[-1]) if binary else "."
    text = str(cell).strip()
    if text in {"#", "-", "0"}:
        return "#"
    if text == "1":
        return "."
    if LETTER.fullmatch(text):
        return text.upper()
    return "."


def _fill_solution(
    template: Sequence[str],
    answers: Mapping[tuple[str, int], str],
) -> list[str]:
    grid = [list(row) for row in template]
    entries = _entry_cells(template)
    for key, cells in entries.items():
        answer = answers.get(key)
        if answer is None:
            raise ConversionError(f"missing reference answer for {key[0]} {key[1]}")
        if len(answer) != len(cells):
            raise ConversionError(
                f"{key[0]} {key[1]} has length {len(cells)} but answer {answer!r} "
                f"has length {len(answer)}"
            )
        for (row, col), letter in zip(cells, answer, strict=True):
            current = grid[row][col]
            if current not in {".", letter}:
                raise ConversionError(f"conflicting answers at row {row}, column {col}")
            grid[row][col] = letter
    if any(char == "." for row in grid for char in row):
        raise ConversionError("reference answers did not fill every open grid cell")
    return ["".join(row) for row in grid]


def _entry_numbers(template: Sequence[str]) -> list[tuple[str, int]]:
    return list(_entry_cells(template))


def _entry_cells(
    template: Sequence[str],
) -> dict[tuple[str, int], list[tuple[int, int]]]:
    height = len(template)
    width = len(template[0])
    entries: dict[tuple[str, int], list[tuple[int, int]]] = {}
    number = 1

    def collect(row: int, col: int, dr: int, dc: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        while (
            0 <= row < height
            and 0 <= col < width
            and template[row][col] != "#"
        ):
            cells.append((row, col))
            row += dr
            col += dc
        return cells

    for row in range(height):
        for col in range(width):
            if template[row][col] == "#":
                continue
            starts_across = col == 0 or template[row][col - 1] == "#"
            starts_down = row == 0 or template[row - 1][col] == "#"
            across_cells = collect(row, col, 0, 1) if starts_across else []
            down_cells = collect(row, col, 1, 0) if starts_down else []
            starts_across = len(across_cells) >= 2
            starts_down = len(down_cells) >= 2
            if starts_across:
                entries[("across", number)] = across_cells
            if starts_down:
                entries[("down", number)] = down_cells
            if starts_across or starts_down:
                number += 1
    return entries


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _decode(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_decode(child) for child in value]
    if isinstance(value, str):
        decoded = _maybe_json(value)
        return _decode(decoded) if decoded is not value else value
    return value


def _maybe_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _is_grid_like(value: Any) -> bool:
    rows = _coerce_rows(value)
    return len(rows) >= 3 and len({len(row) for row in rows}) == 1 and len(rows[0]) >= 3


def _first_string(value: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _looks_like_answer(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+", value.strip()))


def _clean_answer(value: str) -> str:
    answer = re.sub(r"[^A-Za-z]", "", value).upper()
    if not answer:
        raise ConversionError(f"invalid reference answer {value!r}")
    return answer


def _puzzle_fingerprint(puzzle: Mapping[str, Any]) -> str:
    payload = {
        "solution": puzzle["solution"],
    }
    if "entries" in puzzle:
        payload["entries"] = puzzle["entries"]
    else:
        payload["across"] = puzzle["across"]
        payload["down"] = puzzle["down"]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


if __name__ == "__main__":
    main()
