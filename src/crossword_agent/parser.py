from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

from crossword_agent.models import (
    Cell,
    Direction,
    Entry,
    Intersection,
    PuzzleDefinition,
    PuzzleRecord,
)

_CLUE_PATTERN = re.compile(r"^\s*(\d+)[.)]?\s+(.+?)\s*$")
_GRID_PATTERN = re.compile(r"^[A-Za-z.#?_-]+$")


class PuzzleFormatError(ValueError):
    pass


def build_puzzle_record(
    *,
    puzzle_id: str,
    title: str,
    solution_rows: Sequence[str],
    across_clues: Mapping[int | str, str],
    down_clues: Mapping[int | str, str],
    split: str = "demo",
) -> PuzzleRecord:
    solution = tuple(row.upper().replace("_", ".").replace("?", ".") for row in solution_rows)
    _validate_grid(solution)
    template = tuple("".join("#" if char == "#" else "." for char in row) for row in solution)
    entries = _extract_entries(template, across_clues, down_clues)
    if not entries:
        raise PuzzleFormatError("Puzzle must contain at least one Across or Down entry")
    if not any(entry.direction is Direction.ACROSS for entry in entries):
        raise PuzzleFormatError("Puzzle must contain at least one Across entry")
    if not any(entry.direction is Direction.DOWN for entry in entries):
        raise PuzzleFormatError("Puzzle must contain at least one Down entry")
    intersections = _build_intersections(entries)
    puzzle = PuzzleDefinition(
        id=puzzle_id,
        title=title,
        width=len(template[0]),
        height=len(template),
        template=template,
        entries=tuple(entries),
        intersections=tuple(intersections),
    )
    has_solution = all(char in "#ABCDEFGHIJKLMNOPQRSTUVWXYZ" for row in solution for char in row)
    return PuzzleRecord(puzzle=puzzle, solution=solution if has_solution else None, split=split)


def build_explicit_puzzle_record(
    *,
    puzzle_id: str,
    title: str,
    solution_rows: Sequence[str],
    entry_records: Sequence[Mapping[str, object]],
    split: str = "external",
) -> PuzzleRecord:
    """Build a puzzle from source-provided entry locations without renumbering it."""

    solution = tuple(row.upper().replace("-", "#") for row in solution_rows)
    _validate_grid(solution)
    height = len(solution)
    width = len(solution[0])
    entries: list[Entry] = []
    seen_ids: set[str] = set()
    covered_cells: set[Cell] = set()
    for index, item in enumerate(entry_records):
        try:
            direction = Direction(str(item["direction"]).lower())
            number = int(item["number"])
            clue = str(item["clue"]).strip()
            answer = re.sub(r"[^A-Za-z]", "", str(item["answer"])).upper()
            start_value = item["start"]
            if not isinstance(start_value, Sequence) or isinstance(start_value, str):
                raise TypeError("start must be a row/column pair")
            row, col = (int(value) for value in start_value)
        except (KeyError, TypeError, ValueError) as error:
            raise PuzzleFormatError(f"Invalid explicit entry at index {index}: {error}") from error
        entry_id = str(
            item.get(
                "id",
                f"{number}{'A' if direction is Direction.ACROSS else 'D'}",
            )
        )
        if not clue:
            raise PuzzleFormatError(f"Entry {entry_id} has no clue")
        if not answer:
            raise PuzzleFormatError(f"Entry {entry_id} has no answer")
        if entry_id in seen_ids:
            raise PuzzleFormatError(f"Duplicate entry ID: {entry_id}")
        seen_ids.add(entry_id)
        dr, dc = (0, 1) if direction is Direction.ACROSS else (1, 0)
        cells = tuple((row + offset * dr, col + offset * dc) for offset in range(len(answer)))
        if any(
            cell_row < 0
            or cell_row >= height
            or cell_col < 0
            or cell_col >= width
            for cell_row, cell_col in cells
        ):
            raise PuzzleFormatError(f"Entry {entry_id} extends beyond the grid")
        grid_answer = "".join(solution[cell_row][cell_col] for cell_row, cell_col in cells)
        if grid_answer != answer:
            raise PuzzleFormatError(
                f"Entry {entry_id} is {grid_answer!r} in the grid but {answer!r} in the record"
            )
        entries.append(
            Entry(
                id=entry_id,
                number=number,
                direction=direction,
                clue=clue,
                cells=cells,
            )
        )
        covered_cells.update(cells)

    if not entries:
        raise PuzzleFormatError("Puzzle must contain at least one entry")
    masked_solution = tuple(
        "".join(
            solution[row][col] if (row, col) in covered_cells else "#"
            for col in range(width)
        )
        for row in range(height)
    )
    template = tuple(
        "".join("#" if char == "#" else "." for char in row)
        for row in masked_solution
    )
    puzzle = PuzzleDefinition(
        id=puzzle_id,
        title=title,
        width=width,
        height=height,
        template=template,
        entries=tuple(entries),
        intersections=tuple(_build_intersections(entries)),
    )
    return PuzzleRecord(puzzle=puzzle, solution=masked_solution, split=split)


def parse_crosswordbench_text(
    text: str,
    *,
    puzzle_id: str,
    title: str | None = None,
    split: str = "external",
) -> PuzzleRecord:
    """Parse a normalized CrossWordBench-style text puzzle.

    Accepted sections are GRID, ACROSS and DOWN. Grid rows may contain letters,
    dots, question marks, underscores and # blocks. Clues use ``12. clue text``.
    """

    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    parsed_title = title
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = line.rstrip(":").upper()
        if heading in {"GRID", "ACROSS", "DOWN"}:
            current = heading
            continue
        if heading.startswith("TITLE:") and parsed_title is None:
            parsed_title = line.split(":", 1)[1].strip()
            continue
        if current is not None:
            sections[current].append(line)

    if not sections["GRID"] or not sections["ACROSS"] or not sections["DOWN"]:
        raise PuzzleFormatError("Puzzle text must include GRID, ACROSS, and DOWN sections")

    grid = [line.replace(" ", "") for line in sections["GRID"] if _GRID_PATTERN.fullmatch(line)]
    across = _parse_clues(sections["ACROSS"])
    down = _parse_clues(sections["DOWN"])
    return build_puzzle_record(
        puzzle_id=puzzle_id,
        title=parsed_title or puzzle_id,
        solution_rows=grid,
        across_clues=across,
        down_clues=down,
        split=split,
    )


def _parse_clues(lines: Sequence[str]) -> dict[int, str]:
    clues: dict[int, str] = {}
    for line in lines:
        match = _CLUE_PATTERN.match(line)
        if not match:
            raise PuzzleFormatError(f"Invalid clue line: {line!r}")
        clues[int(match.group(1))] = match.group(2)
    return clues


def _validate_grid(rows: Sequence[str]) -> None:
    if not rows:
        raise PuzzleFormatError("Grid cannot be empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise PuzzleFormatError("Grid rows must be non-empty and equal width")
    if any(char not in "#.ABCDEFGHIJKLMNOPQRSTUVWXYZ" for row in rows for char in row):
        raise PuzzleFormatError("Grid contains unsupported characters")


def _extract_entries(
    template: Sequence[str],
    across_clues: Mapping[int | str, str],
    down_clues: Mapping[int | str, str],
) -> list[Entry]:
    height = len(template)
    width = len(template[0])
    numbers: dict[Cell, int] = {}
    next_number = 1

    def run_length(row: int, col: int, direction: Direction) -> int:
        dr, dc = (0, 1) if direction is Direction.ACROSS else (1, 0)
        length = 0
        while 0 <= row < height and 0 <= col < width and template[row][col] != "#":
            length += 1
            row += dr
            col += dc
        return length

    for row in range(height):
        for col in range(width):
            if template[row][col] == "#":
                continue
            starts_across = (col == 0 or template[row][col - 1] == "#") and (
                run_length(row, col, Direction.ACROSS) >= 2
            )
            starts_down = (row == 0 or template[row - 1][col] == "#") and (
                run_length(row, col, Direction.DOWN) >= 2
            )
            if starts_across or starts_down:
                numbers[(row, col)] = next_number
                next_number += 1

    normalized_across = {int(number): clue for number, clue in across_clues.items()}
    normalized_down = {int(number): clue for number, clue in down_clues.items()}
    entries: list[Entry] = []
    for (row, col), number in numbers.items():
        for direction, clue_map in (
            (Direction.ACROSS, normalized_across),
            (Direction.DOWN, normalized_down),
        ):
            length = run_length(row, col, direction)
            starts = (
                direction is Direction.ACROSS
                and (col == 0 or template[row][col - 1] == "#")
                or direction is Direction.DOWN
                and (row == 0 or template[row - 1][col] == "#")
            )
            if not starts or length < 2:
                continue
            if number not in clue_map:
                raise PuzzleFormatError(f"Missing {direction.value} clue {number}")
            dr, dc = (0, 1) if direction is Direction.ACROSS else (1, 0)
            cells = tuple((row + index * dr, col + index * dc) for index in range(length))
            entries.append(
                Entry(
                    id=f"{number}{'A' if direction is Direction.ACROSS else 'D'}",
                    number=number,
                    direction=direction,
                    clue=clue_map[number],
                    cells=cells,
                )
            )

    used_across = {entry.number for entry in entries if entry.direction is Direction.ACROSS}
    used_down = {entry.number for entry in entries if entry.direction is Direction.DOWN}
    extra = (set(normalized_across) - used_across) | (set(normalized_down) - used_down)
    if extra:
        raise PuzzleFormatError(f"Clues do not map to grid entries: {sorted(extra)}")
    return sorted(entries, key=lambda item: (item.number, item.direction.value))


def _build_intersections(entries: Sequence[Entry]) -> list[Intersection]:
    cell_entries: dict[Cell, list[tuple[str, int]]] = defaultdict(list)
    for entry in entries:
        for index, cell in enumerate(entry.cells):
            cell_entries[cell].append((entry.id, index))

    intersections: list[Intersection] = []
    for cell, occupants in cell_entries.items():
        if len(occupants) != 2:
            continue
        (entry_a, index_a), (entry_b, index_b) = sorted(occupants)
        intersections.append(
            Intersection(
                entry_a=entry_a,
                index_a=index_a,
                entry_b=entry_b,
                index_b=index_b,
                cell=cell,
            )
        )
    return sorted(intersections, key=lambda item: (item.entry_a, item.entry_b, item.cell))
