from __future__ import annotations

import json

import pytest

from crossword_agent.models import Direction
from crossword_agent.parser import (
    PuzzleFormatError,
    build_explicit_puzzle_record,
    parse_crosswordbench_text,
)
from crossword_agent.puzzles import PuzzleRepository


def test_bundled_grid_builds_entries_and_intersections(demo_record) -> None:
    puzzle = demo_record.puzzle

    assert puzzle.width == 7
    assert puzzle.height == 7
    assert len(puzzle.entries) == 24
    assert len(puzzle.intersections) == 36
    assert puzzle.entry_map["1A"].length == 3
    assert puzzle.entry_map["1D"].direction is Direction.DOWN
    assert "solution" not in puzzle.model_dump()


def test_bundled_puzzles_include_connected_symmetric_grids() -> None:
    connected = [
        record.puzzle
        for record in PuzzleRepository.bundled().all()
        if _open_cells_are_connected(record.puzzle.template)
    ]

    assert len(connected) >= 3
    for puzzle in connected:
        assert puzzle.template == tuple(
            row[::-1] for row in reversed(puzzle.template)
        )
        assert all(set(row) != {"#"} for row in puzzle.template)
        assert all(
            any(puzzle.template[row][col] != "#" for row in range(puzzle.height))
            for col in range(puzzle.width)
        )


def test_parse_normalized_crosswordbench_text() -> None:
    text = """
    TITLE: Tiny
    GRID:
    CAT
    ARE
    TEN

    ACROSS:
    1. Pet
    4. Exist
    5. Number

    DOWN:
    1. Pet
    2. Exist
    3. Number
    """

    record = parse_crosswordbench_text(text, puzzle_id="tiny")

    assert record.puzzle.title == "Tiny"
    assert record.solution == ("CAT", "ARE", "TEN")
    assert {entry.id for entry in record.puzzle.entries} == {
        "1A",
        "1D",
        "2D",
        "3D",
        "4A",
        "5A",
    }


def test_missing_clue_is_rejected() -> None:
    with pytest.raises(PuzzleFormatError, match="Missing down clue"):
        parse_crosswordbench_text(
            """
            GRID:
            CAT
            ARE
            TEN
            ACROSS:
            1. Pet
            4. Exist
            5. Number
            DOWN:
            1. Pet
            2. Exist
            """,
            puzzle_id="invalid",
        )


def test_grid_without_entries_is_rejected() -> None:
    with pytest.raises(PuzzleFormatError, match="at least one Across or Down"):
        from crossword_agent.parser import build_puzzle_record

        build_puzzle_record(
            puzzle_id="empty",
            title="Empty",
            solution_rows=["###", "#A#", "###"],
            across_clues={},
            down_clues={},
        )


def test_configured_repository_reads_normalized_dataset(tmp_path, monkeypatch) -> None:
    record = PuzzleRepository.bundled().all()[0]
    dataset = tmp_path / "crosswordbench.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": record.puzzle.id,
                    "title": "Imported puzzle",
                    "split": "demo",
                    "solution": record.solution,
                    "across": {
                        str(entry.number): entry.clue
                        for entry in record.puzzle.entries
                        if entry.direction is Direction.ACROSS
                    },
                    "down": {
                        str(entry.number): entry.clue
                        for entry in record.puzzle.entries
                        if entry.direction is Direction.DOWN
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CROSSWORD_DATASET_PATH", str(dataset))

    configured = PuzzleRepository.configured()

    assert len(configured.all()) == 1
    assert configured.all()[0].puzzle.title == "Imported puzzle"


def test_invalid_prepared_dataset_requests_regeneration(tmp_path) -> None:
    dataset = tmp_path / "invalid.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "invalid",
                    "title": "Invalid",
                    "solution": ["###", "#A#", "###"],
                    "across": {},
                    "down": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(PuzzleFormatError, match="Regenerate it"):
        PuzzleRepository.from_normalized_json(dataset)


def test_explicit_entries_preserve_source_numbering() -> None:
    entries = [
        {
            "id": f"{number}A",
            "number": number,
            "direction": "across",
            "clue": clue,
            "answer": answer,
            "start": [row, 0],
        }
        for number, row, clue, answer in (
            (5, 0, "Pet", "CAT"),
            (8, 1, "Exist", "ARE"),
            (12, 2, "Number after nine", "TEN"),
        )
    ] + [
        {
            "id": f"{number}D",
            "number": number,
            "direction": "down",
            "clue": clue,
            "answer": answer,
            "start": [0, col],
        }
        for number, col, clue, answer in (
            (7, 0, "Pet", "CAT"),
            (9, 1, "Exist", "ARE"),
            (14, 2, "Number after nine", "TEN"),
        )
    ]

    record = build_explicit_puzzle_record(
        puzzle_id="source-numbering",
        title="Source numbering",
        solution_rows=["CAT", "ARE", "TEN"],
        entry_records=entries,
    )

    assert {entry.id for entry in record.puzzle.entries} == {
        "5A",
        "8A",
        "12A",
        "7D",
        "9D",
        "14D",
    }
    assert len(record.puzzle.intersections) == 9


def test_explicit_entries_mask_non_target_source_cells() -> None:
    record = build_explicit_puzzle_record(
        puzzle_id="target-subset",
        title="Target subset",
        solution_rows=["CATX"],
        entry_records=[
            {
                "id": "1A",
                "number": 1,
                "direction": "across",
                "clue": "Pet",
                "answer": "CAT",
                "start": [0, 0],
            }
        ],
    )

    assert record.puzzle.template == ("...#",)
    assert record.solution == ("CAT#",)


def _open_cells_are_connected(template: tuple[str, ...]) -> bool:
    open_cells = {
        (row, col)
        for row, line in enumerate(template)
        for col, char in enumerate(line)
        if char != "#"
    }
    pending = [next(iter(open_cells))]
    visited: set[tuple[int, int]] = set()
    while pending:
        cell = pending.pop()
        if cell in visited:
            continue
        visited.add(cell)
        row, col = cell
        pending.extend(
            neighbor
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if neighbor in open_cells and neighbor not in visited
        )
    return visited == open_cells
