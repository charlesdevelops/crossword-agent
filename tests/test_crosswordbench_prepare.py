from __future__ import annotations

import json

from scripts.prepare_crosswordbench import (
    VARIANTS,
    _parse_prompt_clues,
    _parse_reference_answers,
    normalize_row,
)


def test_crosswordbench_variants_point_to_expected_sources() -> None:
    assert VARIANTS["english"]["14x14"].startswith("english/")
    assert VARIANTS["english_simple"]["7x7"].startswith("english_simple/")
    assert "14x14" not in VARIANTS["english_simple"]


def test_normalizes_crosswordbench_template_answers_and_prompt() -> None:
    row = {
        "grid": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "reference_answers": {
            "across 1": "CAT",
            "across 4": "ARE",
            "across 5": "TEN",
            "down 1": "CAT",
            "down 2": "ARE",
            "down 3": "TEN",
        },
        "question": """
        Across:
        1. Pet
        4. Exist
        5. Number after nine
        Down:
        1. Pet
        2. Exist
        3. Number after nine
        """,
    }

    puzzle = normalize_row(row, source_index=0, size="3x3", split="demo")

    assert puzzle["solution"] == ["CAT", "ARE", "TEN"]
    assert puzzle["across"] == {
        "1": "Pet",
        "4": "Exist",
        "5": "Number after nine",
    }
    assert puzzle["down"] == {
        "1": "Pet",
        "2": "Exist",
        "3": "Number after nine",
    }
    assert puzzle["id"].startswith("crosswordbench-3x3-")


def test_crosswordbench_uses_one_for_open_and_zero_for_block() -> None:
    row = {
        "grid": [
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
        ],
        "reference_answers": {
            "across 1": "CAT",
            "across 4": "DOG",
            "across 7": "ARE",
            "across 8": "ORE",
            "across 9": "TEN",
            "across 10": "GEM",
            "across 11": "SUN",
            "across 14": "PEN",
            "across 17": "USE",
            "across 18": "ERA",
            "across 19": "NET",
            "across 20": "NET",
            "down 1": "CAT",
            "down 2": "ARE",
            "down 3": "TEN",
            "down 4": "DOG",
            "down 5": "ORE",
            "down 6": "GEM",
            "down 11": "SUN",
            "down 12": "USE",
            "down 13": "NET",
            "down 14": "PEN",
            "down 15": "ERE",
            "down 16": "NAT",
        },
        "clues": {
            "across": {
                str(number): f"Across clue {number}"
                for number in (1, 4, 7, 8, 9, 10, 11, 14, 17, 18, 19, 20)
            },
            "down": {
                str(number): f"Down clue {number}"
                for number in (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)
            },
        },
    }

    puzzle = normalize_row(row, source_index=0, size="7x7", split="demo")

    assert puzzle["solution"][0] == "CAT#DOG"
    assert len(puzzle["across"]) == 12
    assert len(puzzle["down"]) == 12


def test_normalizes_nested_completed_grid_and_clue_maps() -> None:
    row = {
        "puzzle": {
            "solution_grid": ["CAT", "ARE", "TEN"],
            "clues": {
                "across": {"1": "Pet", "4": "Exist", "5": "Number after nine"},
                "down": {"1": "Pet", "2": "Exist", "3": "Number after nine"},
            },
        }
    }

    puzzle = normalize_row(row, source_index=8, size="3x3", split="evaluation")

    assert puzzle["solution"] == ["CAT", "ARE", "TEN"]
    assert puzzle["split"] == "evaluation"


def test_parses_puzzle_state_and_reference_answer_text() -> None:
    clues = _parse_prompt_clues(
        """
        Across Clues:
        1. Pet
        4. Exist
        Down Clues:
        1. Pet
        2. Exist
        """
    )
    answers = _parse_reference_answers(
        """
        Across Answers:
        1. CAT
        4. ARE
        Down Answers:
        1. CAT
        2. ARE
        """
    )

    assert clues[("across", 1)] == "Pet"
    assert clues[("down", 2)] == "Exist"
    assert answers[("across", 4)] == "ARE"
    assert answers[("down", 1)] == "CAT"


def test_uses_completed_puzzle_state_instead_of_partial_grid() -> None:
    row = {
        "id": 69,
        "partial_grid_0.25": json.dumps(
            {
                "grid": [["-", "-", "-"], ["-", "R", "-"], ["-", "-", "-"]],
                "mask": [[False, False, False], [False, True, False], [False, False, False]],
            }
        ),
        "puzzle_state": json.dumps(
            {
                "grid": [["C", "A", "T"], ["A", "R", "E"], ["T", "E", "N"]],
                "wordlist": [
                    ["CAT", "Pet", 0, 0, 0],
                    ["ARE", "Exist", 1, 0, 0],
                    ["TEN", "Number after nine", 2, 0, 0],
                    ["CAT", "Pet", 0, 0, 1],
                    ["ARE", "Exist", 0, 1, 1],
                    ["TEN", "Number after nine", 0, 2, 1],
                    ["AR", "Generator-only placement", 1, 0, 0],
                ],
            }
        ),
        "reference_answer": json.dumps(
            [
                {"direction": "across 1", "clue": "Pet", "answer": "CAT"},
                {"direction": "across 4", "clue": "Exist", "answer": "ARE"},
                {
                    "direction": "across 5",
                    "clue": "Number after nine",
                    "answer": "TEN",
                },
                {"direction": "down 1", "clue": "Pet", "answer": "CAT"},
                {"direction": "down 2", "clue": "Exist", "answer": "ARE"},
                {
                    "direction": "down 3",
                    "clue": "Number after nine",
                    "answer": "TEN",
                },
            ]
        ),
    }

    puzzle = normalize_row(row, source_index=0, size="3x3", split="demo")

    assert puzzle["title"] == "CrossWordBench 3x3 #69"
    assert puzzle["solution"] == ["CAT", "ARE", "TEN"]
    entries = {entry["id"]: entry for entry in puzzle["entries"]}
    assert entries["1A"]["clue"] == "Pet"
    assert entries["3D"]["clue"] == "Number after nine"
    assert entries["1A"]["start"] == [0, 0]
    assert puzzle["source_unreferenced_wordlist_entries"] == 1
