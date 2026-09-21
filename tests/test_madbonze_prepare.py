from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_madbonze.py"
    spec = importlib.util.spec_from_file_location("prepare_madbonze", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load MadBonze preparation script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_madbonze_row_builds_grid_and_entries() -> None:
    prepare = _load_prepare_module()
    row = {
        "Unnamed: 0": 7,
        "name": "Synthetic human puzzle",
        "puzzle": (
            "Crossword Grid :\n"
            "_ _\n"
            "_ _\n"
            "Word Positions and Clues: "
            "1a (across, 2): First [Row 0, Col 0] "
            "1d (down, 2): First down [Row 0, Col 0] "
            "2d (down, 2): Second down [Row 0, Col 1] "
            "3aContainera (across, 2): Second [Row 1, Col 0]"
        ),
        "answer": "{'1a': 'AB', '1d': 'AC', '2d': 'BD', '3a': 'CD'}",
    }

    record = prepare.normalize_row(row, source_index=0, split="train")

    assert record["solution"] == ["AB", "CD"]
    assert [entry["id"] for entry in record["entries"]] == [
        "1A",
        "1D",
        "2D",
        "3A",
    ]
    assert record["entries"][0]["start"] == [0, 0]
    assert record["entries"][2]["direction"] == "down"


def test_normalize_madbonze_row_accepts_answer_keys_with_suffixes() -> None:
    prepare = _load_prepare_module()
    row = {
        "puzzle": (
            "Crossword Grid :\n"
            "_ _\n"
            "_ _\n"
            "Word Positions and Clues:\n"
            "1a (across, 2): First [Row 0, Col 0]\n"
            "1d (down, 2): First down [Row 0, Col 0]\n"
            "2d (down, 2): Second down [Row 0, Col 1]\n"
            "3a (across, 2): Second [Row 1, Col 0]"
        ),
        "answer": (
            "{'1aContainer': 'AB', '1dContainer': 'AC', "
            "'2dContainer': 'BD', '3aContainer': 'CD'}"
        ),
    }

    record = prepare.normalize_row(row, source_index=3, split="test")

    assert record["solution"] == ["AB", "CD"]
    assert record["split"] == "test"


def test_clean_answer_folds_crossword_diacritics() -> None:
    prepare = _load_prepare_module()

    assert prepare._clean_answer("crinière") == "CRINIERE"
    assert prepare._clean_answer("nô") == "NO"
