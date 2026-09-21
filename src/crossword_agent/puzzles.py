from __future__ import annotations

import json
import os
import random
from importlib.resources import files
from pathlib import Path
from typing import Any

from crossword_agent.models import PuzzleRecord
from crossword_agent.parser import (
    PuzzleFormatError,
    build_explicit_puzzle_record,
    build_puzzle_record,
)


class PuzzleRepository:
    def __init__(self, records: list[PuzzleRecord]) -> None:
        if not records:
            raise ValueError("Puzzle repository requires at least one puzzle")
        self._records = {record.puzzle.id: record for record in records}

    @classmethod
    def bundled(cls) -> PuzzleRepository:
        path = files("crossword_agent").joinpath("data/demo_puzzles.json")
        return cls(_load_records(json.loads(path.read_text(encoding="utf-8"))))

    @classmethod
    def configured(cls) -> PuzzleRepository:
        path = os.getenv("CROSSWORD_DATASET_PATH")
        return cls.from_normalized_json(Path(path)) if path else cls.bundled()

    @classmethod
    def from_normalized_json(cls, path: Path) -> PuzzleRepository:
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured crossword dataset does not exist: {path}. "
                "Run one of the dataset preparation scripts first."
            )
        payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        try:
            return cls(_load_records(payload))
        except (KeyError, TypeError, ValueError) as error:
            raise PuzzleFormatError(
                f"Prepared dataset is invalid: {path}. Regenerate it with "
                "the appropriate dataset preparation script. "
                f"Original error: {error}"
            ) from error

    def get(self, puzzle_id: str) -> PuzzleRecord:
        try:
            return self._records[puzzle_id]
        except KeyError as error:
            raise KeyError(f"Unknown puzzle ID: {puzzle_id}") from error

    def random(
        self,
        *,
        exclude: str | None = None,
        rng: random.Random | None = None,
    ) -> PuzzleRecord:
        choices = [record for key, record in self._records.items() if key != exclude]
        if not choices:
            choices = list(self._records.values())
        return (rng or random.SystemRandom()).choice(choices)

    def all(self) -> tuple[PuzzleRecord, ...]:
        return tuple(self._records.values())


def load_normalized_records(path: Path) -> list[PuzzleRecord]:
    payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return _load_records(payload)


def _load_records(payload: list[dict[str, Any]]) -> list[PuzzleRecord]:
    records: list[PuzzleRecord] = []
    for item in payload:
        common = {
            "puzzle_id": item["id"],
            "title": item.get("title", item["id"]),
            "solution_rows": item["solution"],
            "split": item.get("split", "external"),
        }
        if "entries" in item:
            records.append(
                build_explicit_puzzle_record(
                    **common,
                    entry_records=item["entries"],
                )
            )
        else:
            records.append(
                build_puzzle_record(
                    **common,
                    across_clues=item["across"],
                    down_clues=item["down"],
                )
            )
    return records
