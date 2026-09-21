from __future__ import annotations

import pytest

from crossword_agent.models import PuzzleRecord
from crossword_agent.puzzles import PuzzleRepository
from crossword_agent.storage import reset_run_store_for_tests


@pytest.fixture
def demo_record() -> PuzzleRecord:
    return PuzzleRepository.bundled().all()[0]


@pytest.fixture(autouse=True)
def reset_store() -> None:
    reset_run_store_for_tests()

