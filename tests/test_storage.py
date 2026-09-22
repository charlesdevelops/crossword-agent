import pytest

from crossword_agent.puzzles import PuzzleRepository
from crossword_agent.storage import (
    DailyQuotaExceededError,
    InMemoryRunStore,
    RunBusyError,
)


def test_store_enforces_global_lock_and_daily_quota() -> None:
    repository = PuzzleRepository.bundled()
    puzzle_id = repository.all()[0].puzzle.id
    store = InMemoryRunStore(max_daily_runs=1, repository=repository)

    first = store.create_run(puzzle_id)
    assert first.reasoning_effort == "none"
    with pytest.raises(RunBusyError):
        store.create_run(puzzle_id)

    store.release_run(first.run_id)
    with pytest.raises(DailyQuotaExceededError):
        store.create_run(puzzle_id)


def test_claim_is_idempotent() -> None:
    repository = PuzzleRepository.bundled()
    puzzle_id = repository.all()[0].puzzle.id
    store = InMemoryRunStore(repository=repository)
    record = store.create_run(puzzle_id)

    assert store.claim_run(record.run_id)
    assert not store.claim_run(record.run_id)
