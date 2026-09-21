from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

from crossword_agent.models import PuzzleDefinition, RunSnapshot, SolveMetrics, SolveStatus
from crossword_agent.puzzles import PuzzleRepository


class RunBusyError(RuntimeError):
    pass


class DailyQuotaExceededError(RuntimeError):
    pass


class RunRecord(BaseModel):
    run_id: str
    puzzle_id: str
    status: SolveStatus
    snapshot: RunSnapshot
    created_at: int
    updated_at: int


class RunStore(Protocol):
    def get_puzzle(self, puzzle_id: str) -> PuzzleDefinition: ...

    def create_run(self, puzzle_id: str) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def claim_run(self, run_id: str) -> bool: ...

    def update_snapshot(self, snapshot: RunSnapshot) -> None: ...

    def release_run(self, run_id: str) -> None: ...


def _initial_record(puzzle: PuzzleDefinition, run_id: str | None = None) -> RunRecord:
    now = int(time.time())
    resolved_run_id = run_id or str(uuid.uuid4())
    snapshot = RunSnapshot(
        run_id=resolved_run_id,
        puzzle_id=puzzle.id,
        status=SolveStatus.PENDING,
        grid=puzzle.template,
        unresolved_entries=tuple(entry.id for entry in puzzle.entries),
        metrics=SolveMetrics(),
    )
    return RunRecord(
        run_id=resolved_run_id,
        puzzle_id=puzzle.id,
        status=SolveStatus.PENDING,
        snapshot=snapshot,
        created_at=now,
        updated_at=now,
    )


class InMemoryRunStore:
    def __init__(
        self,
        *,
        max_daily_runs: int = 10,
        repository: PuzzleRepository | None = None,
    ) -> None:
        self._max_daily_runs = max_daily_runs
        self._repository = repository or PuzzleRepository.configured()
        self._records: dict[str, RunRecord] = {}
        self._active_run_id: str | None = None
        self._daily_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def get_puzzle(self, puzzle_id: str) -> PuzzleDefinition:
        return self._repository.get(puzzle_id).puzzle

    def create_run(self, puzzle_id: str) -> RunRecord:
        puzzle = self.get_puzzle(puzzle_id)
        day = datetime.now(UTC).date().isoformat()
        with self._lock:
            if self._active_run_id is not None:
                raise RunBusyError("Another solve is already active")
            if self._daily_counts.get(day, 0) >= self._max_daily_runs:
                raise DailyQuotaExceededError("Daily solve quota exhausted")
            record = _initial_record(puzzle)
            self._daily_counts[day] = self._daily_counts.get(day, 0) + 1
            self._active_run_id = record.run_id
            self._records[record.run_id] = record
            return record.model_copy(deep=True)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            return record.model_copy(deep=True) if record else None

    def claim_run(self, run_id: str) -> bool:
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.status is not SolveStatus.PENDING:
                return False
            record.status = SolveStatus.RUNNING
            record.snapshot.status = SolveStatus.RUNNING
            record.updated_at = int(time.time())
            return True

    def update_snapshot(self, snapshot: RunSnapshot) -> None:
        with self._lock:
            record = self._records[snapshot.run_id]
            record.snapshot = snapshot.model_copy(deep=True)
            record.status = snapshot.status
            record.updated_at = int(time.time())

    def release_run(self, run_id: str) -> None:
        with self._lock:
            if self._active_run_id == run_id:
                self._active_run_id = None


_store: RunStore | None = None


def get_run_store(*, repository: PuzzleRepository | None = None) -> RunStore:
    global _store
    if _store is None:
        max_daily_runs = int(os.getenv("MAX_DAILY_RUNS", "10"))
        _store = InMemoryRunStore(
            max_daily_runs=max_daily_runs,
            repository=repository,
        )
    return _store


def reset_run_store_for_tests() -> None:
    global _store
    _store = None
