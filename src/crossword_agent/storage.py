from __future__ import annotations

import threading
import time
import uuid
from typing import Protocol

from pydantic import BaseModel

from crossword_agent.models import PuzzleDefinition, RunSnapshot, SolveMetrics, SolveStatus
from crossword_agent.puzzles import PuzzleRepository


class RunBusyError(RuntimeError):
    pass


class RunRecord(BaseModel):
    run_id: str
    puzzle_id: str
    model_id: str | None = None
    reasoning_effort: str = "none"
    status: SolveStatus
    snapshot: RunSnapshot
    created_at: int
    updated_at: int


class RunStore(Protocol):
    def get_puzzle(self, puzzle_id: str) -> PuzzleDefinition: ...

    def create_run(
        self,
        puzzle_id: str,
        model_id: str | None = None,
        reasoning_effort: str = "none",
    ) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def claim_run(self, run_id: str) -> bool: ...

    def update_snapshot(self, snapshot: RunSnapshot) -> None: ...

    def release_run(self, run_id: str) -> None: ...


def _initial_record(
    puzzle: PuzzleDefinition,
    run_id: str | None = None,
    model_id: str | None = None,
    reasoning_effort: str = "none",
) -> RunRecord:
    now = int(time.time())
    resolved_run_id = run_id or str(uuid.uuid4())
    snapshot = RunSnapshot(
        run_id=resolved_run_id,
        puzzle_id=puzzle.id,
        model_id=model_id,
        status=SolveStatus.PENDING,
        grid=puzzle.template,
        unresolved_entries=tuple(entry.id for entry in puzzle.entries),
        metrics=SolveMetrics(),
    )
    return RunRecord(
        run_id=resolved_run_id,
        puzzle_id=puzzle.id,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        status=SolveStatus.PENDING,
        snapshot=snapshot,
        created_at=now,
        updated_at=now,
    )


class InMemoryRunStore:
    def __init__(
        self,
        *,
        repository: PuzzleRepository | None = None,
    ) -> None:
        self._repository = repository or PuzzleRepository.configured()
        self._records: dict[str, RunRecord] = {}
        self._active_run_id: str | None = None
        self._lock = threading.Lock()

    def get_puzzle(self, puzzle_id: str) -> PuzzleDefinition:
        return self._repository.get(puzzle_id).puzzle

    def create_run(
        self,
        puzzle_id: str,
        model_id: str | None = None,
        reasoning_effort: str = "none",
    ) -> RunRecord:
        puzzle = self.get_puzzle(puzzle_id)
        with self._lock:
            if self._active_run_id is not None:
                raise RunBusyError("Another solve is already active")
            record = _initial_record(
                puzzle,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
            )
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
        _store = InMemoryRunStore(repository=repository)
    return _store


def reset_run_store_for_tests() -> None:
    global _store
    _store = None
