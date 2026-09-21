from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

Cell = tuple[int, int]


class Direction(StrEnum):
    ACROSS = "across"
    DOWN = "down"


class SolveStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SOLVED = "SOLVED"
    STALLED = "STALLED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    number: int
    direction: Direction
    clue: str
    cells: tuple[Cell, ...]

    @computed_field
    @property
    def length(self) -> int:
        return len(self.cells)


class Intersection(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_a: str
    index_a: int
    entry_b: str
    index_b: int
    cell: Cell


class PuzzleDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    width: int
    height: int
    template: tuple[str, ...]
    entries: tuple[Entry, ...]
    intersections: tuple[Intersection, ...]

    @computed_field
    @property
    def entry_map(self) -> dict[str, Entry]:
        return {entry.id: entry for entry in self.entries}


class PuzzleRecord(BaseModel):
    puzzle: PuzzleDefinition
    solution: tuple[str, ...] | None = None
    split: str = "demo"


class Candidate(BaseModel):
    answer: str
    rank: int = Field(default=1, ge=1)
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "model"

    @computed_field
    @property
    def score(self) -> float:
        sources = set(self.source.split("+"))
        verifier_support = 1.0 if "verifier" in sources else 0.0
        corroboration = min(1.0, max(0, len(sources) - 1) / 2)
        return (
            0.70 * (1.0 / self.rank)
            + 0.15 * self.retrieval_score
            + 0.05 * self.lexical_score
            + 0.07 * verifier_support
            + 0.03 * corroboration
        )


class ClueRequest(BaseModel):
    entry_id: str
    clue: str
    length: int = Field(ge=1)
    pattern: str
    rejected_answers: tuple[str, ...] = ()
    lexical_options: tuple[str, ...] = ()
    candidate_limit: int = Field(default=5, ge=1, le=20)
    strategy: Literal["generate", "verify"] = "generate"
    current_answer: str | None = None
    alternative_answers: tuple[str, ...] = ()
    crossing_context: tuple[str, ...] = ()


class CandidateAnswer(BaseModel):
    entry_id: str
    answer: str = Field(
        validation_alias=AliasChoices("answer", "clue_answer"),
        serialization_alias="answer",
    )


class CandidateBatch(BaseModel):
    candidates: list[CandidateAnswer]

    @model_validator(mode="before")
    @classmethod
    def retain_valid_candidates(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("candidates"), list):
            return value
        candidates = value["candidates"]
        valid = [
            candidate
            for candidate in candidates
            if isinstance(candidate, CandidateAnswer)
            or (
                isinstance(candidate, dict)
                and candidate.get("entry_id")
                and (candidate.get("answer") or candidate.get("clue_answer"))
            )
        ]
        if valid and len(valid) != len(candidates):
            return {**value, "candidates": valid}
        return value


class UsageMetrics(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float | None = None

    def add(self, other: UsageMetrics) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_ms += other.latency_ms
        if other.estimated_cost_usd is not None:
            self.estimated_cost_usd = (self.estimated_cost_usd or 0.0) + other.estimated_cost_usd


class AgentEvent(BaseModel):
    phase: str
    message: str
    entry_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SolveMetrics(BaseModel):
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: float = 0.0
    elapsed_ms: float = 0.0
    search_nodes: int = 0
    candidate_replacements: int = 0
    constraint_violations: int = 0
    search_hypotheses: int = 0
    search_score_margin: float | None = None
    verification_calls: int = 0
    estimated_cost_usd: float | None = None


class RunSnapshot(BaseModel):
    run_id: str
    puzzle_id: str
    status: SolveStatus
    grid: tuple[str, ...]
    assignment: dict[str, str] = Field(default_factory=dict)
    candidate_pool: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    unresolved_entries: tuple[str, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    metrics: SolveMetrics = Field(default_factory=SolveMetrics)
    error: str | None = None


class PublicEntry(BaseModel):
    id: str
    number: int
    direction: Direction
    clue: str
    length: int
    start: Cell


class PublicPuzzle(BaseModel):
    id: str
    title: str
    width: int
    height: int
    template: tuple[str, ...]
    entries: tuple[PublicEntry, ...]


def public_puzzle(puzzle: PuzzleDefinition) -> PublicPuzzle:
    return PublicPuzzle(
        id=puzzle.id,
        title=puzzle.title,
        width=puzzle.width,
        height=puzzle.height,
        template=puzzle.template,
        entries=tuple(
            PublicEntry(
                id=entry.id,
                number=entry.number,
                direction=entry.direction,
                clue=entry.clue,
                length=entry.length,
                start=entry.cells[0],
            )
            for entry in puzzle.entries
        ),
    )
