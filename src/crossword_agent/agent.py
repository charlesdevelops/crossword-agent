from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from crossword_agent.constraints import (
    SearchHypothesis,
    count_constraint_violations,
    entry_pattern,
    render_grid,
    search_assignments,
    validate_domain,
)
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.models import (
    AgentEvent,
    Candidate,
    ClueRequest,
    PuzzleDefinition,
    RunSnapshot,
    SolveMetrics,
    SolveStatus,
)
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.tracing import start_span

logger = logging.getLogger(__name__)


class AgentSettings(BaseModel):
    max_model_calls: int = Field(default=12, ge=1)
    deadline_seconds: float = Field(default=120.0, gt=0)
    stagnant_round_limit: int = Field(default=2, ge=1)
    max_search_nodes: int = Field(default=50_000, ge=1)
    candidates_per_entry: int = Field(default=10, ge=1, le=20)
    initial_batch_size: int = Field(default=4, ge=1, le=20)
    initial_requested_candidates: int = Field(default=4, ge=1, le=20)
    requested_candidates: int = Field(default=8, ge=1, le=20)
    lexical_option_limit: int = Field(default=12, ge=1, le=100)
    n_best_hypotheses: int = Field(default=5, ge=1, le=20)
    minimum_verification_calls: int = Field(default=2, ge=0, le=10)
    same_pattern_requery_limit: int = Field(default=2, ge=1, le=5)
    trusted_crossing_support: float = Field(default=0.68, ge=0.0, le=1.0)


class AgentState(TypedDict, total=False):
    run_id: str
    puzzle: PuzzleDefinition
    candidate_pool: dict[str, list[Candidate]]
    domains: dict[str, list[Candidate]]
    assignment: dict[str, Candidate]
    hypotheses: tuple[SearchHypothesis, ...]
    attempts: dict[str, int]
    rejected: dict[str, list[str]]
    last_patterns: dict[str, str]
    events: list[AgentEvent]
    metrics: SolveMetrics
    status: SolveStatus
    started_at: float
    last_coverage: int
    last_candidate_count: int
    last_assignment_score: float
    stagnant_rounds: int
    verification_calls: int
    last_verification_calls: int
    selected_entry_id: str | None
    selected_request: ClueRequest | None
    error: str | None


SnapshotObserver = Callable[[RunSnapshot], Awaitable[None]]


def _traced_node(phase: str):
    def decorate(function):
        @wraps(function)
        async def wrapped(state: AgentState) -> dict[str, Any]:
            puzzle = state["puzzle"]
            with start_span(
                f"agent.{phase}",
                attributes={
                    "agent.phase": phase,
                    "agent.run.id": state["run_id"],
                    "crossword.puzzle.id": puzzle.id,
                },
            ) as span:
                result = await function(state)
                status = result.get("status")
                if status is not None:
                    span.set_attribute(
                        "agent.status",
                        status.value if isinstance(status, SolveStatus) else status,
                    )
                entry_id = result.get("selected_entry_id") or state.get(
                    "selected_entry_id"
                )
                if entry_id:
                    span.set_attribute("crossword.entry.id", entry_id)
                metrics = result.get("metrics")
                if metrics is not None:
                    span.set_attribute(
                        "gen_ai.client.call.count",
                        metrics.model_calls,
                    )
                    span.set_attribute(
                        "agent.search.hypothesis_count",
                        metrics.search_hypotheses,
                    )
                    span.set_attribute(
                        "agent.verification.call_count",
                        metrics.verification_calls,
                    )
                request = result.get("selected_request")
                if request is not None:
                    span.set_attribute("agent.request.strategy", request.strategy)
                span.set_status("OK")
                return result

        return wrapped

    return decorate


def build_agent_graph(
    provider: CandidateProvider,
    *,
    lexicon: PatternLexicon | None = None,
    settings: AgentSettings | None = None,
    observer: SnapshotObserver | None = None,
):
    lexicon = lexicon or PatternLexicon.empty()
    settings = settings or AgentSettings()

    @_traced_node("initialize")
    async def initialize(state: AgentState) -> dict[str, Any]:
        puzzle = state["puzzle"]
        return {
            "domains": {},
            "candidate_pool": {},
            "assignment": {},
            "hypotheses": (),
            "attempts": {entry.id: 0 for entry in puzzle.entries},
            "rejected": {entry.id: [] for entry in puzzle.entries},
            "last_patterns": {},
            "events": [AgentEvent(phase="initialize", message="Puzzle initialized")],
            "metrics": SolveMetrics(),
            "status": SolveStatus.RUNNING,
            "started_at": time.monotonic(),
            "last_coverage": -1,
            "last_candidate_count": -1,
            "last_assignment_score": -1.0,
            "stagnant_rounds": 0,
            "verification_calls": 0,
            "last_verification_calls": 0,
            "selected_entry_id": None,
            "selected_request": None,
            "error": None,
        }

    @_traced_node("generate")
    async def generate_initial(state: AgentState) -> dict[str, Any]:
        requests = []
        candidate_pool: dict[str, list[Candidate]] = {}
        for entry in state["puzzle"].entries:
            candidate_pool[entry.id] = []
            requests.append(
                ClueRequest(
                    entry_id=entry.id,
                    clue=entry.clue,
                    length=entry.length,
                    pattern="?" * entry.length,
                    candidate_limit=settings.initial_requested_candidates,
                )
            )
        if not requests:
            metrics = state["metrics"].model_copy(deep=True)
            metrics.elapsed_ms = (time.monotonic() - state["started_at"]) * 1000
            return {
                "candidate_pool": {},
                "status": SolveStatus.FAILED,
                "error": "Puzzle contains no Across or Down entries",
                "metrics": metrics,
                "events": [
                    *state["events"],
                    AgentEvent(phase="validate", message="Puzzle contains no entries"),
                ],
            }
        batches = [
            requests[offset : offset + settings.initial_batch_size]
            for offset in range(0, len(requests), settings.initial_batch_size)
        ]
        batch_count = len(batches)
        progress_events = [
            *state["events"],
            AgentEvent(
                phase="generate",
                message=(
                    f"Generating initial candidates in {batch_count} concurrent "
                    f"{'batch' if batch_count == 1 else 'batches'}"
                ),
                details={
                    "batches": batch_count,
                    "entries": len(requests),
                    "candidates_per_entry": settings.initial_requested_candidates,
                },
            ),
        ]

        async def publish_progress() -> None:
            if observer is None:
                return
            progress_state: AgentState = {
                **state,
                "candidate_pool": candidate_pool,
                "events": progress_events,
                "metrics": _metrics_with_provider_usage(state, provider),
            }
            await observer(snapshot_from_state(progress_state))

        async def generate_batch(
            batch_number: int,
            batch: list[ClueRequest],
        ) -> tuple[
            int,
            list[ClueRequest],
            dict[str, list[Candidate]],
            Exception | None,
        ]:
            try:
                generated = await _generate_with_deadline(
                    provider,
                    batch,
                    deadline=state["started_at"] + settings.deadline_seconds,
                )
            except Exception as error:  # noqa: BLE001
                logger.exception(
                    "Initial model candidate batch failed",
                    extra={
                        "run_id": state["run_id"],
                        "puzzle_id": state["puzzle"].id,
                        "phase": "generate",
                        "request_count": len(batch),
                    },
                )
                return batch_number, batch, {}, error
            return batch_number, batch, generated, None

        await publish_progress()
        tasks = [
            asyncio.create_task(generate_batch(batch_number, batch))
            for batch_number, batch in enumerate(batches, start=1)
        ]
        failed_batches = 0
        last_error: Exception | None = None
        for completed in asyncio.as_completed(tasks):
            batch_number, batch, generated, error = await completed
            if error is not None:
                failed_batches += 1
                last_error = error
                progress_events.append(
                    AgentEvent(
                        phase="generate",
                        message=f"Initial batch {batch_number}/{batch_count} failed",
                        details={
                            "batch": batch_number,
                            "batches": batch_count,
                            "entries": len(batch),
                        },
                    )
                )
            else:
                for entry_id, candidates in generated.items():
                    candidate_pool.setdefault(entry_id, []).extend(candidates)
                progress_events.append(
                    AgentEvent(
                        phase="generate",
                        message=(
                            f"Initial batch {batch_number}/{batch_count} completed: "
                            f"{sum(len(items) for items in generated.values())} candidates "
                            f"for {sum(bool(items) for items in generated.values())} entries"
                        ),
                        details={
                            "batch": batch_number,
                            "batches": batch_count,
                            "entries": len(batch),
                            "candidate_count": sum(
                                len(items) for items in generated.values()
                            ),
                        },
                    )
                )
            await publish_progress()
        if failed_batches and not any(candidate_pool.values()):
            metrics = _metrics_with_provider_usage(state, provider)
            metrics.elapsed_ms = (time.monotonic() - state["started_at"]) * 1000
            return {
                "candidate_pool": {},
                "status": SolveStatus.FAILED,
                "error": str(last_error or "All initial model candidate batches failed"),
                "metrics": metrics,
                "events": [
                    *state["events"],
                    AgentEvent(phase="generate", message="Initial model call failed"),
                ],
            }
        return {
            "candidate_pool": candidate_pool,
            "metrics": _metrics_with_provider_usage(state, provider),
            "events": [
                *progress_events,
                AgentEvent(
                    phase="generate",
                    message=(
                        f"Generated candidates for "
                        f"{sum(bool(items) for items in candidate_pool.values())} entries"
                    ),
                    details={
                        "batches": batch_count,
                        "failed_batches": failed_batches,
                    },
                ),
            ],
        }

    @_traced_node("validate")
    async def validate_domains(state: AgentState) -> dict[str, Any]:
        puzzle = state["puzzle"]
        validated: dict[str, list[Candidate]] = {}
        for entry in puzzle.entries:
            candidates = [
                candidate.model_copy(
                    update={"lexical_score": lexicon.frequency_score(candidate.answer)}
                )
                for candidate in state.get("candidate_pool", {}).get(entry.id, [])
            ]
            validated[entry.id] = validate_domain(entry, candidates)[
                : settings.candidates_per_entry
            ]
        return {
            "domains": validated,
            "events": [
                *state["events"],
                AgentEvent(phase="validate", message="Validated candidate domains"),
            ],
        }

    @_traced_node("search")
    async def search_constraints(state: AgentState) -> dict[str, Any]:
        previous = state.get("assignment", {})
        result = search_assignments(
            state["puzzle"],
            state["domains"],
            max_nodes=settings.max_search_nodes,
            n_best=settings.n_best_hypotheses,
        )
        replacements = sum(
            entry_id not in result.assignment
            or result.assignment[entry_id].answer != candidate.answer
            for entry_id, candidate in previous.items()
        )
        metrics = state["metrics"].model_copy(deep=True)
        metrics.search_nodes += result.nodes_visited
        metrics.candidate_replacements += replacements
        metrics.search_hypotheses = len(result.hypotheses)
        metrics.search_score_margin = result.score_margin
        rejected = {key: list(value) for key, value in state["rejected"].items()}
        for entry_id, previous_candidate in previous.items():
            replacement = result.assignment.get(entry_id)
            if replacement is None or replacement.answer != previous_candidate.answer:
                rejected[entry_id] = sorted(
                    set(rejected.get(entry_id, [])) | {previous_candidate.answer}
                )
        return {
            "assignment": result.assignment,
            "hypotheses": result.hypotheses,
            "metrics": metrics,
            "rejected": rejected,
            "events": [
                *state["events"],
                AgentEvent(
                    phase="search",
                    message=(
                        f"Filled {len(result.assignment)}/{len(state['puzzle'].entries)} entries"
                    ),
                    details={
                        "nodes": result.nodes_visited,
                        "complete": result.complete,
                        "hypotheses": len(result.hypotheses),
                        "score_margin": result.score_margin,
                    },
                ),
            ],
        }

    @_traced_node("assess")
    async def assess_progress(state: AgentState) -> dict[str, Any]:
        metrics = _metrics_with_provider_usage(state, provider)
        elapsed = time.monotonic() - state["started_at"]
        metrics.elapsed_ms = elapsed * 1000
        metrics.constraint_violations = count_constraint_violations(
            state["puzzle"], state["assignment"]
        )
        metrics.verification_calls = state["verification_calls"]
        coverage = len(state["assignment"])
        candidate_count = sum(len(candidates) for candidates in state["domains"].values())
        assignment_score = sum(candidate.score for candidate in state["assignment"].values())
        made_progress = (
            coverage > state["last_coverage"]
            or candidate_count > state["last_candidate_count"]
            or assignment_score > state["last_assignment_score"] + 1e-9
            or state["verification_calls"] > state["last_verification_calls"]
        )
        stagnant = 0 if made_progress else state["stagnant_rounds"] + 1

        if state.get("status") is SolveStatus.FAILED:
            status = SolveStatus.FAILED
        elif (
            coverage == len(state["puzzle"].entries)
            and metrics.constraint_violations == 0
            and state["verification_calls"] >= settings.minimum_verification_calls
        ):
            status = SolveStatus.SOLVED
        elif (
            not state["assignment"]
            and not any(state["domains"].values())
            and sum(state["attempts"].values()) > 0
        ):
            status = SolveStatus.STALLED
        elif (
            metrics.model_calls >= settings.max_model_calls
            or elapsed >= settings.deadline_seconds
        ):
            status = SolveStatus.BUDGET_EXCEEDED
        elif stagnant >= settings.stagnant_round_limit:
            status = SolveStatus.STALLED
        else:
            status = SolveStatus.RUNNING
        return {
            "status": status,
            "metrics": metrics,
            "last_coverage": coverage,
            "last_candidate_count": candidate_count,
            "last_assignment_score": assignment_score,
            "stagnant_rounds": stagnant,
            "last_verification_calls": state["verification_calls"],
        }

    @_traced_node("select")
    async def select_requery(state: AgentState) -> dict[str, Any]:
        puzzle = state["puzzle"]
        unresolved = [entry for entry in puzzle.entries if entry.id not in state["assignment"]]
        eligible_unresolved = [
            entry
            for entry in unresolved
            if state["attempts"].get(entry.id, 0)
            < settings.same_pattern_requery_limit
            or (
                not state["domains"].get(entry.id)
                and state["last_patterns"].get(entry.id)
                != entry_pattern(puzzle, entry.id, state["assignment"])
            )
        ]
        selection_pool = eligible_unresolved
        if not unresolved:
            selection_pool = list(puzzle.entries)
        if not selection_pool:
            unresolved_ids = {entry.id for entry in unresolved}
            frontier_ids = {
                other_id
                for intersection in puzzle.intersections
                for entry_id, other_id in (
                    (intersection.entry_a, intersection.entry_b),
                    (intersection.entry_b, intersection.entry_a),
                )
                if entry_id in unresolved_ids and other_id in state["assignment"]
            }
            selection_pool = [
                puzzle.entry_map[entry_id]
                for entry_id in frontier_ids
                if entry_id in state["assignment"]
            ]
        if not selection_pool:
            selection_pool = list(puzzle.entries)
        priorities = {
            entry.id: _entry_priority(
                puzzle,
                entry.id,
                assignment=state["assignment"],
                domains=state["domains"],
                hypotheses=state.get("hypotheses", ()),
                attempts=state["attempts"].get(entry.id, 0),
                unresolved=entry.id not in state["assignment"],
            )
            for entry in selection_pool
        }
        selected = max(
            selection_pool,
            key=lambda entry: (priorities[entry.id]["priority"], entry.id),
        )
        strategy = "verify" if selected.id in state["assignment"] else "generate"
        pattern = _trusted_pattern(
            puzzle,
            selected.id,
            assignment=state["assignment"],
            hypotheses=state.get("hypotheses", ()),
            minimum_score=(
                settings.trusted_crossing_support
                + (0.10 if strategy == "verify" else 0.0)
                + min(0.12, state["attempts"].get(selected.id, 0) * 0.04)
            ),
            force_relax=strategy == "verify",
        )
        known = sum(char != "?" for char in pattern)
        lexical_words = (
            lexicon.find(pattern, settings.lexical_option_limit)
            if known >= max(1, selected.length // 3)
            else []
        )
        lexical_options = tuple(lexical_words[: settings.lexical_option_limit])
        current = state["assignment"].get(selected.id)
        alternatives = tuple(
            candidate.answer
            for candidate in state["domains"].get(selected.id, [])
            if current is None or candidate.answer != current.answer
        )[: settings.requested_candidates]
        request = ClueRequest(
            entry_id=selected.id,
            clue=selected.clue,
            length=selected.length,
            pattern=pattern,
            rejected_answers=tuple(state["rejected"].get(selected.id, [])),
            lexical_options=lexical_options,
            candidate_limit=settings.requested_candidates,
            strategy=strategy,
            current_answer=current.answer if current else None,
            alternative_answers=alternatives,
            crossing_context=_crossing_context(
                puzzle,
                selected.id,
                state["assignment"],
            ),
        )
        attempts = dict(state["attempts"])
        attempts[selected.id] = attempts.get(selected.id, 0) + 1
        last_patterns = dict(state["last_patterns"])
        last_patterns[selected.id] = pattern
        return {
            "selected_entry_id": selected.id,
            "selected_request": request,
            "attempts": attempts,
            "last_patterns": last_patterns,
            "events": [
                *state["events"],
                AgentEvent(
                    phase="select",
                    entry_id=selected.id,
                    message=(
                        f"{'Verifying' if strategy == 'verify' else 'Re-querying'} "
                        f"{selected.id} with pattern {pattern}"
                    ),
                    details={
                        "lexical_options": list(lexical_options),
                        **priorities[selected.id],
                    },
                ),
            ],
        }

    @_traced_node("requery")
    async def requery(state: AgentState) -> dict[str, Any]:
        request = state["selected_request"]
        if request is None:
            return {"status": SolveStatus.STALLED}
        try:
            generated = await _generate_with_deadline(
                provider,
                [request],
                deadline=state["started_at"] + settings.deadline_seconds,
            )
        except Exception as error:  # noqa: BLE001
            logger.exception(
                "Targeted model candidate generation failed",
                extra={
                    "run_id": state["run_id"],
                    "puzzle_id": state["puzzle"].id,
                    "entry_id": request.entry_id,
                    "phase": "requery",
                    "request_count": 1,
                },
            )
            metrics = _metrics_with_provider_usage(state, provider)
            metrics.elapsed_ms = (time.monotonic() - state["started_at"]) * 1000
            return {
                "status": SolveStatus.FAILED,
                "error": str(error),
                "metrics": metrics,
                "events": [
                    *state["events"],
                    AgentEvent(
                        phase="requery",
                        entry_id=request.entry_id,
                        message="Targeted model call failed",
                    ),
                ],
            }
        candidate_pool = {
            key: list(value) for key, value in state["candidate_pool"].items()
        }
        new_candidates = generated.get(request.entry_id, [])
        known = sum(char != "?" for char in request.pattern)
        if known * 2 >= request.length:
            new_candidates.extend(
                _lexical_candidates(
                    lexicon.find(request.pattern, settings.lexical_option_limit),
                    lexicon,
                )
            )
        candidate_pool.setdefault(request.entry_id, []).extend(new_candidates)
        verification_calls = state["verification_calls"] + (
            1 if request.strategy == "verify" else 0
        )
        return {
            "candidate_pool": candidate_pool,
            "verification_calls": verification_calls,
            "events": [
                *state["events"],
                AgentEvent(
                    phase="requery",
                    entry_id=request.entry_id,
                    message=(
                        f"Added {len(new_candidates)} "
                        f"{'verification' if request.strategy == 'verify' else 'targeted'} "
                        "candidates"
                    ),
                ),
            ],
        }

    @_traced_node("finalize")
    async def finalize(state: AgentState) -> dict[str, Any]:
        status = state["status"]
        assignment_complete = len(state.get("assignment", {})) == len(
            state["puzzle"].entries
        )
        if status is SolveStatus.BUDGET_EXCEEDED and assignment_complete:
            if state["metrics"].constraint_violations:
                message = "Grid filled, but crossing conflicts still need review"
            else:
                message = "Grid filled, but targeted verification did not complete"
        elif status is SolveStatus.BUDGET_EXCEEDED:
            message = "Run stopped after reaching its model-call or time limit"
        else:
            message = f"Run finished with status {status.value}"
        return {
            "events": [
                *state["events"],
                AgentEvent(
                    phase="finalize",
                    message=message,
                ),
            ]
        }

    def route_after_initial(state: AgentState) -> str:
        return "finalize" if state.get("status") is SolveStatus.FAILED else "validate_domains"

    def route_after_assess(state: AgentState) -> str:
        return "select_requery" if state["status"] is SolveStatus.RUNNING else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("generate_initial", generate_initial)
    graph.add_node("validate_domains", validate_domains)
    graph.add_node("search_constraints", search_constraints)
    graph.add_node("assess_progress", assess_progress)
    graph.add_node("select_requery", select_requery)
    graph.add_node("requery", requery)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "generate_initial")
    graph.add_conditional_edges(
        "generate_initial",
        route_after_initial,
        {"validate_domains": "validate_domains", "finalize": "finalize"},
    )
    graph.add_edge("validate_domains", "search_constraints")
    graph.add_edge("search_constraints", "assess_progress")
    graph.add_conditional_edges(
        "assess_progress",
        route_after_assess,
        {"select_requery": "select_requery", "finalize": "finalize"},
    )
    graph.add_edge("select_requery", "requery")
    graph.add_edge("requery", "validate_domains")
    graph.add_edge("finalize", END)
    return graph.compile()


async def solve_puzzle(
    *,
    run_id: str,
    puzzle: PuzzleDefinition,
    provider: CandidateProvider,
    lexicon: PatternLexicon | None = None,
    settings: AgentSettings | None = None,
    observer: SnapshotObserver | None = None,
) -> RunSnapshot:
    with start_span(
        "agent.graph",
        attributes={
            "agent.run.id": run_id,
            "crossword.puzzle.id": puzzle.id,
            "crossword.entries": len(puzzle.entries),
        },
    ) as graph_span:
        graph = build_agent_graph(
            provider,
            lexicon=lexicon,
            settings=settings,
            observer=observer,
        )
        latest: AgentState | None = None
        initial: AgentState = {"run_id": run_id, "puzzle": puzzle}
        async for state in graph.astream(initial, stream_mode="values"):
            latest = state
            if observer is not None and "status" in state:
                await observer(snapshot_from_state(state))
        if latest is None:
            raise RuntimeError("Agent graph produced no state")
        result = snapshot_from_state(latest)
        graph_span.set_attribute("agent.status", result.status.value)
        graph_span.set_attribute("gen_ai.client.call.count", result.metrics.model_calls)
        graph_span.set_attribute(
            "agent.search.hypothesis_count",
            result.metrics.search_hypotheses,
        )
        graph_span.set_attribute(
            "agent.verification.call_count",
            result.metrics.verification_calls,
        )
        graph_span.set_attribute("crossword.entries.filled", len(result.assignment))
        if result.status is SolveStatus.FAILED:
            graph_span.set_status("ERROR", result.error)
        else:
            graph_span.set_status("OK")
        return result


def snapshot_from_state(state: AgentState) -> RunSnapshot:
    puzzle = state["puzzle"]
    assignment = state.get("assignment", {})
    status = state.get("status", SolveStatus.PENDING)
    return RunSnapshot(
        run_id=state["run_id"],
        puzzle_id=puzzle.id,
        status=status,
        grid=render_grid(puzzle, assignment, tolerate_conflicts=True),
        assignment={key: candidate.answer for key, candidate in assignment.items()},
        candidate_pool={
            entry_id: tuple(candidate.answer for candidate in candidates)
            for entry_id, candidates in (
                state.get("domains") or state.get("candidate_pool", {})
            ).items()
        },
        unresolved_entries=tuple(
            entry.id for entry in puzzle.entries if entry.id not in assignment
        ),
        events=tuple(state.get("events", [])[-30:]),
        metrics=state.get("metrics", SolveMetrics()),
        error=state.get("error"),
    )


def _metrics_with_provider_usage(
    state: AgentState,
    provider: CandidateProvider,
) -> SolveMetrics:
    metrics = state["metrics"].model_copy(deep=True)
    usage = provider.usage
    metrics.model_calls = usage.calls
    metrics.input_tokens = usage.input_tokens
    metrics.output_tokens = usage.output_tokens
    metrics.model_latency_ms = usage.latency_ms
    metrics.estimated_cost_usd = usage.estimated_cost_usd
    return metrics


async def _generate_with_deadline(
    provider: CandidateProvider,
    requests: list[ClueRequest],
    *,
    deadline: float,
) -> dict[str, list[Candidate]]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Model request exceeded the solve deadline")
    try:
        async with asyncio.timeout(remaining):
            return await provider.generate_candidates(requests)
    except TimeoutError as error:
        raise TimeoutError("Model request exceeded the solve deadline") from error


def _lexical_candidates(words: list[str], lexicon: PatternLexicon) -> list[Candidate]:
    return [
        Candidate(
            answer=word,
            rank=rank,
            lexical_score=lexicon.frequency_score(word),
            source="lexicon",
        )
        for rank, word in enumerate(words, start=1)
    ]


def _entry_priority(
    puzzle: PuzzleDefinition,
    entry_id: str,
    *,
    assignment: dict[str, Candidate],
    domains: dict[str, list[Candidate]],
    hypotheses: tuple[SearchHypothesis, ...],
    attempts: int,
    unresolved: bool,
) -> dict[str, float]:
    assigned = assignment.get(entry_id)
    uncertainty = 1.0 - assigned.score if assigned else 1.0
    hypothesis_answers = [
        hypothesis.assignment[entry_id].answer
        for hypothesis in hypotheses
        if entry_id in hypothesis.assignment
    ]
    disagreement = _normalized_entropy(hypothesis_answers)
    information_gain = _crossing_information_gain(puzzle, entry_id, domains)
    conflict_pressure = _frontier_conflict_pressure(
        puzzle,
        entry_id,
        assignment=assignment,
        domains=domains,
    )
    degree = sum(
        entry_id in {intersection.entry_a, intersection.entry_b}
        for intersection in puzzle.intersections
    )
    max_degree = max(
        (
            sum(
                entry.id in {intersection.entry_a, intersection.entry_b}
                for intersection in puzzle.intersections
            )
            for entry in puzzle.entries
        ),
        default=1,
    )
    influence = degree / max(max_degree, 1)
    pattern = entry_pattern(puzzle, entry_id, assignment)
    pattern_information = sum(char != "?" for char in pattern) / len(pattern)
    sparse_domain = 1.0 / (1.0 + len(domains.get(entry_id, [])))
    priority = (
        0.34 * uncertainty
        + 0.24 * disagreement
        + 0.20 * information_gain
        + 0.24 * conflict_pressure
        + 0.12 * influence
        + 0.10 * (pattern_information if unresolved else sparse_domain)
        - min(0.30, attempts * 0.08)
    )
    return {
        "priority": round(priority, 4),
        "uncertainty": round(uncertainty, 4),
        "hypothesis_disagreement": round(disagreement, 4),
        "expected_information_gain": round(information_gain, 4),
        "frontier_conflict_pressure": round(conflict_pressure, 4),
    }


def _normalized_entropy(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    entropy = -sum(
        (count / len(values)) * math.log(count / len(values))
        for count in counts.values()
    )
    maximum = math.log(min(len(values), len(counts)))
    return entropy / maximum if maximum else 0.0


def _crossing_information_gain(
    puzzle: PuzzleDefinition,
    entry_id: str,
    domains: dict[str, list[Candidate]],
) -> float:
    crossing_positions = {
        intersection.index_a
        if intersection.entry_a == entry_id
        else intersection.index_b
        for intersection in puzzle.intersections
        if entry_id in {intersection.entry_a, intersection.entry_b}
    }
    candidates = domains.get(entry_id, [])
    if len(candidates) < 2 or not crossing_positions:
        return 0.0
    entropies = [
        _normalized_entropy(
            [
                candidate.answer[position]
                for candidate in candidates
                if position < len(candidate.answer)
            ]
        )
        for position in crossing_positions
    ]
    return sum(entropies) / len(entropies)


def _frontier_conflict_pressure(
    puzzle: PuzzleDefinition,
    entry_id: str,
    *,
    assignment: dict[str, Candidate],
    domains: dict[str, list[Candidate]],
) -> float:
    current = assignment.get(entry_id)
    if current is None:
        return 0.0
    pressures = []
    for intersection in puzzle.intersections:
        if intersection.entry_a == entry_id:
            current_position = intersection.index_a
            other_id = intersection.entry_b
            other_position = intersection.index_b
        elif intersection.entry_b == entry_id:
            current_position = intersection.index_b
            other_id = intersection.entry_a
            other_position = intersection.index_a
        else:
            continue
        if other_id in assignment:
            continue
        alternatives = domains.get(other_id, [])
        if not alternatives:
            continue
        disagreements = sum(
            candidate.answer[other_position] != current.answer[current_position]
            for candidate in alternatives
        )
        pressures.append(disagreements / len(alternatives))
    return max(pressures, default=0.0)


def _trusted_pattern(
    puzzle: PuzzleDefinition,
    entry_id: str,
    *,
    assignment: dict[str, Candidate],
    hypotheses: tuple[SearchHypothesis, ...],
    minimum_score: float,
    force_relax: bool,
) -> str:
    entry = puzzle.entry_map[entry_id]
    letters = ["?"] * entry.length
    trusted: list[tuple[float, int]] = []
    for intersection in puzzle.intersections:
        if intersection.entry_a == entry_id:
            position = intersection.index_a
            other_id = intersection.entry_b
            other_position = intersection.index_b
        elif intersection.entry_b == entry_id:
            position = intersection.index_b
            other_id = intersection.entry_a
            other_position = intersection.index_a
        else:
            continue
        other = assignment.get(other_id)
        if other is None:
            continue
        hypothesis_letters = {
            hypothesis.assignment[other_id].answer[other_position]
            for hypothesis in hypotheses
            if other_id in hypothesis.assignment
        }
        stable = len(hypothesis_letters) <= 1
        if stable and other.score >= minimum_score:
            letters[position] = other.answer[other_position]
            trusted.append((other.score, position))
    if force_relax and trusted:
        _score, weakest_position = min(trusted)
        letters[weakest_position] = "?"
    return "".join(letters)


def _crossing_context(
    puzzle: PuzzleDefinition,
    entry_id: str,
    assignment: dict[str, Candidate],
) -> tuple[str, ...]:
    context = []
    for intersection in puzzle.intersections:
        if intersection.entry_a == entry_id:
            position = intersection.index_a
            other_id = intersection.entry_b
            other_position = intersection.index_b
        elif intersection.entry_b == entry_id:
            position = intersection.index_b
            other_id = intersection.entry_a
            other_position = intersection.index_a
        else:
            continue
        other = assignment.get(other_id)
        if other is None:
            continue
        other_entry = puzzle.entry_map[other_id]
        context.append(
            f"letter {position + 1} may be {other.answer[other_position]} from "
            f"{other_id} ({other_entry.clue}) = {other.answer}; "
            f"crossing support {other.score:.2f}"
        )
    return tuple(context)
