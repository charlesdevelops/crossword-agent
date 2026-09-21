from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from importlib.resources import files
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from crossword_agent.logging_config import configure_logging
from crossword_agent.models import (
    AgentEvent,
    Direction,
    PublicPuzzle,
    RunSnapshot,
    SolveStatus,
    public_puzzle,
)
from crossword_agent.puzzles import PuzzleRepository
from crossword_agent.storage import (
    DailyQuotaExceededError,
    RunBusyError,
    RunRecord,
    get_run_store,
)
from crossword_agent.tracing import (
    configure_tracing,
    current_trace_context,
    extract_traceparent,
    format_traceparent,
    start_span,
)
from crossword_agent.worker import process_run

configure_logging()
configure_tracing()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crossword Agent",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
repository = PuzzleRepository.configured()
TERMINAL_STATUSES = frozenset(
    {
        SolveStatus.SOLVED,
        SolveStatus.STALLED,
        SolveStatus.FAILED,
        SolveStatus.BUDGET_EXCEEDED,
    }
)


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puzzle_id: str


class StartRunResponse(BaseModel):
    run_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    provider: str


class AnswerReviewEntry(BaseModel):
    entry_id: str
    number: int
    direction: Direction
    agent_answer: str | None
    correct_answer: str
    correct: bool


class AnswerReviewResponse(BaseModel):
    correct_entries: int
    total_entries: int
    accuracy: float
    entries: tuple[AnswerReviewEntry, ...]


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.perf_counter()
    with start_span(
        "http.request",
        kind="SERVER",
        parent=extract_traceparent(request.headers),
        attributes={
            "http.request.method": request.method,
            "url.path": request.url.path,
            "request.id": request_id,
        },
    ) as span:
        context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        }
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Unhandled API request failure",
                extra={
                    **context,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        span.set_attribute("http.response.status_code", response.status_code)
        span.set_attribute("http.server.request.duration_ms", duration_ms)
        if response.status_code >= 500:
            span.set_status("ERROR", f"HTTP {response.status_code}")
        else:
            span.set_status("OK")
        logger.info(
            "HTTP request completed",
            extra={
                **context,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["traceparent"] = format_traceparent(span.context)
        return response


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return files("crossword_agent").joinpath("static/index.html").read_text(encoding="utf-8")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", provider=os.getenv("LLM_PROVIDER", "bedrock"))


@app.get("/api/puzzles/random", response_model=PublicPuzzle)
async def random_puzzle(
    exclude: Annotated[str | None, Query(max_length=64)] = None,
) -> PublicPuzzle:
    return public_puzzle(repository.random(exclude=exclude).puzzle)


@app.post("/api/runs", response_model=StartRunResponse, status_code=202)
async def start_run(
    request: StartRunRequest,
    background_tasks: BackgroundTasks,
) -> StartRunResponse:
    try:
        repository.get(request.puzzle_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown puzzle ID") from error
    store = get_run_store(repository=repository)
    try:
        record = store.create_run(request.puzzle_id)
    except RunBusyError as error:
        logger.warning(
            "Run rejected because another solve is active",
            extra={"puzzle_id": request.puzzle_id},
        )
        raise HTTPException(
            status_code=429,
            detail={"message": str(error), "retry_after": 5},
            headers={"Retry-After": "5"},
        ) from error
    except DailyQuotaExceededError as error:
        logger.warning(
            "Run rejected because the daily quota was reached",
            extra={"puzzle_id": request.puzzle_id},
        )
        raise HTTPException(
            status_code=429,
            detail={"message": str(error), "retry_after": 3600},
            headers={"Retry-After": "3600"},
        ) from error
    logger.info(
        "Run created",
        extra={"run_id": record.run_id, "puzzle_id": record.puzzle_id},
    )
    try:
        _dispatch_run(record, background_tasks)
    except Exception as error:  # noqa: BLE001
        logger.exception(
            "Unable to dispatch crossword worker",
            extra={"run_id": record.run_id, "puzzle_id": record.puzzle_id},
        )
        failed = record.snapshot.model_copy(
            update={
                "status": SolveStatus.FAILED,
                "error": "Unable to dispatch the crossword worker",
                "events": (
                    AgentEvent(phase="dispatch", message="Worker dispatch failed"),
                ),
            }
        )
        store.update_snapshot(failed)
        store.release_run(record.run_id)
        raise HTTPException(status_code=503, detail="Worker unavailable") from error
    return StartRunResponse(run_id=record.run_id, status=record.status.value)


@app.get("/api/runs/{run_id}", response_model=RunSnapshot)
async def get_run(run_id: str) -> RunSnapshot:
    record = _get_run_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record.snapshot


@app.get("/api/runs/{run_id}/answers", response_model=AnswerReviewResponse)
async def answer_review(run_id: str) -> AnswerReviewResponse:
    run = _get_run_record(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="Correct answers are available after the run finishes",
        )

    puzzle_record = repository.get(run.puzzle_id)
    if puzzle_record.solution is None:
        raise HTTPException(status_code=404, detail="Correct answers are unavailable")

    entries = tuple(
        _answer_review_entry(
            entry=entry,
            solution=puzzle_record.solution,
            agent_answer=run.snapshot.assignment.get(entry.id),
        )
        for entry in puzzle_record.puzzle.entries
    )
    correct_entries = sum(entry.correct for entry in entries)
    return AnswerReviewResponse(
        correct_entries=correct_entries,
        total_entries=len(entries),
        accuracy=correct_entries / len(entries) if entries else 0.0,
        entries=entries,
    )


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    if _get_run_record(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def snapshots():
        last_payload: str | None = None
        while not await request.is_disconnected():
            record = _get_run_record(run_id)
            if record is None:
                return
            payload = record.snapshot.model_dump_json()
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if record.status in TERMINAL_STATUSES:
                return
            await asyncio.sleep(0.15)

    return StreamingResponse(
        snapshots(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _get_run_record(run_id: str) -> RunRecord | None:
    if len(run_id) > 64:
        return None
    return get_run_store(repository=repository).get_run(run_id)


def _answer_review_entry(*, entry, solution, agent_answer) -> AnswerReviewEntry:
    correct_answer = "".join(solution[row][col] for row, col in entry.cells)
    return AnswerReviewEntry(
        entry_id=entry.id,
        number=entry.number,
        direction=entry.direction,
        agent_answer=agent_answer,
        correct_answer=correct_answer,
        correct=agent_answer == correct_answer,
    )


def _dispatch_run(record: RunRecord, background_tasks: BackgroundTasks) -> None:
    background_tasks.add_task(
        process_run,
        record.run_id,
        store=get_run_store(repository=repository),
        parent_trace=current_trace_context(),
    )
