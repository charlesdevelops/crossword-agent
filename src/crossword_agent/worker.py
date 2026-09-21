from __future__ import annotations

import logging
import time

from crossword_agent.agent import AgentSettings, solve_puzzle
from crossword_agent.models import AgentEvent, RunSnapshot, SolveMetrics, SolveStatus
from crossword_agent.runtime import create_clue_index, create_lexicon, create_provider
from crossword_agent.storage import RunStore, get_run_store
from crossword_agent.tracing import TraceContext, start_span

logger = logging.getLogger(__name__)


async def process_run(
    run_id: str,
    *,
    store: RunStore | None = None,
    parent_trace: TraceContext | None = None,
) -> RunSnapshot | None:
    resolved_store = store or get_run_store()
    record = resolved_store.get_run(run_id)
    if record is None or not resolved_store.claim_run(run_id):
        return None
    puzzle = resolved_store.get_puzzle(record.puzzle_id)
    started = time.monotonic()
    with start_span(
        "agent.solve",
        parent=parent_trace,
        attributes={
            "agent.run.id": run_id,
            "crossword.puzzle.id": puzzle.id,
            "crossword.entries": len(puzzle.entries),
            "crossword.intersections": len(puzzle.intersections),
        },
    ) as run_span:
        last_event_signature: tuple[str, str, str | None] | None = None
        logger.info(
            "Run started",
            extra={"run_id": run_id, "puzzle_id": puzzle.id},
        )

        async def observer(snapshot: RunSnapshot) -> None:
            nonlocal last_event_signature
            resolved_store.update_snapshot(snapshot)
            if not snapshot.events:
                return
            event = snapshot.events[-1]
            signature = (event.phase, event.message, event.entry_id)
            if signature == last_event_signature:
                return
            last_event_signature = signature
            run_span.add_event(
                f"agent.{event.phase}",
                {
                    "agent.event.message": event.message,
                    "crossword.entry.id": event.entry_id,
                    "agent.status": snapshot.status.value,
                    "gen_ai.client.call.count": snapshot.metrics.model_calls,
                },
            )
            logger.info(
                "Agent state updated",
                extra={
                    "run_id": run_id,
                    "puzzle_id": puzzle.id,
                    "entry_id": event.entry_id,
                    "phase": event.phase,
                    "event_message": event.message,
                    "status": snapshot.status.value,
                    "model_calls": snapshot.metrics.model_calls,
                },
            )

        try:
            result = await solve_puzzle(
                run_id=run_id,
                puzzle=puzzle,
                provider=create_provider(),
                lexicon=create_lexicon(),
                clue_index=create_clue_index(),
                settings=AgentSettings(),
                observer=observer,
            )
            resolved_store.update_snapshot(result)
            run_span.set_attribute("agent.status", result.status.value)
            run_span.set_attribute("gen_ai.client.call.count", result.metrics.model_calls)
            run_span.set_attribute("agent.duration_ms", round(result.metrics.elapsed_ms, 2))
            run_span.set_attribute(
                "crossword.entries.filled",
                len(result.assignment),
            )
            if result.status is SolveStatus.FAILED:
                run_span.set_status("ERROR", result.error)
            else:
                run_span.set_status("OK")
            logger.info(
                "Run finished",
                extra={
                    "run_id": run_id,
                    "puzzle_id": puzzle.id,
                    "status": result.status.value,
                    "model_calls": result.metrics.model_calls,
                    "duration_ms": round(result.metrics.elapsed_ms, 2),
                },
            )
            return result
        except Exception as error:  # noqa: BLE001
            run_span.record_exception(error)
            run_span.set_status("ERROR", str(error))
            logger.exception(
                "Worker failed before completion",
                extra={"run_id": run_id, "puzzle_id": puzzle.id},
            )
            failed = RunSnapshot(
                run_id=run_id,
                puzzle_id=puzzle.id,
                status=SolveStatus.FAILED,
                grid=puzzle.template,
                unresolved_entries=tuple(entry.id for entry in puzzle.entries),
                events=(
                    AgentEvent(phase="worker", message="Worker failed before completion"),
                ),
                metrics=SolveMetrics(elapsed_ms=(time.monotonic() - started) * 1000),
                error=str(error),
            )
            resolved_store.update_snapshot(failed)
            return failed
        finally:
            resolved_store.release_run(run_id)
