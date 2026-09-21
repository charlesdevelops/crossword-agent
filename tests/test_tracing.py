from __future__ import annotations

import json
import logging

import pytest

from crossword_agent.agent import solve_puzzle
from crossword_agent.logging_config import configure_logging
from crossword_agent.models import (
    Candidate,
    CandidateAnswer,
    CandidateBatch,
    ClueRequest,
    UsageMetrics,
)
from crossword_agent.providers.common import invoke_structured_model
from crossword_agent.providers.fake import ScriptedCandidateProvider
from crossword_agent.tracing import (
    configure_tracing,
    extract_traceparent,
    force_flush_traces,
    format_traceparent,
    start_span,
)


def _read_spans(path):
    force_flush_traces()
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_nested_spans_share_trace_and_record_exceptions(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    configure_tracing(trace_file=trace_file, force=True)

    with start_span("parent") as parent:
        with start_span("child") as child:
            pass
        with pytest.raises(ValueError), start_span("failing-child"):
            raise ValueError("trace me")

    spans = {span["name"]: span for span in _read_spans(trace_file)}
    assert spans["child"]["trace_id"] == spans["parent"]["trace_id"]
    assert spans["child"]["parent_span_id"] == parent.context.span_id
    assert spans["failing-child"]["status"]["code"] == "ERROR"
    assert spans["failing-child"]["exception"]["type"] == "ValueError"
    assert "trace me" in spans["failing-child"]["exception"]["stacktrace"]
    assert child.context.trace_id == parent.context.trace_id

    configure_tracing(force=True)


def test_traceparent_round_trip() -> None:
    context = extract_traceparent(
        {
            "traceparent": (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-"
                "00f067aa0ba902b7-01"
            )
        }
    )

    assert context is not None
    assert format_traceparent(context) == (
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )


async def test_agent_graph_emits_nested_phase_spans(demo_record, tmp_path) -> None:
    trace_file = tmp_path / "agent-traces.jsonl"
    configure_tracing(trace_file=trace_file, force=True)
    gold = {
        entry.id: "".join(demo_record.solution[row][col] for row, col in entry.cells)
        for entry in demo_record.puzzle.entries
    }
    provider = ScriptedCandidateProvider(
        {
            entry_id: [[Candidate(answer=answer)]]
            for entry_id, answer in gold.items()
        }
    )

    await solve_puzzle(
        run_id="trace-test",
        puzzle=demo_record.puzzle,
        provider=provider,
    )

    spans = _read_spans(trace_file)
    graph = next(span for span in spans if span["name"] == "agent.graph")
    phase_spans = [span for span in spans if span["name"].startswith("agent.")]
    names = {span["name"] for span in phase_spans}
    assert {"agent.initialize", "agent.generate", "agent.search", "agent.finalize"} <= names
    assert all(span["trace_id"] == graph["trace_id"] for span in phase_spans)
    assert all(span["duration_ms"] >= 0 for span in phase_spans)

    configure_tracing(force=True)


async def test_provider_calls_are_nested_generation_spans(tmp_path) -> None:
    trace_file = tmp_path / "provider-traces.jsonl"
    configure_tracing(trace_file=trace_file, force=True)

    async def invoke(_messages):
        return CandidateBatch(
            candidates=[
                CandidateAnswer(entry_id="1A", answer="CAT"),
            ]
        )

    with start_span("agent.generate") as parent:
        await invoke_structured_model(
            invoke=invoke,
            requests=[ClueRequest(entry_id="1A", clue="Pet", length=3, pattern="???")],
            usage=UsageMetrics(),
            provider_name="test-provider",
            model_id="test-model",
        )

    spans = _read_spans(trace_file)
    generation = next(
        span for span in spans if span["name"] == "gen_ai.generate_candidates"
    )
    assert generation["parent_span_id"] == parent.context.span_id
    assert generation["attributes"]["gen_ai.provider.name"] == "test-provider"
    assert generation["attributes"]["gen_ai.request.model"] == "test-model"
    assert generation["attributes"]["gen_ai.response.candidate_count"] == 1

    configure_tracing(force=True)


def test_logs_receive_active_trace_and_span_ids(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    log_file = tmp_path / "logs.jsonl"
    configure_tracing(trace_file=trace_file, force=True)
    configure_logging(log_file=log_file, force=True)
    logger = logging.getLogger("crossword_agent.trace-test")

    with start_span("correlated") as span:
        logger.info("Inside a trace")
    for handler in logging.getLogger("crossword_agent").handlers:
        handler.flush()

    payload = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["trace_id"] == span.context.trace_id
    assert payload["span_id"] == span.context.span_id

    configure_logging(force=True)
    configure_tracing(force=True)
