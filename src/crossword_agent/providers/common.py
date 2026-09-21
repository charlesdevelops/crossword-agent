from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from crossword_agent.models import (
    Candidate,
    CandidateBatch,
    ClueRequest,
    UsageMetrics,
)
from crossword_agent.tracing import start_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You solve standard American-style crossword clues.
Return candidate answers only through the provided schema.
Answers must contain letters only, match the requested length, and fit every known pattern letter.
Apply crossword conventions: abbreviations, plurals, tense, wordplay, proper names,
and fill-in-the-blank grammar.
Treat crossing context as useful but fallible evidence.
For verification requests, independently check the current answer and replace it when
a listed or new answer is better.
Return a diverse ranked list rather than near-duplicates. Rank the most likely answer first.
Do not provide hidden reasoning or explanatory prose."""


def requests_payload(requests: list[ClueRequest]) -> str:
    maximum = max((request.candidate_limit for request in requests), default=8)
    return json.dumps(
        {
            "task": (
                f"Return up to {maximum} ranked candidate answers for every entry. "
                "Do not omit an entry merely because the clue is uncertain."
            ),
            "entries": [
                {
                    "entry_id": request.entry_id,
                    "clue": request.clue,
                    "length": request.length,
                    "pattern": request.pattern,
                    "strategy": request.strategy,
                    "current_answer": request.current_answer,
                    "alternative_answers": list(request.alternative_answers),
                    "crossing_context": list(request.crossing_context),
                    "rejected_answers": list(request.rejected_answers),
                    "lexical_options": list(request.lexical_options),
                    "candidate_limit": request.candidate_limit,
                }
                for request in requests
            ],
        },
        separators=(",", ":"),
    )


async def invoke_structured_model(
    *,
    invoke: Callable[[list[Any]], Awaitable[Any]],
    requests: list[ClueRequest],
    usage: UsageMetrics,
    provider_name: str = "unknown",
    model_id: str = "unknown",
) -> dict[str, list[Candidate]]:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=requests_payload(requests)),
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        started = time.perf_counter()
        usage.calls += 1
        with start_span(
            "gen_ai.generate_candidates",
            kind="CLIENT",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider_name,
                "gen_ai.request.model": model_id,
                "gen_ai.request.entry_count": len(requests),
                "retry.attempt": attempt + 1,
            },
        ) as model_span:
            try:
                result = await invoke(messages)
                usage.latency_ms += (time.perf_counter() - started) * 1000
                parsed, raw = _unwrap_result(result)
                grouped: dict[str, list[Candidate]] = defaultdict(list)
                requests_by_id = {request.entry_id: request for request in requests}
                for item in parsed.candidates:
                    request = requests_by_id.get(item.entry_id)
                    if request is None:
                        continue
                    if len(grouped[item.entry_id]) >= request.candidate_limit:
                        continue
                    grouped[item.entry_id].append(
                        Candidate(
                            answer=item.answer,
                            rank=len(grouped[item.entry_id]) + 1,
                            source=(
                                "verifier"
                                if request.strategy == "verify"
                                else "model"
                            ),
                        )
                    )
                input_tokens, output_tokens = _add_message_usage(usage, raw)
                model_span.set_attribute(
                    "gen_ai.response.candidate_count",
                    sum(len(candidates) for candidates in grouped.values()),
                )
                model_span.set_attribute(
                    "gen_ai.usage.input_tokens",
                    input_tokens,
                )
                model_span.set_attribute(
                    "gen_ai.usage.output_tokens",
                    output_tokens,
                )
                model_span.set_status("OK")
                return dict(grouped)
            except Exception as error:  # noqa: BLE001 - provider exceptions vary
                usage.latency_ms += (time.perf_counter() - started) * 1000
                last_error = error
                model_span.record_exception(error)
                model_span.set_status("ERROR", str(error))
                logger.warning(
                    "Model request attempt failed",
                    exc_info=True,
                    extra={
                        "provider": provider_name,
                        "attempt": attempt + 1,
                        "request_count": len(requests),
                    },
                )
                messages = [
                    *messages,
                    HumanMessage(
                        content=(
                            "The previous response was invalid. Return only schema-valid "
                            "candidates for these entry IDs: "
                            f"{sorted(request.entry_id for request in requests)}."
                        )
                    ),
                ]
                if attempt == 0:
                    continue
    raise RuntimeError(
        f"Model request failed after two attempts: {last_error}"
    ) from last_error


def _unwrap_result(result: Any) -> tuple[CandidateBatch, Any]:
    if isinstance(result, CandidateBatch):
        return result, None
    if isinstance(result, dict) and "parsed" in result:
        parsed = result["parsed"]
        if parsed is None:
            raise ValueError(result.get("parsing_error") or "Structured output was empty")
        if not isinstance(parsed, CandidateBatch):
            parsed = CandidateBatch.model_validate(parsed)
        return parsed, result.get("raw")
    return CandidateBatch.model_validate(result), None


def _add_message_usage(usage: UsageMetrics, raw: Any) -> tuple[int, int]:
    if raw is None:
        return 0, 0
    metadata = getattr(raw, "usage_metadata", None) or {}
    response_metadata = getattr(raw, "response_metadata", None) or {}
    provider_usage = (
        response_metadata.get("usage", {}) if isinstance(response_metadata, dict) else {}
    )
    input_tokens = int(
        metadata.get("input_tokens", provider_usage.get("input_tokens", 0)) or 0
    )
    output_tokens = int(
        metadata.get("output_tokens", provider_usage.get("output_tokens", 0)) or 0
    )
    usage.input_tokens += input_tokens
    usage.output_tokens += output_tokens
    input_price = float(os.getenv("INPUT_COST_PER_MTOK", "0"))
    output_price = float(os.getenv("OUTPUT_COST_PER_MTOK", "0"))
    if input_price or output_price:
        usage.estimated_cost_usd = (
            usage.input_tokens * input_price + usage.output_tokens * output_price
        ) / 1_000_000
    return input_tokens, output_tokens
