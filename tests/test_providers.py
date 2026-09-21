from __future__ import annotations

import json

import pytest

from crossword_agent.models import CandidateAnswer, CandidateBatch, ClueRequest, UsageMetrics
from crossword_agent.providers.common import invoke_structured_model, requests_payload


async def test_structured_provider_groups_and_caps_candidates() -> None:
    async def invoke(_messages):
        return CandidateBatch(
            candidates=[
                CandidateAnswer(entry_id="1A", answer=f"CAT{index}")
                for index in range(7)
            ]
        )

    usage = UsageMetrics()
    result = await invoke_structured_model(
        invoke=invoke,
        requests=[
            ClueRequest(entry_id="1A", clue="Pet", length=3, pattern="???"),
        ],
        usage=usage,
    )

    assert len(result["1A"]) == 5
    assert usage.calls == 1


async def test_structured_provider_uses_rank_without_model_confidence() -> None:
    async def invoke(_messages):
        return CandidateBatch(
            candidates=[
                CandidateAnswer(entry_id="1A", answer="CAT"),
                CandidateAnswer(entry_id="1A", answer="DOG"),
            ]
        )

    result = await invoke_structured_model(
        invoke=invoke,
        requests=[ClueRequest(entry_id="1A", clue="Pet", length=3, pattern="???")],
        usage=UsageMetrics(),
    )

    assert [candidate.rank for candidate in result["1A"]] == [1, 2]
    assert [candidate.source for candidate in result["1A"]] == ["model", "model"]


def test_provider_schema_does_not_request_confidence() -> None:
    schema = json.dumps(CandidateBatch.model_json_schema())

    assert "confidence" not in schema
    assert '"answer"' in schema
    assert "clue_answer" not in schema


def test_provider_accepts_common_clue_answer_alias() -> None:
    batch = CandidateBatch.model_validate(
        {
            "candidates": [
                {"entry_id": "9D", "clue_answer": "ARM"},
            ]
        }
    )

    assert batch.candidates[0].answer == "ARM"
    assert batch.model_dump(mode="json", by_alias=True) == {
        "candidates": [{"entry_id": "9D", "answer": "ARM"}]
    }


def test_provider_still_rejects_records_without_an_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        CandidateBatch.model_validate(
            {
                "candidates": [
                    {"entry_id": "1D", "clue": "End of a professor's address?"},
                ]
            }
        )


def test_provider_keeps_valid_candidates_from_a_partially_malformed_batch() -> None:
    batch = CandidateBatch.model_validate(
        {
            "candidates": [
                {"entry_id": "9D", "clue_answer": "ARM"},
                {"entry_id": "1D", "clue": "End of a professor's address?"},
            ]
        }
    )

    assert [(item.entry_id, item.answer) for item in batch.candidates] == [("9D", "ARM")]


async def test_structured_provider_retries_once() -> None:
    calls = 0

    async def invoke(_messages):
        nonlocal calls
        calls += 1
        raise ValueError("invalid")

    usage = UsageMetrics()
    with pytest.raises(RuntimeError, match="Model request failed after two attempts: invalid"):
        await invoke_structured_model(
            invoke=invoke,
            requests=[
                ClueRequest(entry_id="1A", clue="Pet", length=3, pattern="???"),
            ],
            usage=usage,
        )

    assert calls == 2
    assert usage.calls == 2


def test_verification_prompt_contains_current_answer_and_crossing_context() -> None:
    payload = json.loads(
        requests_payload(
            [
                ClueRequest(
                    entry_id="6D",
                    clue="Movie dog",
                    length=4,
                    pattern="?A??",
                    strategy="verify",
                    current_answer="LASS",
                    alternative_answers=("LADY",),
                    crossing_context=("letter 2 may be A from 5A",),
                    candidate_limit=8,
                )
            ]
        )
    )

    entry = payload["entries"][0]
    assert entry["strategy"] == "verify"
    assert entry["current_answer"] == "LASS"
    assert entry["crossing_context"] == ["letter 2 may be A from 5A"]
    assert entry["candidate_limit"] == 8
