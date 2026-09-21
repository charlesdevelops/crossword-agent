from __future__ import annotations

import asyncio

from crossword_agent.agent import AgentSettings, solve_puzzle
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.models import Candidate, PuzzleDefinition, SolveStatus, UsageMetrics
from crossword_agent.providers.fake import ScriptedCandidateProvider


def _gold_answers(record) -> dict[str, str]:
    return {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }


async def test_agent_recovers_by_reconsidering_weak_crossing(demo_record) -> None:
    gold = _gold_answers(demo_record)
    scripted = {
        entry_id: [[Candidate(answer=answer)]]
        for entry_id, answer in gold.items()
    }
    scripted["1A"] = [
        [Candidate(answer="CAR")],
        [Candidate(answer="CAT")],
    ]
    scripted["3D"] = [
        [],
        [],
        [Candidate(answer="TEN")],
    ]
    provider = ScriptedCandidateProvider(scripted)

    result = await solve_puzzle(
        run_id="test",
        puzzle=demo_record.puzzle,
        provider=provider,
        lexicon=PatternLexicon.from_words(["CAT", "CAR", "TEN"]),
        settings=AgentSettings(max_model_calls=8, deadline_seconds=5),
    )

    assert result.status is SolveStatus.SOLVED
    assert result.grid == demo_record.solution
    assert result.metrics.candidate_replacements >= 1
    assert any(event.phase == "requery" and event.entry_id == "1A" for event in result.events)


async def test_agent_stalls_cleanly_without_candidates(demo_record) -> None:
    provider = ScriptedCandidateProvider({})

    result = await solve_puzzle(
        run_id="empty",
        puzzle=demo_record.puzzle,
        provider=provider,
        settings=AgentSettings(
            max_model_calls=4,
            deadline_seconds=5,
            stagnant_round_limit=2,
        ),
    )

    assert result.status is SolveStatus.STALLED
    assert result.assignment == {}
    assert result.unresolved_entries


async def test_agent_reports_failed_model_attempts(demo_record) -> None:
    class FailingProvider:
        def __init__(self) -> None:
            self.usage = UsageMetrics()

        async def generate_candidates(self, _requests):
            self.usage.calls += 2
            self.usage.latency_ms += 25
            raise RuntimeError("provider unavailable")

    result = await solve_puzzle(
        run_id="failed",
        puzzle=demo_record.puzzle,
        provider=FailingProvider(),
        settings=AgentSettings(deadline_seconds=5),
    )

    assert result.status is SolveStatus.FAILED
    batch_count = (
        len(demo_record.puzzle.entries) + AgentSettings().initial_batch_size - 1
    ) // AgentSettings().initial_batch_size
    assert result.metrics.model_calls == batch_count * 2
    assert result.metrics.model_latency_ms == batch_count * 25
    assert result.metrics.elapsed_ms > 0
    assert result.error == "provider unavailable"


async def test_initial_candidate_batches_run_concurrently_and_stream_progress(
    demo_record,
) -> None:
    gold = _gold_answers(demo_record)
    snapshots = []

    class ConcurrentProvider:
        def __init__(self) -> None:
            self.usage = UsageMetrics()
            self.active_calls = 0
            self.max_active_calls = 0
            self.candidate_limits: list[int] = []

        async def generate_candidates(self, requests):
            self.usage.calls += 1
            self.candidate_limits.extend(request.candidate_limit for request in requests)
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                await asyncio.sleep(0.02)
                return {
                    request.entry_id: [Candidate(answer=gold[request.entry_id])]
                    for request in requests
                }
            finally:
                self.active_calls -= 1

    provider = ConcurrentProvider()
    settings = AgentSettings(
        deadline_seconds=5,
        minimum_verification_calls=0,
    )

    async def observe(snapshot) -> None:
        snapshots.append(snapshot)

    result = await solve_puzzle(
        run_id="concurrent-initial",
        puzzle=demo_record.puzzle,
        provider=provider,
        settings=settings,
        observer=observe,
    )

    batch_count = (
        len(demo_record.puzzle.entries) + settings.initial_batch_size - 1
    ) // settings.initial_batch_size
    progress_messages = [
        snapshot.events[-1].message
        for snapshot in snapshots
        if snapshot.events and snapshot.events[-1].phase == "generate"
    ]

    assert result.status is SolveStatus.SOLVED
    assert provider.max_active_calls == batch_count
    assert provider.max_active_calls > 1
    assert provider.candidate_limits == [
        settings.initial_requested_candidates
    ] * len(demo_record.puzzle.entries)
    assert result.metrics.model_calls == batch_count
    assert any("concurrent batches" in message for message in progress_messages)
    assert (
        sum(
            "Initial batch" in message and "completed" in message
            for message in progress_messages
        )
        == batch_count
    )


async def test_complete_unverified_grid_is_labeled_for_review(demo_record) -> None:
    gold = _gold_answers(demo_record)
    provider = ScriptedCandidateProvider(
        {
            entry_id: [[Candidate(answer=answer)]]
            for entry_id, answer in gold.items()
        }
    )

    result = await solve_puzzle(
        run_id="review",
        puzzle=demo_record.puzzle,
        provider=provider,
        lexicon=PatternLexicon.empty(),
        settings=AgentSettings(max_model_calls=1, deadline_seconds=5),
    )

    assert result.status is SolveStatus.BUDGET_EXCEEDED
    assert len(result.assignment) == len(demo_record.puzzle.entries)
    assert result.events[-1].message == "Grid filled, but targeted verification did not complete"


async def test_agent_rejects_puzzle_without_entries() -> None:
    provider = ScriptedCandidateProvider({})
    puzzle = PuzzleDefinition(
        id="invalid",
        title="Invalid",
        width=3,
        height=3,
        template=("###", "#.#", "###"),
        entries=(),
        intersections=(),
    )

    result = await solve_puzzle(
        run_id="invalid",
        puzzle=puzzle,
        provider=provider,
        settings=AgentSettings(deadline_seconds=5),
    )

    assert result.status is SolveStatus.FAILED
    assert result.metrics.model_calls == 0
    assert result.error == "Puzzle contains no Across or Down entries"


async def test_all_bundled_puzzles_solve_with_semantically_correct_candidates() -> None:
    from crossword_agent.puzzles import PuzzleRepository

    for record in PuzzleRepository.bundled().all():
        gold = _gold_answers(record)
        provider = ScriptedCandidateProvider(
            {
                entry_id: [[Candidate(answer=answer)]]
                for entry_id, answer in gold.items()
            }
        )

        result = await solve_puzzle(
            run_id=f"acceptance-{record.puzzle.id}",
            puzzle=record.puzzle,
            provider=provider,
            settings=AgentSettings(deadline_seconds=5),
        )

        assert result.status is SolveStatus.SOLVED
        assert result.grid == record.solution
        assert result.metrics.constraint_violations == 0
