from __future__ import annotations

import json

from crossword_agent.benchmark import (
    select_benchmark_records,
    write_benchmark_dashboard,
)
from crossword_agent.evaluation import evaluate_records
from crossword_agent.models import Candidate
from crossword_agent.providers.fake import ScriptedCandidateProvider


def _gold_answers(record) -> dict[str, str]:
    return {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }


def test_benchmark_selection_is_seeded_and_requires_enough_records(demo_record) -> None:
    records = [
        demo_record.model_copy(
            update={
                "puzzle": demo_record.puzzle.model_copy(update={"id": f"puzzle-{index}"})
            },
            deep=True,
        )
        for index in range(25)
    ]

    first = select_benchmark_records(records, count=20, seed=42)
    second = select_benchmark_records(records, count=20, seed=42)

    assert len(first) == 20
    assert [record.puzzle.id for record in first] == [
        record.puzzle.id for record in second
    ]


async def test_html_dashboard_contains_requested_metrics(
    demo_record,
    tmp_path,
) -> None:
    gold = _gold_answers(demo_record)

    def provider_factory() -> ScriptedCandidateProvider:
        return ScriptedCandidateProvider(
            {
                entry_id: [[Candidate(answer=answer)]]
                for entry_id, answer in gold.items()
            }
        )

    results = await evaluate_records(
        [demo_record],
        provider_factory=provider_factory,
        mode="full",
    )
    dashboard = tmp_path / "benchmark.html"
    write_benchmark_dashboard(
        dashboard,
        results=results,
        dataset="prepared.json",
        provider="fake",
        model="scripted",
        seed=42,
        requested_puzzles=20,
    )

    html = dashboard.read_text(encoding="utf-8")
    raw = json.loads(dashboard.with_suffix(".json").read_text(encoding="utf-8"))
    assert "Puzzle solve rate" in html
    assert "Conflict recovery" in html
    assert "Candidate recall" in html
    assert "Oracle solve rate" in html
    assert "Cost / successful solve" in html
    assert "Per-puzzle results" in html
    assert 'href="benchmark.json"' in html
    assert raw["metadata"]["completed_puzzles"] == 1
    assert raw["summary"]["full_puzzle_solve_rate"] == 1.0
