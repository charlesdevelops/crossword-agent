from __future__ import annotations

import pytest

from crossword_agent.constraints import (
    count_constraint_violations,
    entry_pattern,
    render_grid,
    search_assignments,
    validate_domain,
)
from crossword_agent.models import Candidate
from crossword_agent.parser import build_puzzle_record


def _gold_answers(record) -> dict[str, str]:
    return {
        entry.id: "".join(record.solution[row][col] for row, col in entry.cells)
        for entry in record.puzzle.entries
    }


def test_validate_domain_enforces_length_pattern_and_deduplicates(demo_record) -> None:
    entry = demo_record.puzzle.entry_map["1A"]
    validated = validate_domain(
        entry,
        [
            Candidate(answer="cat", rank=2),
            Candidate(answer="CAT", rank=1),
            Candidate(answer="CAR", rank=3),
            Candidate(answer="TOOLONG"),
        ],
        pattern="CA?",
    )

    assert [candidate.answer for candidate in validated] == ["CAT", "CAR"]
    assert validated[0].rank == 1


def test_render_grid_can_display_conflicting_intermediate_assignments(demo_record) -> None:
    conflicting = {
        "1A": Candidate(answer="CAT"),
        "1D": Candidate(answer="DOG"),
    }

    with pytest.raises(ValueError, match="Inconsistent assignment"):
        render_grid(demo_record.puzzle, conflicting)

    rendered = render_grid(
        demo_record.puzzle,
        conflicting,
        tolerate_conflicts=True,
    )

    assert "?" in "".join(rendered)


def test_weighted_search_finds_complete_consistent_solution(demo_record) -> None:
    gold = _gold_answers(demo_record)
    domains = {
        entry_id: [Candidate(answer=answer)]
        for entry_id, answer in gold.items()
    }

    result = search_assignments(demo_record.puzzle, domains)

    assert result.complete
    assert result.words == gold
    assert render_grid(demo_record.puzzle, result.words) == demo_record.solution
    assert count_constraint_violations(demo_record.puzzle, result.words) == 0


def test_search_returns_best_partial_when_candidate_is_missing(demo_record) -> None:
    gold = _gold_answers(demo_record)
    domains = {
        entry_id: [Candidate(answer=answer)]
        for entry_id, answer in gold.items()
        if entry_id != "3D"
    }

    result = search_assignments(demo_record.puzzle, domains)

    assert not result.complete
    assert len(result.assignment) == len(demo_record.puzzle.entries) - 1
    assert entry_pattern(demo_record.puzzle, "3D", result.assignment) == "TEN"


def test_tie_breaking_is_deterministic(demo_record) -> None:
    entry = demo_record.puzzle.entry_map["1A"]
    domains = {
        item.id: [
            Candidate(
                answer="".join(demo_record.solution[r][c] for r, c in item.cells),
            )
        ]
        for item in demo_record.puzzle.entries
    }
    domains[entry.id] = [
        Candidate(answer="CAT"),
        Candidate(answer="CAR"),
    ]

    first = search_assignments(demo_record.puzzle, domains)
    second = search_assignments(demo_record.puzzle, domains)

    assert first.words == second.words


def test_search_retains_ranked_complete_hypotheses() -> None:
    record = build_puzzle_record(
        puzzle_id="word-squares",
        title="Word squares",
        solution_rows=("AB", "CD"),
        across_clues={1: "First row", 3: "Second row"},
        down_clues={1: "First column", 2: "Second column"},
    )
    domains = {
        "1A": [
            Candidate(answer="AB"),
            Candidate(answer="EF"),
        ],
        "3A": [
            Candidate(answer="CD"),
            Candidate(answer="GH"),
        ],
        "1D": [
            Candidate(answer="AC"),
            Candidate(answer="EG"),
        ],
        "2D": [
            Candidate(answer="BD"),
            Candidate(answer="FH"),
        ],
    }

    result = search_assignments(record.puzzle, domains, n_best=2)

    assert len(result.hypotheses) == 2
    assert all(hypothesis.complete for hypothesis in result.hypotheses)
    assert result.hypotheses[0].words == {
        "1A": "AB",
        "3A": "CD",
        "1D": "AC",
        "2D": "BD",
    }
    assert result.hypotheses[1].words == {
        "1A": "EF",
        "3A": "GH",
        "1D": "EG",
        "2D": "FH",
    }
