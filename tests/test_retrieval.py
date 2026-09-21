from __future__ import annotations

import json

import pytest

from crossword_agent.retrieval import ClueAnswerExample, ClueAnswerIndex


def test_clue_index_ranks_exact_and_pattern_compatible_answers() -> None:
    index = ClueAnswerIndex(
        [
            ClueAnswerExample(clue="Movie dog", answer="LASS"),
            ClueAnswerExample(clue="Movie star", answer="HERO"),
            ClueAnswerExample(clue="Dog's warning", answer="BARK"),
        ]
    )

    exact = index.search("Movie dog", length=4)
    patterned = index.search("Movie dog", length=4, pattern="L?S?")

    assert exact[0].answer == "LASS"
    assert patterned == [exact[0]]


def test_clue_index_loads_portable_json(tmp_path) -> None:
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "examples": [
                    {"clue": "Sky sight", "answer": "STAR"},
                ],
            }
        ),
        encoding="utf-8",
    )

    index = ClueAnswerIndex.from_json(path)

    assert len(index) == 1
    assert index.search("Sky sight", length=4)[0].answer == "STAR"


def test_clue_index_rejects_declared_evaluation_splits(tmp_path) -> None:
    path = tmp_path / "leaked-index.json"
    path.write_text(
        json.dumps(
            {
                "allowed_splits": ["evaluation"],
                "examples": [{"clue": "Pet", "answer": "CAT"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation-like splits"):
        ClueAnswerIndex.from_json(path)
