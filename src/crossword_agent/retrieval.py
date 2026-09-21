from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossword_agent.constraints import normalize_answer

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class ClueAnswerExample:
    clue: str
    answer: str


@dataclass(frozen=True)
class RetrievedAnswer:
    answer: str
    score: float
    matched_clue: str


class ClueAnswerIndex:
    """Small local BM25 clue index built only from explicitly approved training data."""

    def __init__(self, examples: Iterable[ClueAnswerExample]) -> None:
        normalized: list[ClueAnswerExample] = []
        seen: set[tuple[str, str]] = set()
        for example in examples:
            clue = " ".join(example.clue.split())
            answer = normalize_answer(example.answer)
            key = (clue.casefold(), answer)
            if not clue or len(answer) < 2 or key in seen:
                continue
            seen.add(key)
            normalized.append(ClueAnswerExample(clue=clue, answer=answer))
        self._examples = tuple(normalized)
        self._tokens = tuple(_tokenize(example.clue) for example in self._examples)
        self._document_frequency: Counter[str] = Counter()
        self._by_length: dict[int, list[int]] = defaultdict(list)
        for index, (example, tokens) in enumerate(
            zip(self._examples, self._tokens, strict=True)
        ):
            self._document_frequency.update(set(tokens))
            self._by_length[len(example.answer)].append(index)
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    @classmethod
    def empty(cls) -> ClueAnswerIndex:
        return cls(())

    @classmethod
    def from_json(cls, path: Path) -> ClueAnswerIndex:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        declared_splits = {
            str(value).casefold()
            for value in payload.get("allowed_splits", [])
        }
        forbidden = declared_splits & {"demo", "evaluation", "heldout", "test"}
        if forbidden:
            raise ValueError(
                f"Clue index declares evaluation-like splits {sorted(forbidden)}: {path}"
            )
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError(f"Clue index has no examples list: {path}")
        return cls(
            ClueAnswerExample(
                clue=str(item["clue"]),
                answer=str(item["answer"]),
            )
            for item in examples
            if isinstance(item, dict) and "clue" in item and "answer" in item
        )

    def __len__(self) -> int:
        return len(self._examples)

    def search(
        self,
        clue: str,
        *,
        length: int,
        pattern: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedAnswer]:
        if not self._examples or limit < 1:
            return []
        query_tokens = _tokenize(clue)
        if not query_tokens:
            return []
        normalized_pattern = (pattern or "?" * length).upper()
        scored: list[tuple[float, str, str]] = []
        for index in self._by_length.get(length, []):
            example = self._examples[index]
            if not _matches_pattern(example.answer, normalized_pattern):
                continue
            score = self._bm25(query_tokens, self._tokens[index])
            if example.clue.casefold() == " ".join(clue.split()).casefold():
                score += 2.0
            if score > 0:
                scored.append((score, example.answer, example.clue))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        answers: dict[str, RetrievedAnswer] = {}
        for score, answer, matched_clue in scored:
            normalized_score = 1.0 - math.exp(-score / 3.0)
            existing = answers.get(answer)
            candidate = RetrievedAnswer(
                answer=answer,
                score=normalized_score,
                matched_clue=matched_clue,
            )
            if existing is None or candidate.score > existing.score:
                answers[answer] = candidate
        return sorted(
            answers.values(),
            key=lambda item: (-item.score, item.answer),
        )[:limit]

    def _bm25(self, query_tokens: tuple[str, ...], document: tuple[str, ...]) -> float:
        frequencies = Counter(document)
        document_length = len(document)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in set(query_tokens):
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_frequency = math.log(
                1.0
                + (len(self._examples) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = frequency + k1 * (
                1.0
                - b
                + b * document_length / max(self._average_length, 1.0)
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / normalization
        return score


def _tokenize(value: str) -> tuple[str, ...]:
    tokens = tuple(
        token.casefold()
        for token in _TOKEN.findall(value)
        if token.casefold() not in _STOP_WORDS
    )
    return tokens or tuple(token.casefold() for token in _TOKEN.findall(value))


def _matches_pattern(answer: str, pattern: str) -> bool:
    return len(answer) == len(pattern) and all(
        expected == "?" or expected == actual
        for expected, actual in zip(pattern, answer, strict=True)
    )
