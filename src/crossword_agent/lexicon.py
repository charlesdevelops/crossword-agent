from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


class PatternLexicon:
    def __init__(
        self,
        words: Iterable[str],
        *,
        frequency_scores: dict[str, float] | None = None,
    ) -> None:
        normalized = sorted(
            {
                word.strip().upper()
                for word in words
                if len(word.strip()) >= 2 and word.strip().isalpha()
            }
        )
        self._words = frozenset(normalized)
        self._frequency_scores = frequency_scores or {}
        self._by_length: dict[int, set[str]] = defaultdict(set)
        self._positions: dict[tuple[int, int, str], set[str]] = defaultdict(set)
        for word in normalized:
            self._by_length[len(word)].add(word)
            for position, char in enumerate(word):
                self._positions[(len(word), position, char)].add(word)

    @classmethod
    def from_wordfreq(cls, limit: int = 100_000) -> PatternLexicon:
        from wordfreq import top_n_list, zipf_frequency

        words = [word for word in top_n_list("en", limit) if word.isalpha()]
        scores = {
            word.upper(): max(0.0, min(1.0, (zipf_frequency(word, "en") - 1.0) / 7.0))
            for word in words
        }
        return cls(words, frequency_scores=scores)

    @classmethod
    def empty(cls) -> PatternLexicon:
        return cls(())

    @classmethod
    def from_words(cls, words: Iterable[str]) -> PatternLexicon:
        return cls(words)

    def contains(self, word: str) -> bool:
        return word.upper() in self._words

    def frequency_score(self, word: str) -> float:
        return self._frequency_scores.get(word.upper(), 0.0)

    def find(self, pattern: str, limit: int = 20) -> list[str]:
        normalized = pattern.upper().replace("?", ".")
        candidates = set(self._by_length.get(len(normalized), set()))
        for position, char in enumerate(normalized):
            if char == ".":
                continue
            candidates &= self._positions.get((len(normalized), position, char), set())
            if not candidates:
                break
        return sorted(
            candidates,
            key=lambda word: (-self.frequency_score(word), word),
        )[:limit]
