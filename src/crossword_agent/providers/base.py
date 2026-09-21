from __future__ import annotations

from typing import Protocol

from crossword_agent.models import Candidate, ClueRequest, UsageMetrics


class CandidateProvider(Protocol):
    usage: UsageMetrics

    async def generate_candidates(
        self,
        requests: list[ClueRequest],
    ) -> dict[str, list[Candidate]]: ...

