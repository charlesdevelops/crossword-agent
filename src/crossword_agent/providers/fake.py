from __future__ import annotations

from collections import defaultdict

from crossword_agent.models import Candidate, ClueRequest, UsageMetrics


class ScriptedCandidateProvider:
    """Deterministic provider used by tests and examples."""

    def __init__(self, scripted: dict[str, list[list[Candidate]]]) -> None:
        self._scripted = scripted
        self._calls_by_entry: dict[str, int] = defaultdict(int)
        self.usage = UsageMetrics()

    async def generate_candidates(
        self,
        requests: list[ClueRequest],
    ) -> dict[str, list[Candidate]]:
        self.usage.calls += 1
        response: dict[str, list[Candidate]] = {}
        for request in requests:
            batches = self._scripted.get(request.entry_id, [])
            call_index = self._calls_by_entry[request.entry_id]
            self._calls_by_entry[request.entry_id] += 1
            if batches:
                selected = batches[min(call_index, len(batches) - 1)]
                response[request.entry_id] = [
                    candidate.model_copy(
                        update={
                            "source": (
                                "verifier"
                                if request.strategy == "verify"
                                else candidate.source
                            )
                        }
                    )
                    for candidate in selected
                ]
        return response
