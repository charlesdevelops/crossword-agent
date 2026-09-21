from __future__ import annotations

import os

import pytest

from crossword_agent.models import ClueRequest
from crossword_agent.providers.nebius import NebiusCandidateProvider

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_NEBIUS_TESTS") != "1",
    reason="Set RUN_NEBIUS_TESTS=1 to make a paid Token Factory request",
)
async def test_nebius_returns_structured_candidates() -> None:
    provider = NebiusCandidateProvider()
    result = await provider.generate_candidates(
        [
            ClueRequest(
                entry_id="1A",
                clue="Courageous",
                length=5,
                pattern="BRA?E",
            )
        ]
    )

    assert result["1A"][0].answer.upper() == "BRAVE"
    assert provider.usage.calls >= 1

