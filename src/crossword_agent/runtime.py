from __future__ import annotations

import os
from functools import lru_cache

from crossword_agent.config import load_local_env
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.providers.nebius import DEFAULT_NEBIUS_MODEL, NebiusCandidateProvider

load_local_env()


def create_provider(
    model_id: str | None = None,
    reasoning_effort: str | None = None,
) -> CandidateProvider:
    return NebiusCandidateProvider(
        model_id=model_id or os.getenv("NEBIUS_MODEL_ID", DEFAULT_NEBIUS_MODEL),
        api_key=_nebius_api_key(),
        reasoning_effort=reasoning_effort,
    )


@lru_cache(maxsize=1)
def create_lexicon() -> PatternLexicon:
    limit = int(os.getenv("WORDFREQ_LIMIT", "100000"))
    return PatternLexicon.from_wordfreq(limit=limit)


@lru_cache(maxsize=1)
def _nebius_api_key() -> str | None:
    return os.getenv("NEBIUS_API_KEY")
