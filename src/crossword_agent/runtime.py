from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from crossword_agent.config import load_local_env
from crossword_agent.lexicon import PatternLexicon
from crossword_agent.providers.base import CandidateProvider
from crossword_agent.providers.bedrock import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_BEDROCK_REGION,
    BedrockCandidateProvider,
)
from crossword_agent.providers.nebius import DEFAULT_NEBIUS_MODEL, NebiusCandidateProvider
from crossword_agent.retrieval import ClueAnswerIndex

load_local_env()


def create_provider() -> CandidateProvider:
    provider_name = os.getenv("LLM_PROVIDER", "bedrock").lower()
    if provider_name == "bedrock":
        return BedrockCandidateProvider(
            model_id=os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL),
            region_name=os.getenv("BEDROCK_REGION", DEFAULT_BEDROCK_REGION),
        )
    if provider_name == "nebius":
        return NebiusCandidateProvider(
            model_id=os.getenv("NEBIUS_MODEL_ID", DEFAULT_NEBIUS_MODEL),
            api_key=_nebius_api_key(),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider_name}")


@lru_cache(maxsize=1)
def create_lexicon() -> PatternLexicon:
    limit = int(os.getenv("WORDFREQ_LIMIT", "100000"))
    return PatternLexicon.from_wordfreq(limit=limit)


@lru_cache(maxsize=1)
def create_clue_index() -> ClueAnswerIndex:
    configured = os.getenv("CROSSWORD_CLUE_INDEX_PATH")
    if not configured:
        return ClueAnswerIndex.empty()
    path = Path(configured)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured clue index does not exist: {path}. "
            "Run scripts/build_clue_index.py against training data first."
        )
    return ClueAnswerIndex.from_json(path)


@lru_cache(maxsize=1)
def _nebius_api_key() -> str | None:
    return os.getenv("NEBIUS_API_KEY")
