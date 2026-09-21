from __future__ import annotations

import os

from pydantic import SecretStr

from crossword_agent.models import Candidate, CandidateBatch, ClueRequest, UsageMetrics
from crossword_agent.providers.common import invoke_structured_model

DEFAULT_NEBIUS_MODEL = "Qwen/Qwen3-235B-A22B"


class NebiusCandidateProvider:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_NEBIUS_MODEL,
        api_key: str | None = None,
    ) -> None:
        from langchain_nebius import ChatNebius

        resolved_key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not resolved_key:
            raise ValueError("NEBIUS_API_KEY is required for the Nebius provider")
        model = ChatNebius(
            model=model_id,
            api_key=SecretStr(resolved_key),
            temperature=0,
            max_tokens=4096,
        )
        self._model = model.with_structured_output(
            CandidateBatch,
            include_raw=True,
            method="json_schema",
            strict=True,
        )
        self._model_id = model_id
        self.usage = UsageMetrics()

    async def generate_candidates(
        self,
        requests: list[ClueRequest],
    ) -> dict[str, list[Candidate]]:
        return await invoke_structured_model(
            invoke=self._model.ainvoke,
            requests=requests,
            usage=self.usage,
            provider_name="nebius",
            model_id=self._model_id,
        )
