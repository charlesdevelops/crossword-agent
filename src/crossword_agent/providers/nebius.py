from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from crossword_agent.models import Candidate, CandidateBatch, ClueRequest, UsageMetrics
from crossword_agent.providers.common import invoke_structured_model

NEBIUS_MODEL_OPTIONS = (
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "moonshotai/Kimi-K3",
)
KIMI_MODEL_ID = "moonshotai/Kimi-K3"
DEFAULT_NEBIUS_MODEL = KIMI_MODEL_ID
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.us-north1.nebius.com/v1/"
DEFAULT_KIMI_NEBIUS_BASE_URL = (
    "https://api.tokenfactory.eu-west2.nebius.com/v1/"
)
DEFAULT_NEBIUS_MAX_TOKENS = 4096
DEFAULT_NEBIUS_REQUEST_TIMEOUT = 120.0
NEBIUS_REASONING_EFFORT_OPTIONS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


class NebiusCandidateProvider:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_NEBIUS_MODEL,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not resolved_key:
            raise ValueError("NEBIUS_API_KEY is required for the Nebius provider")
        if model_id == KIMI_MODEL_ID:
            base_url = os.getenv(
                "NEBIUS_KIMI_API_BASE",
                DEFAULT_KIMI_NEBIUS_BASE_URL,
            )
        else:
            base_url = os.getenv("NEBIUS_API_BASE", DEFAULT_NEBIUS_BASE_URL)
        max_tokens = int(
            os.getenv("NEBIUS_MAX_TOKENS", str(DEFAULT_NEBIUS_MAX_TOKENS))
        )
        request_timeout = float(
            os.getenv(
                "NEBIUS_REQUEST_TIMEOUT",
                str(DEFAULT_NEBIUS_REQUEST_TIMEOUT),
            )
        )
        configured_reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.getenv("NEBIUS_REASONING_EFFORT")
        )
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
            max_retries=0,
            timeout=request_timeout,
        )
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._reasoning_effort = configured_reasoning_effort
        self._response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "candidate_batch",
                "strict": True,
                "schema": candidate_batch_json_schema(),
            },
        }
        self.usage = UsageMetrics()

    async def generate_candidates(
        self,
        requests: list[ClueRequest],
    ) -> dict[str, list[Candidate]]:
        return await invoke_structured_model(
            invoke=self._invoke,
            requests=requests,
            usage=self.usage,
            provider_name="nebius",
            model_id=self._model_id,
        )

    async def _invoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "response_format": self._response_format,
        }
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort
        response = await self._client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"Nebius model refused structured output: {refusal}")
        content = _message_content(message.content)
        if not content:
            raise ValueError(
                "Nebius model returned empty structured output "
                f"(finish_reason={choice.finish_reason!r})"
            )
        return {
            "parsed": CandidateBatch.model_validate_json(content),
            "raw": response,
        }


def candidate_batch_json_schema() -> dict[str, Any]:
    return _strict_schema(CandidateBatch.model_json_schema())


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result = dict(schema)
    if result.get("type") == "object":
        result["additionalProperties"] = False
    for key, value in list(result.items()):
        if isinstance(value, dict):
            result[key] = _strict_schema(value)
        elif isinstance(value, list):
            result[key] = [
                _strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
    return result


def _message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""
