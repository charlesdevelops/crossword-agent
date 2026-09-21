from __future__ import annotations

from crossword_agent.models import Candidate, CandidateBatch, ClueRequest, UsageMetrics
from crossword_agent.providers.common import invoke_structured_model

DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-sonnet-5"
DEFAULT_BEDROCK_REGION = "ap-southeast-2"


class BedrockCandidateProvider:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_BEDROCK_MODEL,
        region_name: str = DEFAULT_BEDROCK_REGION,
    ) -> None:
        from langchain_aws import ChatBedrockConverse

        model = ChatBedrockConverse(
            model_id=model_id,
            region_name=region_name,
            max_tokens=4096,
        )
        self._model = model.with_structured_output(
            CandidateBatch,
            include_raw=True,
            method="function_calling",
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
            provider_name="aws.bedrock",
            model_id=self._model_id,
        )
