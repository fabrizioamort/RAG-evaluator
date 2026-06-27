"""Tests for the Legal RAG Bench binary judge service."""

from __future__ import annotations

from typing import Any

import pytest

from app.services.legal_rag_bench_judge import LegalRAGBenchJudge, _is_non_answer
from app.services.llm_provider import LLMCompletionResponse, TokenUsage


class FakeProviderService:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, str]] | None = None

    async def completion(self, **kwargs: Any) -> LLMCompletionResponse:
        self.messages = kwargs["messages"]
        return LLMCompletionResponse(
            content=self.content,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=kwargs["model"],
            provider=kwargs.get("provider") or "test",
            latency_seconds=0.0,
        )


@pytest.mark.asyncio
async def test_judge_overrides_refusal_marked_success_by_model() -> None:
    provider = FakeProviderService(
        content=(
            '{"correct": true, "grounded": true, "reasoning": '
            '"The reference answer is supported by the retrieved context."}'
        )
    )
    judge = LegalRAGBenchJudge(provider_service=provider)  # type: ignore[arg-type]

    result = await judge.judge(
        question=(
            "Should you notify the jury that the standard sentence for statutory "
            "murder of an emergency worker is 30 years imprisonment?"
        ),
        reference_answer=(
            "No. Counsel should not refer to the penalty prescribed by law for "
            "the offence charged."
        ),
        generated_answer="I cannot answer this question based on the provided context.",
        retrieved_context=[
            "Counsel should not refer to the penalty prescribed by law for the "
            "offence charged or make any other reference to the consequences "
            "which will flow from the jury's verdict."
        ],
        model="judge-model",
        provider="test-provider",
        base_url=None,
        api_key=None,
    )

    assert result["correct"] is False
    assert result["grounded"] is False
    assert result["overrides"] == [
        "non_answer_correct_false",
        "non_answer_grounded_false",
    ]
    assert "Deterministic override" in result["reasoning"]


def test_non_answer_detector_does_not_flag_legal_no_answer() -> None:
    assert not _is_non_answer(
        "No. Counsel should not refer to the penalty prescribed by law."
    )


@pytest.mark.asyncio
async def test_judge_prompt_separates_generated_answer_from_reference() -> None:
    provider = FakeProviderService(
        content='{"correct": false, "grounded": false, "reasoning": "No answer."}'
    )
    judge = LegalRAGBenchJudge(provider_service=provider)  # type: ignore[arg-type]

    await judge.judge(
        question="Question?",
        reference_answer="Reference.",
        generated_answer="Generated.",
        retrieved_context=["Context."],
        model="judge-model",
        provider="test-provider",
        base_url=None,
        api_key=None,
    )

    assert provider.messages is not None
    assert "Do not mark the answer correct merely because" in provider.messages[0]["content"]
    assert "ANSWER UNDER EVALUATION:\nGenerated." in provider.messages[1]["content"]
    assert "Reference answer, used only for correctness:\nReference." in provider.messages[1][
        "content"
    ]
