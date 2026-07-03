"""Binary Legal RAG Bench judge using the platform LLM provider service."""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.llm_provider import LLMProviderService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)
JUDGE_PARSE_RETRY_ATTEMPTS = 3
JUDGE_CONTEXT_MAX_CHARS = 40_000
JUDGE_CONTEXT_CHUNK_MAX_CHARS = 8_000

_NAVIGATION_CONTEXT_MARKERS = (
    "# Question Seeds",
    "_index/questions/question_seeds.md",
)

_NON_ANSWER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:i|we|the assistant|this model)\s+"
        r"(?:cannot|can't|can not|am unable to|are unable to)\s+"
        r"(?:answer|determine|provide)\b",
        r"\b(?:cannot|can't|can not|unable to)\s+"
        r"(?:answer|determine)\s+(?:this|the)\s+question\s+based\s+on\b",
        r"\b(?:not enough|insufficient)\s+(?:information|context|evidence)\b",
        r"\bprovided context\s+"
        r"(?:does not|doesn't|do not|don't)\s+(?:contain|provide|include|specify)\b",
        r"\bbased on (?:the )?provided context[, ]+"
        r"(?:i\s+)?(?:cannot|can't|can not|am unable to)\b",
        r"\bi don't know\b",
    )
)


class LegalRAGBenchJudge:
    """Run a paper-style binary correctness/groundedness judge."""

    def __init__(self, provider_service: LLMProviderService | None = None) -> None:
        self.provider_service = provider_service or LLMProviderService()

    async def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_context: list[str],
        model: str,
        provider: str | None,
        base_url: str | None,
        api_key: str | None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        context = _format_judge_context(retrieved_context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for Legal RAG Bench answers. "
                    "Evaluate only the generated answer, labelled ANSWER UNDER "
                    "EVALUATION. Do not answer the legal question yourself. Do "
                    "not mark the answer correct merely because the reference "
                    "answer is supported by the retrieved context. Return only "
                    "JSON with keys correct, grounded, and reasoning. correct is "
                    "true only when the ANSWER UNDER EVALUATION directly answers "
                    "the question and entails every material part of the reference "
                    "answer. If the ANSWER UNDER EVALUATION refuses to answer, "
                    "says the context is insufficient, says it cannot determine "
                    "the answer, or otherwise omits the requested legal conclusion, "
                    "correct must be false. grounded is true only when every "
                    "material claim in the ANSWER UNDER EVALUATION is fully "
                    "supported by the retrieved context. A refusal or non-answer "
                    "is not a grounded legal answer; mark grounded false."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"ANSWER UNDER EVALUATION:\n{generated_answer}\n\n"
                    f"Reference answer, used only for correctness:\n{reference_answer}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Before returning, check whether the ANSWER UNDER EVALUATION "
                    "itself gave the legal conclusion. If it did not, correct=false. "
                    "Return JSON only."
                ),
            },
        ]
        parsed: dict[str, Any] | None = None
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost_usd = 0.0

        for attempt in range(1, JUDGE_PARSE_RETRY_ATTEMPTS + 1):
            completion_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            if timeout_seconds is not None:
                completion_kwargs["timeout"] = timeout_seconds

            response = await self.provider_service.completion(
                **completion_kwargs,
            )
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens
            cost_usd += response.usage.cost_usd

            parsed = _parse_judge_json(response.content)
            if not parsed.get("parse_error"):
                break

            logger.warning(
                "Legal RAG judge returned unparsable response",
                model=model,
                provider=provider,
                attempt=attempt,
                max_attempts=JUDGE_PARSE_RETRY_ATTEMPTS,
                content_preview=response.content[:200],
            )

        assert parsed is not None
        parsed = _apply_answer_sanity_checks(parsed, generated_answer)
        parsed["model"] = model
        parsed["provider"] = provider
        parsed["attempts"] = attempt
        parsed["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        parsed["cost_usd"] = cost_usd
        return parsed


def _parse_judge_json(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {
                "correct": None,
                "grounded": None,
                "reasoning": content.strip(),
                "parse_error": "judge_response_not_json",
            }
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {
                "correct": None,
                "grounded": None,
                "reasoning": content.strip(),
                "parse_error": "judge_response_not_json",
            }

    return {
        "correct": _coerce_bool(data.get("correct")),
        "grounded": _coerce_bool(data.get("grounded")),
        "reasoning": str(data.get("reasoning", "")).strip(),
        "raw": data,
    }


def _format_judge_context(retrieved_context: list[str]) -> str:
    """Build a bounded evidence context for the judge prompt."""
    chunks: list[str] = []
    total_chars = 0

    for raw_chunk in retrieved_context:
        chunk = str(raw_chunk).strip()
        if not chunk or _is_navigation_context(chunk):
            continue

        if len(chunk) > JUDGE_CONTEXT_CHUNK_MAX_CHARS:
            chunk = (
                chunk[:JUDGE_CONTEXT_CHUNK_MAX_CHARS].rstrip()
                + "\n... [retrieved context chunk truncated]"
            )

        separator_chars = 2 if chunks else 0
        remaining = JUDGE_CONTEXT_MAX_CHARS - total_chars - separator_chars
        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = (
                chunk[: max(0, remaining - 41)].rstrip()
                + "\n... [retrieved context truncated]"
            )
            chunks.append(chunk)
            break

        chunks.append(chunk)
        total_chars += len(chunk) + separator_chars

    return "\n\n".join(chunks)


def _is_navigation_context(chunk: str) -> bool:
    return any(marker in chunk[:500] for marker in _NAVIGATION_CONTEXT_MARKERS)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _apply_answer_sanity_checks(result: dict[str, Any], generated_answer: str) -> dict[str, Any]:
    if not _is_non_answer(generated_answer):
        return result

    adjusted = dict(result)
    adjusted["abstention"] = True
    overrides = list(adjusted.get("overrides", []))

    if adjusted.get("correct") is not False:
        adjusted["correct"] = False
        overrides.append("non_answer_correct_false")

    if adjusted.get("grounded") is not False:
        adjusted["grounded"] = False
        overrides.append("non_answer_grounded_false")

    if overrides:
        adjusted["overrides"] = overrides
        reason = adjusted.get("reasoning", "").strip()
        override_reason = (
            "Deterministic override: the generated answer is a refusal or non-answer."
        )
        adjusted["reasoning"] = f"{reason} {override_reason}".strip()

    return adjusted


def _is_non_answer(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in _NON_ANSWER_PATTERNS)
