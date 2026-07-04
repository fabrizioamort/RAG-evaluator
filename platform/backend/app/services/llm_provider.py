"""Unified interface for LLM providers using LiteLLM."""

import asyncio
import time
from typing import Any

import litellm
from pydantic import BaseModel
from rag_evaluator.common.llm_utils import (
    is_reasoning_model,
    is_transient_llm_error,
    rejects_temperature,
)

from app.config import settings
from app.services.provider_resolver import normalize_model_for_provider
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class TokenUsage(BaseModel):
    """Token usage tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class LLMCompletionResponse(BaseModel):
    """Standardized response for LLM completions."""

    content: str
    usage: TokenUsage
    model: str
    provider: str
    latency_seconds: float


class LLMEmbeddingResponse(BaseModel):
    """Standardized response for LLM embeddings."""

    embedding: list[float]
    usage: TokenUsage
    model: str
    provider: str
    latency_seconds: float


class LLMProviderService:
    """Service for interacting with various LLM providers via LiteLLM."""

    def __init__(self) -> None:
        """Initialize the LLM provider service."""
        # Configure litellm
        litellm.telemetry = False
        litellm.drop_params = True  # Drop unsupported params for different providers
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    def _completion_model_for_litellm(
        self,
        model: str,
        provider: str | None,
        base_url: str | None,
    ) -> str:
        """Build the LiteLLM model string for a completion request.

        OpenRouter exposes an OpenAI-compatible endpoint. Routing it through
        LiteLLM's generic OpenAI-compatible adapter avoids provider-specific
        optional params that newer OpenAI clients reject.
        """
        provider_name = (provider or "").lower()
        if provider_name == "openrouter" and base_url:
            openrouter_model = normalize_model_for_provider(provider, model)
            return f"openai/{openrouter_model}"

        if provider and "/" not in model:
            return f"{provider}/{model}"

        return model

    def _completion_response_from_litellm(
        self,
        response: Any,
        *,
        model: str,
        provider: str | None,
        latency_seconds: float,
    ) -> LLMCompletionResponse:
        """Convert a LiteLLM response into the platform response model."""
        content = response.choices[0].message.content or ""
        usage_data = response.get("usage", {})
        cost = response.get("_total_cost", 0.0)

        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cost_usd=float(cost) if cost else 0.0,
        )

        return LLMCompletionResponse(
            content=content,
            usage=usage,
            model=model,
            provider=provider or "unknown",
            latency_seconds=latency_seconds,
        )

    async def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> LLMCompletionResponse:
        """Get a chat completion from an LLM.

        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-haiku-20240307')
            messages: List of chat messages
            provider: Explicit provider name (optional)
            base_url: Custom API base URL (e.g., for Ollama)
            api_key: Explicit API key (preferred over env-based resolution)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level (low/medium/high) for supported models
            **kwargs: Additional provider-specific parameters

        Returns:
            Standardized completion response
        """
        # Form the model string for litellm if needed
        full_model = self._completion_model_for_litellm(model, provider, base_url)

        start_time = time.time()

        # Omit temperature only for models whose API rejects it; other
        # reasoning-capable models accept it and need it for determinism.
        actual_temp: float | None = temperature
        if rejects_temperature(model):
            actual_temp = None

        if reasoning_effort is not None and is_reasoning_model(model):
            kwargs["reasoning_effort"] = reasoning_effort

        # Passing an explicit api_key uses litellm's well-tested credential path
        # instead of env-based pickup.
        if api_key is not None:
            kwargs["api_key"] = api_key

        retry_attempts = int(kwargs.pop("retry_attempts", settings.LLM_COMPLETION_RETRY_ATTEMPTS))
        retry_base_delay = float(
            kwargs.pop(
                "retry_base_delay",
                settings.LLM_COMPLETION_RETRY_BASE_DELAY_SECONDS,
            )
        )
        timeout = kwargs.pop("timeout", settings.LLM_COMPLETION_TIMEOUT_SECONDS)
        timeout_seconds = float(timeout) if timeout is not None else None

        max_attempts = max(1, retry_attempts)
        attempt = 0
        while attempt < max_attempts:
            try:
                completion_kwargs = {
                    "model": full_model,
                    "messages": messages,
                    "base_url": base_url,
                    "temperature": actual_temp,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if timeout_seconds and timeout_seconds > 0:
                    completion_kwargs["timeout"] = timeout_seconds

                completion = litellm.acompletion(**completion_kwargs)
                if timeout_seconds and timeout_seconds > 0:
                    response = await asyncio.wait_for(completion, timeout=timeout_seconds)
                else:
                    response = await completion
                latency = time.time() - start_time
                result = self._completion_response_from_litellm(
                    response,
                    model=model,
                    provider=provider,
                    latency_seconds=latency,
                )

                logger.info(
                    "LLM completion successful",
                    model=full_model,
                    tokens=result.usage.total_tokens,
                    latency=latency,
                    attempt=attempt + 1,
                )

                return result

            except Exception as e:
                error_str = str(e).lower()
                if (
                    "temperature" in error_str
                    and ("does not support" in error_str or "unsupported_value" in error_str)
                    and actual_temp is not None
                ):
                    logger.warning(
                        "Temperature unsupported by model, retrying without it",
                        model=full_model,
                        error=str(e),
                    )
                    actual_temp = None
                    # Retry without temperature; does not consume a retry attempt.
                    continue

                attempt += 1
                if attempt < max_attempts and is_transient_llm_error(e):
                    delay = retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient LLM completion error, retrying",
                        model=full_model,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                        timeout_seconds=timeout_seconds,
                        error=str(e),
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue

                logger.error("LLM completion failed", error=str(e), model=full_model)
                raise

        raise RuntimeError("LLM completion failed without raising an exception")

    async def get_embedding(
        self,
        model: str,
        input_text: str | list[str],
        provider: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> LLMEmbeddingResponse:
        """Get embeddings for text.

        Args:
            model: Embedding model name
            input_text: Text or list of texts to embed
            provider: Explicit provider name (optional)
            base_url: Custom API base URL
            **kwargs: Additional parameters

        Returns:
            Standardized embedding response
        """
        full_model = model
        if provider and "/" not in model:
            full_model = f"{provider}/{model}"

        start_time = time.time()
        try:
            response = await litellm.aembedding(
                model=full_model,
                input=input_text,
                api_base=base_url,
                **kwargs,
            )
            latency = time.time() - start_time

            usage_data = response.get("usage", {})
            cost = response.get("_total_cost", 0.0)

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", usage_data.get("prompt_tokens", 0)),
                cost_usd=float(cost) if cost else 0.0,
            )

            # For list input, this might need adjustment depending on how we want to return it
            # Standardizing to single embedding for now if input was single string
            embedding = response.data[0]["embedding"]

            logger.info(
                "LLM embedding successful",
                model=full_model,
                tokens=usage.total_tokens,
                latency=latency,
            )

            return LLMEmbeddingResponse(
                embedding=embedding,
                usage=usage,
                model=model,
                provider=provider or "unknown",
                latency_seconds=latency,
            )

        except Exception as e:
            logger.error("LLM embedding failed", error=str(e), model=full_model)
            raise

    def count_tokens(self, model: str, text: str) -> int:
        """Count tokens in a string for a specific model.

        Args:
            model: Model name for tokenizer selection
            text: Text to count tokens for

        Returns:
            Token count
        """
        try:
            return int(litellm.token_counter(model=model, text=text))
        except Exception:
            # Fallback to a rough estimate (approx 4 chars per token)
            return len(text) // 4
