"""Unified interface for LLM providers using LiteLLM."""

import time
from typing import Any

import litellm
from pydantic import BaseModel

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

    async def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        provider: str | None = None,
        base_url: str | None = None,
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
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level (low/medium/high) for supported models
            **kwargs: Additional provider-specific parameters

        Returns:
            Standardized completion response
        """
        # Form the model string for litellm if needed
        full_model = model
        if provider and "/" not in model:
            full_model = f"{provider}/{model}"

        start_time = time.time()

        # Avoid temperature for reasoning models
        model_name_lower = model.lower()
        is_reasoning = any(
            x in model_name_lower for x in ["o1-", "o3-", "/o1", "/o3", "gpt-5"]
        ) or model_name_lower in ["o1", "o3", "gpt-5"]

        actual_temp: float | None = temperature
        if is_reasoning:
            actual_temp = None

        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = await litellm.acompletion(
                model=full_model,
                messages=messages,
                api_base=base_url,
                temperature=actual_temp,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency = time.time() - start_time

            # Extract content and usage
            content = response.choices[0].message.content or ""
            usage_data = response.get("usage", {})

            # Actually cost is often in usage or can be calculated by LiteLLM if pricing is loaded
            cost = response.get("_total_cost", 0.0)

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cost_usd=float(cost) if cost else 0.0,
            )

            logger.info(
                "LLM completion successful",
                model=full_model,
                tokens=usage.total_tokens,
                latency=latency,
            )

            return LLMCompletionResponse(
                content=content,
                usage=usage,
                model=model,
                provider=provider or "unknown",
                latency_seconds=latency,
            )

        except Exception as e:
            error_str = str(e).lower()
            # If it's a temperature error, retry without temperature
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
                try:
                    retry_response = await litellm.acompletion(
                        model=full_model,
                        messages=messages,
                        api_base=base_url,
                        temperature=None,  # Explicitly remove temperature
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    latency = time.time() - start_time

                    content = retry_response.choices[0].message.content or ""
                    usage_data = retry_response.get("usage", {})
                    cost = retry_response.get("_total_cost", 0.0)

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
                        latency_seconds=latency,
                    )
                except Exception as retry_e:
                    logger.error("LLM retry failed", error=str(retry_e), model=full_model)
                    raise retry_e

            logger.error("LLM completion failed", error=str(e), model=full_model)
            raise

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
