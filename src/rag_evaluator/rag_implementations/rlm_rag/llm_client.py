"""Two-tier LLM client with circuit breaker and caching."""

from __future__ import annotations

import enum
import hashlib
import logging
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from openai import OpenAI

from .exceptions import CircuitOpenError

if TYPE_CHECKING:
    from rag_evaluator.common.token_tracker import TokenUsage

    from .rlm_rag import RLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 3   # Failures before opening
    timeout: float = 60.0        # Seconds before half-open


class CircuitBreaker:
    """Circuit breaker to prevent thundering herd on API failures.

    States:
    - CLOSED: Normal operation, counting failures
    - OPEN: Rejecting all calls immediately (fail fast)
    - HALF_OPEN: Allowing one test call to check recovery

    Usage:
        breaker = CircuitBreaker()
        if breaker.can_execute():
            try:
                result = api_call()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            raise RuntimeError("Circuit open")
    """

    def __init__(self, config: CircuitConfig | None = None):
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure: float | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current state, transitioning OPEN -> HALF_OPEN if timeout passed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if (self._last_failure is not None and
                    time.time() - self._last_failure >= self.config.timeout):
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def can_execute(self) -> bool:
        """Check if a call can proceed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return True  # Allow test call
        else:  # OPEN
            return False

    def record_success(self) -> None:
        """Record successful call - resets to CLOSED."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker CLOSED after successful test")
            self._state = CircuitState.CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        """Record failed call - may open circuit."""
        with self._lock:
            self._failures += 1
            self._last_failure = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker OPEN after failed test")
                self._state = CircuitState.OPEN
            elif self._failures >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker OPEN after {self._failures} failures"
                )
                self._state = CircuitState.OPEN

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "state": self.state.value,
            "failures": self._failures,
            "last_failure": self._last_failure,
        }


# ============================================================================
# Response Cache
# ============================================================================

class ResponseCache:
    """Thread-safe LRU cache with TTL for LLM responses.

    Caches based on hash of (model, prompt) to avoid repeated API calls.
    """

    def __init__(self, max_entries: int = 100, ttl: float = 300.0):
        self.max_entries = max_entries
        self.ttl = ttl
        self._cache: dict[str, tuple[str, float]] = {}  # key -> (value, timestamp)
        self._lock = threading.RLock()

    def get(self, key: str) -> str | None:
        """Get cached value if exists and not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                # Expired - remove
                del self._cache[key]
            return None

    def set(self, key: str, value: str) -> None:
        """Set cached value, evicting oldest if full."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_entries:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )
                del self._cache[oldest_key]

            self._cache[key] = (value, time.time())

    def make_key(self, prompt: str, model: str, **kwargs) -> str:
        """Create cache key from prompt and model."""
        key_data = f"{model}:{prompt}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "ttl": self.ttl,
            }


# ============================================================================
# Chat Response
# ============================================================================

@dataclass
class ChatResponse:
    """Response from LLM chat completion."""
    content: str
    tokens_used: int
    cached: bool = False


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Two-tier LLM client with circuit breaker and caching.

    Architecture:
    - Orchestrator model (gpt-5-mini): Main reasoning, code generation
    - Worker model (gpt-5-nano): Chunk processing, summaries

    Features:
    - Circuit breaker prevents cascading failures
    - Response caching reduces API calls
    - Exponential backoff retry for transient errors
    - Recursion depth tracking for sub-calls
    """

    def __init__(
        self,
        config: RLMConfig,
        token_usage: TokenUsage,
    ):
        self.config = config
        self.token_usage = token_usage

        # Initialize OpenAI-compatible client (endpoint resolved from config,
        # falling back to the OPENAI_API_KEY env var).
        api_key = config.llm_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key set (config.llm_api_key or OPENAI_API_KEY)")
        self.client = OpenAI(api_key=api_key, base_url=config.llm_base_url)

        # Circuit breaker
        self._circuit = CircuitBreaker(CircuitConfig(
            failure_threshold=config.circuit_failure_threshold,
            timeout=config.circuit_timeout,
        ))

        # Response cache (optional)
        self._cache = ResponseCache(
            max_entries=config.cache_max_entries,
            ttl=config.cache_ttl_seconds,
        ) if config.enable_cache else None

        # Recursion tracking for sub-calls
        self._current_depth = 0
        self._max_tokens_param_by_model: dict[str, str] = {}
        self._temperature_mode_by_model: dict[str, str] = {}

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        # Headroom for gpt-5 reasoning orchestrators: reasoning and visible
        # output share this budget, so a tight cap can starve the response of
        # code and waste exploration steps. Sub-calls override this explicitly.
        max_tokens: int = 8000,
    ) -> ChatResponse:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to orchestrator_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            ChatResponse with content and token usage

        Raises:
            RuntimeError: If circuit breaker is open
        """
        model = model or self.config.orchestrator_model

        # Check circuit breaker
        if not self._circuit.can_execute():
            raise CircuitOpenError(
                "Circuit breaker OPEN - API unavailable",
                timeout_remaining=self.config.circuit_timeout,
            )

        # Check cache
        cache_key = ""
        if self._cache:
            cache_key = self._cache.make_key(str(messages), model)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {model}")
                return ChatResponse(content=cached, tokens_used=0, cached=True)

        # Execute with retry
        param_name = self._get_max_tokens_param(model)
        alt_param = "max_completion_tokens" if param_name == "max_tokens" else "max_tokens"
        max_param_candidates = [param_name]
        if alt_param not in max_param_candidates:
            max_param_candidates.append(alt_param)

        default_temp_only = self._get_temperature_mode(model) == "default_only"
        last_error: Exception | None = None
        response = None

        for max_param in max_param_candidates:
            temp_candidates = [None] if default_temp_only else [temperature, None]
            for temp_value in temp_candidates:
                try:
                    response = self._retry(
                        lambda: self._create_chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temp_value,
                            max_tokens=max_tokens,
                            max_tokens_param=max_param,
                        )
                    )
                    self._max_tokens_param_by_model[model] = max_param
                    if temp_value is None:
                        self._temperature_mode_by_model[model] = "default_only"
                    else:
                        self._temperature_mode_by_model.setdefault(model, "supported")
                    break
                except Exception as e:
                    last_error = e
                    unsupported_temp = self._is_unsupported_temperature(e)
                    unsupported_max = self._is_unsupported_max_tokens(e, max_param)
                    if unsupported_temp:
                        default_temp_only = True
                    if unsupported_max:
                        break
                    if unsupported_temp:
                        continue
                    self._circuit.record_failure()
                    raise
            if response is not None:
                break

        if response is None:
            self._circuit.record_failure()
            raise last_error if last_error else RuntimeError("LLM call failed")
        # Record success
        self._circuit.record_success()

        # Extract content and tokens
        content = response.choices[0].message.content or ""
        tokens = 0

        if response.usage:
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens
            self.token_usage.add_prompt_tokens(response.usage.prompt_tokens)
            self.token_usage.add_completion_tokens(response.usage.completion_tokens)

        # Cache response
        if self._cache:
            self._cache.set(cache_key, content)

        return ChatResponse(content=content, tokens_used=tokens)

    def call(
        self,
        prompt: str,
        context: str | None = None,
        mode: str = "analysis",
    ) -> str:
        """Sub-LLM call for REPL use (uses worker model).

        This method is exposed in the REPL namespace as `call_sub_llm()`.

        Args:
            prompt: Task prompt
            context: Optional context to include
            mode: One of "analysis", "summarize", "extract"

        Returns:
            LLM response text, or error message starting with [ERROR:
        """
        # Check recursion depth
        if self._current_depth >= self.config.max_recursion_depth:
            return "[ERROR: Max recursion depth reached]"

        self._current_depth += 1

        try:
            # Build full prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
            else:
                full_prompt = prompt

            # Get system prompt for mode
            system_prompts = {
                "analysis": "Analyze the content and provide detailed insights. Be thorough but concise.",
                "summarize": "Summarize the content concisely. Extract the most important points.",
                "extract": "Extract specific facts and data. Return structured information.",
            }
            system_prompt = system_prompts.get(mode, system_prompts["analysis"])

            # Use worker model for sub-calls
            response = self.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                model=self.config.worker_model,
                max_tokens=2000,
            )

            return response.content

        except Exception as e:
            logger.error(f"Sub-LLM call failed: {e}")
            return f"[ERROR: {e}]"

        finally:
            self._current_depth -= 1

    def _retry(self, func: Callable[[], T]) -> T:
        """Execute function with exponential backoff retry.

        Only retries on transient errors (rate limits, timeouts, connection).
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_error = e

                if attempt == self.config.max_retries:
                    break

                # Only retry on transient errors
                error_str = str(e).lower()
                transient_indicators = [
                    "rate_limit", "429",
                    "timeout", "timed out",
                    "connection", "connect",
                    "overloaded", "capacity",
                ]

                if any(indicator in error_str for indicator in transient_indicators):
                    delay = min(
                        self.config.retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                        30.0  # Cap at 30 seconds
                    )
                    logger.warning(
                        f"Retrying after {delay:.1f}s (attempt {attempt + 1}): {e}"
                    )
                    time.sleep(delay)
                else:
                    # Non-transient error - don't retry
                    raise

        raise last_error

    def _get_max_tokens_param(self, model: str) -> str:
        return self._max_tokens_param_by_model.get(model, "max_tokens")

    def _get_temperature_mode(self, model: str) -> str:
        return self._temperature_mode_by_model.get(model, "supported")

    def _create_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        max_tokens: int,
        max_tokens_param: str,
    ):
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            max_tokens_param: max_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if model == self.config.orchestrator_model and self.config.orchestrator_reasoning_effort:
            params["reasoning_effort"] = self.config.orchestrator_reasoning_effort
        return self.client.chat.completions.create(**params)

    def _is_unsupported_max_tokens(self, error: Exception, param_name: str) -> bool:
        message = str(error).lower()
        if "unsupported parameter" not in message:
            return False
        if param_name in message:
            return True
        if "max_tokens" in message or "max_completion_tokens" in message:
            return True
        return False

    def _is_unsupported_temperature(self, error: Exception) -> bool:
        message = str(error).lower()
        if "temperature" not in message:
            return False
        if "unsupported parameter" in message:
            return True
        if "unsupported value" in message:
            return True
        if "does not support" in message:
            return True
        return False

    def get_circuit_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return self._circuit.get_status()

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        return self._cache.get_stats() if self._cache else None
