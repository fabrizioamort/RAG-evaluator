"""Utilities for working with different LLM models and their parameters."""

from typing import Any

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT_ERROR_MARKERS = (
    "api connection",
    "bad gateway",
    "badgateway",
    "connection error",
    "gateway timeout",
    "internal server error",
    "overloaded",
    "rate limit",
    "ratelimit",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "upstream error",
)


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is reasoning-capable (emits reasoning tokens).

    Reasoning-capable models accept reasoning_effort and may need generous
    completion budgets. Whether they reject temperature is a separate,
    API-level property: see rejects_temperature.

    Args:
        model_name: Name of the LLM model.

    Returns:
        True if the model is identified as a reasoning model.
    """
    if not model_name:
        return False

    model_lower = model_name.lower()

    if any(
        x in model_lower for x in ["o1-", "o3-", "/o1", "/o3", "gpt-5", "deepseek-v4-flash"]
    ) or model_lower in [
        "o1",
        "o3",
        "gpt-5",
    ]:
        return True

    return False


def rejects_temperature(model_name: str) -> bool:
    """Check if a model's API rejects the temperature parameter.

    Only OpenAI o-series and gpt-5 models error on temperature. Other
    reasoning-capable models (e.g. deepseek-v4-flash) accept it; dropping
    it for them silently falls back to the provider default and makes
    evaluation runs non-deterministic.

    Args:
        model_name: Name of the LLM model.

    Returns:
        True if temperature must be omitted for this model.
    """
    if not model_name:
        return False

    model_lower = model_name.lower()

    return any(x in model_lower for x in ["o1-", "o3-", "/o1", "/o3", "gpt-5"]) or model_lower in [
        "o1",
        "o3",
        "gpt-5",
    ]


def is_transient_llm_error(error: BaseException) -> bool:
    """Return whether an LLM call failure is likely safe to retry."""
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    try:
        if status_code is not None and int(status_code) in _TRANSIENT_STATUS_CODES:
            return True
    except (TypeError, ValueError):
        pass

    error_name = error.__class__.__name__.lower()
    error_text = str(error).lower()
    return any(marker in error_name or marker in error_text for marker in _TRANSIENT_ERROR_MARKERS)


def get_safe_llm_params(
    model_name: str,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get parameters that are safe for the specified model.

    Removes temperature for models whose API rejects it and forwards
    reasoning_effort only for recognized reasoning-capable models.

    Args:
        model_name: Name of the LLM model.
        temperature: Requested temperature.
        reasoning_effort: Reasoning effort level (low/medium/high).
        **kwargs: Additional LLM parameters.

    Returns:
        Dictionary of safe parameters.
    """
    params = kwargs.copy()

    if rejects_temperature(model_name):
        if "temperature" in params:
            params.pop("temperature")
    else:
        if temperature is not None:
            params["temperature"] = temperature
        elif "temperature" not in params:
            params["temperature"] = 0.0

    if reasoning_effort is not None and is_reasoning_model(model_name):
        params["reasoning_effort"] = reasoning_effort

    return params
