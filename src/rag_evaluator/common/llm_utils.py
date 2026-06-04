"""Utilities for working with different LLM models and their parameters."""

from typing import Any


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model (like OpenAI o1 or o3).

    Reasoning models often have restrictions on parameters like temperature.

    Args:
        model_name: Name of the LLM model.

    Returns:
        True if the model is identified as a reasoning model.
    """
    if not model_name:
        return False

    model_lower = model_name.lower()

    # OpenAI reasoning models
    if any(
        x in model_lower
        for x in ["o1-", "o3-", "/o1", "/o3", "gpt-5", "deepseek-v4-flash"]
    ) or model_lower in [
        "o1",
        "o3",
        "gpt-5",
    ]:
        return True

    return False


def get_safe_llm_params(
    model_name: str,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get parameters that are safe for the specified model.

    Handles removing temperature for reasoning models and forwards
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

    if is_reasoning_model(model_name):
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
