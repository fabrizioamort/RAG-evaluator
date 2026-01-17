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
    if any(x in model_lower for x in ["o1-", "o3-", "/o1", "/o3", "gpt-5"]) or model_lower in [
        "o1",
        "o3",
        "gpt-5",
    ]:
        return True

    return False


def get_safe_llm_params(
    model_name: str, temperature: float | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Get parameters that are safe for the specified model.

    Specifically handles removing temperature for reasoning models
    that do not support it.

    Args:
        model_name: Name of the LLM model.
        temperature: Requested temperature.
        **kwargs: Additional LLM parameters.

    Returns:
        Dictionary of safe parameters.
    """
    params = kwargs.copy()

    if is_reasoning_model(model_name):
        # Reasoning models don't support temperature (must be default or omitted)
        if "temperature" in params:
            params.pop("temperature")
    else:
        # Standard models use the provided temperature or default to 0.0
        if temperature is not None:
            params["temperature"] = temperature
        elif "temperature" not in params:
            params["temperature"] = 0.0

    return params
