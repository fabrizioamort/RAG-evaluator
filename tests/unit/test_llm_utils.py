"""Tests for LLM utilities."""

from rag_evaluator.common.llm_utils import get_safe_llm_params, is_reasoning_model


def test_is_reasoning_model():
    """Test reasoning model identification."""
    assert is_reasoning_model("o1-preview") is True
    assert is_reasoning_model("o1-mini") is True
    assert is_reasoning_model("o3-mini-2024-12-31") is True
    assert is_reasoning_model("openai/o1") is True
    assert is_reasoning_model("azure/o1-mini") is True
    assert is_reasoning_model("gpt-5-nano") is True
    assert is_reasoning_model("gpt-5.1") is True
    
    assert is_reasoning_model("gpt-4o") is False
    assert is_reasoning_model("gpt-3.5-turbo") is False
    assert is_reasoning_model("claude-3-5-sonnet") is False
    assert is_reasoning_model(None) is False
    assert is_reasoning_model("") is False


def test_get_safe_llm_params_reasoning_model():
    """Test parameter cleaning for reasoning models."""
    # Temperature should be removed
    params = get_safe_llm_params("o1-mini", temperature=0.0, max_tokens=100)
    assert "temperature" not in params
    assert params["max_tokens"] == 100

    params = get_safe_llm_params("gpt-5-nano", temperature=0.0)
    assert "temperature" not in params

    params = get_safe_llm_params("o1-mini", max_tokens=100)
    assert "temperature" not in params
    
    # kwargs temperature should also be removed
    params = get_safe_llm_params("o1-mini", temperature=0.5, top_p=0.9)
    assert "temperature" not in params
    assert params["top_p"] == 0.9


def test_get_safe_llm_params_standard_model():
    """Test parameter handling for standard models."""
    # Temperature should be kept or defaulted
    params = get_safe_llm_params("gpt-4o", temperature=0.5, max_tokens=100)
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100

    # Default temperature is 0.0
    params = get_safe_llm_params("gpt-4o", max_tokens=100)
    assert params["temperature"] == 0.0
    
    # kwargs temperature should be kept
    params = get_safe_llm_params("gpt-4o", temperature=0.7, top_p=0.9)
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9
