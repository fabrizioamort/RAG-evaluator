"""Known LLM models and capability metadata exposed by the platform."""

from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.common.llm_utils import is_reasoning_model


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for a model offered in provider dropdowns."""

    name: str
    supports_reasoning_effort: bool = False


MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    "openai": [
        ModelInfo("gpt-5.5", supports_reasoning_effort=True),
        ModelInfo("gpt-5.4-mini", supports_reasoning_effort=True),
        ModelInfo("gpt-5.4-nano", supports_reasoning_effort=True),
        ModelInfo("gpt-5.1", supports_reasoning_effort=True),
        ModelInfo("gpt-5-mini", supports_reasoning_effort=True),
        ModelInfo("gpt-5-nano", supports_reasoning_effort=True),
        ModelInfo("gpt-4o-mini"),
        ModelInfo("gpt-4o"),
    ],
    "openrouter": [
        ModelInfo("openai/gpt-5.5", supports_reasoning_effort=True),
        ModelInfo("openai/gpt-5.4-mini", supports_reasoning_effort=True),
        ModelInfo("openai/gpt-5.4-nano", supports_reasoning_effort=True),
        ModelInfo("deepseek/deepseek-v4-flash", supports_reasoning_effort=True),
        ModelInfo("anthropic/claude-sonnet-4"),
        ModelInfo("google/gemini-2.5-pro"),
        ModelInfo("meta-llama/llama-4-maverick"),
    ],
    "deepseek": [
        ModelInfo("deepseek-v4-flash", supports_reasoning_effort=True),
    ],
    "anthropic": [
        ModelInfo("claude-3-5-sonnet-20240620"),
        ModelInfo("claude-3-haiku-20240307"),
    ],
    "ollama": [
        ModelInfo("llama3"),
        ModelInfo("mistral"),
        ModelInfo("phi3"),
    ],
}


def get_models(provider: str) -> list[str]:
    """Return known model names for a provider."""
    return [model.name for model in MODEL_CATALOG.get(provider, [])]


def get_model_capabilities(provider: str) -> dict[str, dict[str, bool]]:
    """Return capability metadata keyed by model name for a provider."""
    return {
        model.name: {"supports_reasoning_effort": model.supports_reasoning_effort}
        for model in MODEL_CATALOG.get(provider, [])
    }


def model_supports_reasoning_effort(model_name: str) -> bool:
    """Return whether a known or recognizable custom model supports reasoning effort."""
    return is_reasoning_model(model_name)
