"""RAG type and LLM provider registry."""

from typing import Any

from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_parameter_schema

from app.schemas.rag_config import LLMProviderInfo, RAGTypeInfo, RAGTypeParameter


class RAGRegistry:
    """Registry for available RAG implementations and LLM providers."""

    @staticmethod
    def get_rag_types() -> list[RAGTypeInfo]:
        """Get all available RAG implementation types from the core registry."""
        return [
            RAGTypeInfo(
                name=rag_type,
                display_name=metadata["name"],
                description=metadata["description"],
                requires_index=True,
                parameters=RAGRegistry._parameters_for_type(rag_type),
            )
            for rag_type, metadata in RAG_TYPES.items()
        ]

    @staticmethod
    def _parameters_for_type(rag_type: str) -> list[RAGTypeParameter]:
        schema = get_parameter_schema(rag_type)
        properties: dict[str, dict[str, Any]] = schema.get("properties", {})
        parameters: list[RAGTypeParameter] = []
        for name, definition in properties.items():
            param_type = definition.get("type", "string")
            if param_type == "number":
                param_type = "float"
            parameters.append(
                RAGTypeParameter(
                    name=name,
                    type=param_type,
                    description=definition.get("description", ""),
                    phase=definition["phase"],
                    required=bool(definition.get("required", False)),
                    default=definition.get("default"),
                    min_value=definition.get("minimum"),
                    max_value=definition.get("maximum"),
                    choices=definition.get("enum"),
                    platform_managed=bool(definition.get("platform_managed", False)),
                )
            )
        return parameters

    @staticmethod
    def get_llm_providers() -> list[LLMProviderInfo]:
        """Get all supported LLM providers."""
        return [
            LLMProviderInfo(
                name="openai",
                display_name="OpenAI",
                models=["gpt-5.1", "gpt-5-mini", "gpt-5-nano"],
                requires_api_key=True,
                supports_base_url=True,
            ),
            LLMProviderInfo(
                name="openrouter",
                display_name="OpenRouter",
                models=[
                    "openrouter/anthropic/claude-sonnet-4",
                    "openrouter/google/gemini-2.5-pro",
                    "openrouter/openai/gpt-5-mini",
                    "openrouter/meta-llama/llama-4-maverick",
                ],
                requires_api_key=True,
                supports_base_url=False,
            ),
            LLMProviderInfo(
                name="anthropic",
                display_name="Anthropic",
                models=["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],
                requires_api_key=True,
                supports_base_url=False,
            ),
            LLMProviderInfo(
                name="ollama",
                display_name="Ollama (Local)",
                models=["llama3", "mistral", "phi3"],
                requires_api_key=False,
                supports_base_url=True,
            ),
        ]
