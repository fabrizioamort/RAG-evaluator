"""RAG type and LLM provider registry."""

from typing import Any

from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_parameter_schema

from app.schemas.rag_config import LLMProviderInfo, RAGTypeInfo, RAGTypeParameter
from app.services.llm_model_catalog import get_model_capabilities, get_models


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
                models=get_models("openai"),
                model_capabilities=get_model_capabilities("openai"),
                requires_api_key=True,
                supports_base_url=True,
                supports_embeddings=True,
            ),
            LLMProviderInfo(
                name="openrouter",
                display_name="OpenRouter",
                models=get_models("openrouter"),
                model_capabilities=get_model_capabilities("openrouter"),
                requires_api_key=True,
                supports_base_url=False,
                supports_embeddings=False,
            ),
            LLMProviderInfo(
                name="deepseek",
                display_name="DeepSeek",
                models=get_models("deepseek"),
                model_capabilities=get_model_capabilities("deepseek"),
                requires_api_key=True,
                supports_base_url=False,
                supports_embeddings=False,
            ),
            LLMProviderInfo(
                name="anthropic",
                display_name="Anthropic",
                models=get_models("anthropic"),
                model_capabilities=get_model_capabilities("anthropic"),
                requires_api_key=True,
                supports_base_url=False,
                supports_embeddings=False,
            ),
            LLMProviderInfo(
                name="ollama",
                display_name="Ollama (Local)",
                models=get_models("ollama"),
                model_capabilities=get_model_capabilities("ollama"),
                requires_api_key=False,
                supports_base_url=True,
                supports_embeddings=True,
            ),
            LLMProviderInfo(
                name="vertex_ai",
                display_name="Google Vertex AI (Gemini)",
                models=get_models("vertex_ai"),
                model_capabilities=get_model_capabilities("vertex_ai"),
                requires_api_key=False,
                supports_base_url=False,
                supports_embeddings=True,
                requires_gcp_project=True,
                accepts_freeform_model=True,
            ),
        ]
