"""RAG configuration Pydantic schemas."""

from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse


class RAGConfigBase(BaseSchema):
    """Base RAG config schema."""

    name: str = Field(min_length=1, max_length=255, description="Config name")
    rag_type: str = Field(
        max_length=50,
        description="RAG implementation type (vector_semantic, hybrid, graph, filesystem)",
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="RAG-specific parameters")
    llm_provider: str = Field(
        default="openai",
        max_length=50,
        description="LLM provider (openai, anthropic, ollama)",
    )
    llm_model: str = Field(default="gpt-4o-mini", max_length=100, description="LLM model name")
    llm_base_url: str | None = Field(
        default=None, max_length=500, description="Custom LLM API base URL"
    )


class RAGConfigCreate(RAGConfigBase):
    """Schema for creating a RAG config."""

    pass


class RAGConfigUpdate(BaseSchema):
    """Schema for updating a RAG config."""

    name: str | None = Field(default=None, min_length=1, max_length=255, description="Config name")
    parameters: dict[str, Any] | None = Field(default=None, description="RAG-specific parameters")
    llm_provider: str | None = Field(default=None, max_length=50, description="LLM provider")
    llm_model: str | None = Field(default=None, max_length=100, description="LLM model")
    llm_base_url: str | None = Field(
        default=None, max_length=500, description="Custom LLM API base URL"
    )


class RAGConfigResponse(RAGConfigBase, BaseResponseSchema):
    """Schema for RAG config response."""

    project_id: UUID = Field(description="Parent project ID")


class RAGConfigSummary(BaseSchema):
    """Minimal RAG config info for selection."""

    id: UUID
    name: str
    rag_type: str
    llm_provider: str
    llm_model: str


class RAGConfigList(PaginatedResponse):
    """Paginated list of RAG configs."""

    items: list[RAGConfigResponse]


class RAGTypeParameter(BaseSchema):
    """Schema for a RAG type parameter definition."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, integer, float, boolean)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether required")
    default: Any = Field(default=None, description="Default value")
    min_value: float | None = Field(default=None, description="Minimum value")
    max_value: float | None = Field(default=None, description="Maximum value")
    choices: list[str] | None = Field(default=None, description="Allowed values")


class RAGTypeInfo(BaseSchema):
    """Information about a RAG implementation type."""

    name: str = Field(description="Type identifier")
    display_name: str = Field(description="Human-readable name")
    description: str = Field(description="Type description")
    parameters: list[RAGTypeParameter] = Field(
        default_factory=list, description="Available parameters"
    )
    requires_index: bool = Field(default=True, description="Whether KB indexing is required")


class LLMProviderInfo(BaseSchema):
    """Information about an LLM provider."""

    name: str = Field(description="Provider identifier")
    display_name: str = Field(description="Human-readable name")
    models: list[str] = Field(description="Available model names")
    requires_api_key: bool = Field(default=True, description="Whether API key is required")
    supports_base_url: bool = Field(
        default=False, description="Whether custom base URL is supported"
    )
