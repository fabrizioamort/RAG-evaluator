"""Schemas for query-time RAG overrides."""

from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class QueryOverrides(BaseSchema):
    """Structured overrides that apply only when querying a ready index."""

    llm_model: str | None = Field(
        default=None,
        max_length=100,
        description="RAG generation/orchestration model override",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Retrieval top_k passed to query execution",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="RAG-type-specific query-phase parameter overrides",
    )
