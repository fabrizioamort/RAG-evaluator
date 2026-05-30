"""Playground Pydantic schemas for RAG testing and comparison."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse
from app.schemas.query_overrides import QueryOverrides

# Request schemas


class PlaygroundQueryRequest(BaseSchema):
    """Request schema for executing a playground query."""

    question: str = Field(min_length=1, max_length=2000, description="Question to ask the RAG systems")
    index_ids: list[UUID] = Field(
        min_length=1, max_length=4, description="List of index IDs to query (max 4 for comparison)"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    query_overrides: QueryOverrides | None = Field(
        default=None,
        description="Optional query-time overrides shared across selected indexes",
    )


# Response schemas


class RetrievedChunkResponse(BaseSchema):
    """Schema for a single retrieved chunk."""

    content: str = Field(description="Chunk content")
    document_id: str = Field(description="Source document ID")
    chunk_id: str = Field(description="Chunk identifier")
    score: float = Field(description="Relevance score")
    rank: int = Field(description="Rank in results")
    source: str = Field(description="Source document name/path")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalTraceStepResponse(BaseSchema):
    """Schema for a single step in the retrieval trace."""

    step_type: str = Field(description="Type of step (e.g., 'embedding', 'vector_search', 'fusion')")
    duration_ms: float = Field(description="Step duration in milliseconds")
    input_data: dict[str, Any] | None = Field(default=None, description="Input data for this step")
    output_summary: str | None = Field(default=None, description="Summary of step output")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Step-specific metadata")


class RetrievalTraceResponse(BaseSchema):
    """Schema for the retrieval trace."""

    strategy: str = Field(description="Retrieval strategy used (vector, hybrid, graph, agentic)")
    steps: list[RetrievalTraceStepResponse] = Field(default_factory=list, description="Execution steps")
    total_duration_ms: float = Field(description="Total retrieval duration in milliseconds")
    fusion_details: dict[str, Any] | None = Field(
        default=None, description="Fusion details for hybrid search"
    )


class RetrievedContextResponse(BaseSchema):
    """Schema for retrieved context."""

    chunks: list[str] = Field(description="Raw chunk contents for backward compatibility")
    chunk_details: list[RetrievedChunkResponse] = Field(description="Detailed chunk information")


class QueryMetrics(BaseSchema):
    """Metrics for a single query execution."""

    retrieval_time_ms: float = Field(description="Time spent on retrieval in milliseconds")
    generation_time_ms: float = Field(description="Time spent on generation in milliseconds")
    total_time_ms: float = Field(description="Total query time in milliseconds")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used")
    completion_tokens: int = Field(default=0, description="Completion tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost_usd: Decimal | None = Field(default=None, description="Estimated cost in USD")


class PlaygroundQueryResult(BaseSchema):
    """Result from a single RAG system."""

    index_id: UUID = Field(description="Index ID that was queried")
    index_name: str = Field(description="Index name")
    rag_type: str = Field(description="RAG type (vector_semantic, vector_hybrid, etc.)")
    knowledge_base_name: str = Field(description="Knowledge base name")

    answer: str | None = Field(default=None, description="Generated answer")
    retrieved_context: RetrievedContextResponse | None = Field(
        default=None, description="Retrieved context with chunks"
    )
    trace: RetrievalTraceResponse | None = Field(default=None, description="Retrieval trace")
    metrics: QueryMetrics | None = Field(default=None, description="Query metrics")
    effective_config_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Effective RAG configuration used for this query",
    )

    error: str | None = Field(default=None, description="Error message if query failed")
    success: bool = Field(default=True, description="Whether the query succeeded")


class PlaygroundQueryResponse(BaseSchema):
    """Response schema for a playground query with multiple results."""

    query_id: UUID = Field(description="Unique query ID for history tracking")
    question: str = Field(description="The question that was asked")
    results: list[PlaygroundQueryResult] = Field(description="Results from each RAG system")
    created_at: datetime = Field(description="When the query was executed")


# Index info schemas


class PlaygroundIndexInfo(BaseSchema):
    """Information about an index available for playground queries."""

    id: UUID = Field(description="Index ID")
    name: str = Field(description="Index name")
    rag_type: str = Field(description="RAG type")
    knowledge_base_id: UUID = Field(description="Knowledge base ID")
    knowledge_base_name: str = Field(description="Knowledge base name")
    project_id: UUID = Field(description="Project ID")
    project_name: str = Field(description="Project name")
    document_count: int = Field(description="Number of documents indexed")
    chunk_count: int = Field(description="Number of chunks indexed")
    status: str = Field(description="Index status")


class PlaygroundIndexList(BaseSchema):
    """List of indexes available for playground."""

    indexes: list[PlaygroundIndexInfo] = Field(description="Available indexes")


# Query history schemas


class PlaygroundQueryHistoryItem(BaseResponseSchema):
    """A query in the history list."""

    question: str = Field(description="Question asked")
    index_count: int = Field(description="Number of indexes queried")
    index_names: list[str] = Field(description="Names of indexes queried")
    success_count: int = Field(description="Number of successful results")
    total_time_ms: float | None = Field(default=None, description="Total query time")


class PlaygroundQueryHistoryList(PaginatedResponse):
    """Paginated list of query history items."""

    items: list[PlaygroundQueryHistoryItem]


class PlaygroundQueryDetail(BaseResponseSchema):
    """Full detail of a saved playground query."""

    question: str = Field(description="Question asked")
    top_k: int = Field(description="Top K parameter used")
    query_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Query-time overrides used",
    )
    results: list[PlaygroundQueryResult] = Field(description="Results from each RAG system")
