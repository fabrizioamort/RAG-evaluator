"""Knowledge Base Index schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeBaseIndexCreate(BaseModel):
    """Request to create a new index."""

    rag_config_id: UUID
    name: str | None = None  # Auto-generated if not provided
    description: str | None = None


class KnowledgeBaseIndexResponse(BaseModel):
    """Response for index details."""

    id: UUID
    knowledge_base_id: UUID
    kb_version_id: UUID | None
    rag_config_id: UUID
    name: str
    description: str | None
    status: str
    physical_id: str
    storage_type: str
    config_snapshot: dict[str, Any]
    document_count: int
    chunk_count: int
    embedding_model: str | None
    build_started_at: datetime | None
    build_completed_at: datetime | None
    build_duration_seconds: float | None
    error_message: str | None
    created_at: datetime

    # Denormalized for display convenience
    knowledge_base_name: str | None = None
    rag_config_name: str | None = None
    project_id: UUID | None = None


class KnowledgeBaseIndexList(BaseModel):
    """Paginated list of indexes."""

    items: list[KnowledgeBaseIndexResponse]
    total: int
    offset: int
    limit: int


class IndexBuildProgress(BaseModel):
    """Progress event for index building."""

    status: str  # building, processing_doc, embedding, storing, complete, failed
    current: int
    total: int
    current_document: str | None = None
    message: str | None = None


class IndexArchiveRequest(BaseModel):
    """Request to archive an index."""
    
    reason: str | None = None


class IndexRetryRequest(BaseModel):
    """Request to retry a failed index build."""
    
    force: bool = False


class KnowledgeBaseIndexSummary(BaseModel):
    """Minimal index info for lists."""

    id: UUID
    name: str
    status: str
    created_at: datetime
