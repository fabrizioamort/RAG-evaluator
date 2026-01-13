"""Knowledge base and document Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse


class DocumentBase(BaseSchema):
    """Base document schema."""

    filename: str = Field(max_length=255, description="Original filename")
    content_type: str | None = Field(default=None, description="MIME content type")
    size_bytes: int | None = Field(default=None, ge=0, description="File size in bytes")


class DocumentResponse(DocumentBase, BaseResponseSchema):
    """Schema for document response."""

    knowledge_base_id: UUID = Field(description="Parent knowledge base ID")
    file_path: str = Field(description="Storage path")
    checksum: str | None = Field(default=None, description="SHA256 checksum")
    status: str = Field(description="Document processing status")


class DocumentUploadResponse(BaseSchema):
    """Response after uploading documents."""

    uploaded: list[DocumentResponse] = Field(description="Successfully uploaded docs")
    failed: list[dict[str, str]] = Field(
        default_factory=list,
        description="Failed uploads with filename and error",
    )
    total_size_bytes: int = Field(description="Total size of uploaded documents")


class KnowledgeBaseBase(BaseSchema):
    """Base knowledge base schema."""

    name: str = Field(min_length=1, max_length=255, description="Knowledge base name")
    description: str | None = Field(default=None, description="Description")
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        alias="metadata",
        description="Additional metadata",
    )


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for creating a knowledge base."""

    pass


class KnowledgeBaseUpdate(BaseSchema):
    """Schema for updating a knowledge base."""

    name: str | None = Field(
        default=None, min_length=1, max_length=255, description="Knowledge base name"
    )
    description: str | None = Field(default=None, description="Description")
    metadata_: dict[str, Any] | None = Field(
        default=None, alias="metadata", description="Additional metadata"
    )


class KnowledgeBaseResponse(KnowledgeBaseBase, BaseResponseSchema):
    """Schema for knowledge base response."""

    project_id: UUID = Field(description="Parent project ID")
    status: str = Field(description="KB status (pending, indexing, ready, error)")
    current_version: int = Field(description="Current version number")
    storage_path: str | None = Field(default=None, description="Document storage path")
    index_path: str | None = Field(default=None, description="Index storage path")
    document_count: int = Field(default=0, description="Number of documents")


class KnowledgeBaseWithDocuments(KnowledgeBaseResponse):
    """Knowledge base response including documents."""

    documents: list[DocumentResponse] = Field(
        default_factory=list, description="Documents in this KB"
    )


class KnowledgeBaseVersionResponse(BaseResponseSchema):
    """Schema for KB version response."""

    knowledge_base_id: UUID = Field(description="Parent knowledge base ID")
    version_number: int = Field(description="Version number")
    change_type: str = Field(description="Type of change (initial, add, remove, update)")
    document_snapshot: list[dict[str, Any]] = Field(
        description="Snapshot of documents at this version"
    )
    change_description: str | None = Field(default=None, description="Change notes")


class KnowledgeBaseSummary(BaseSchema):
    """Minimal KB info for lists and references."""

    id: UUID
    name: str
    status: str
    current_version: int
    document_count: int
    created_at: datetime


class KnowledgeBaseList(PaginatedResponse):
    """Paginated list of knowledge bases."""

    items: list[KnowledgeBaseResponse]
