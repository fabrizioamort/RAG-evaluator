"""Project Pydantic schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.schemas.base import (
    BaseResponseWithUpdateSchema,
    BaseSchema,
    PaginatedResponse,
)


class ProjectBase(BaseSchema):
    """Base project schema with common fields."""

    name: str = Field(min_length=1, max_length=255, description="Project name")
    description: str | None = Field(default=None, description="Project description")
    tags: list[str] = Field(default_factory=list, description="Project tags")


class ProjectCreate(ProjectBase):
    """Schema for creating a project."""

    pass


class ProjectUpdate(BaseSchema):
    """Schema for updating a project."""

    name: str | None = Field(default=None, min_length=1, max_length=255, description="Project name")
    description: str | None = Field(default=None, description="Project description")
    status: str | None = Field(
        default=None,
        pattern="^(active|archived)$",
        description="Project status",
    )
    tags: list[str] | None = Field(default=None, description="Project tags")


class ProjectResponse(ProjectBase, BaseResponseWithUpdateSchema):
    """Schema for project response."""

    status: str = Field(description="Project status")

    # Counts from relationships (computed)
    knowledge_base_count: int = Field(default=0, description="Number of knowledge bases")
    test_set_count: int = Field(default=0, description="Number of test sets")
    rag_config_count: int = Field(default=0, description="Number of RAG configs")
    evaluation_count: int = Field(default=0, description="Number of evaluations")


class ProjectSummary(BaseSchema):
    """Minimal project info for lists and references."""

    id: UUID
    name: str
    status: str
    created_at: datetime


class ProjectList(PaginatedResponse):
    """Paginated list of projects."""

    items: list[ProjectResponse]
