"""Base Pydantic schemas and common utilities."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class BaseResponseSchema(BaseSchema):
    """Base response schema with id and created_at."""

    id: UUID
    created_at: datetime


class BaseResponseWithUpdateSchema(BaseResponseSchema):
    """Base response schema with id, created_at, and updated_at."""

    updated_at: datetime


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Number of items to return")


class PaginatedResponse(BaseSchema):
    """Generic paginated response wrapper."""

    offset: int = Field(description="Number of items skipped")
    limit: int = Field(description="Maximum items returned")
    total: int = Field(description="Total number of items available")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str


class ErrorDetail(BaseModel):
    """Error detail for API responses."""

    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    field: str | None = Field(default=None, description="Field that caused the error")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: list[ErrorDetail] | None = Field(default=None, description="Additional error details")
