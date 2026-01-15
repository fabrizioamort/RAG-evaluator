"""Standardized error response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ValidationErrorDetail(BaseModel):
    """Detail of a single validation error."""

    loc: list[str | int] = Field(..., description="Location of the error (path)")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ErrorResponse(BaseModel):
    """Standardized error response body."""

    detail: str = Field(..., description="High-level error description")
    request_id: str | None = Field(None, description="Request correlation ID")
    errors: list[Any] | None = Field(
        None, description="Detailed error items (e.g., specific validation failures)"
    )
