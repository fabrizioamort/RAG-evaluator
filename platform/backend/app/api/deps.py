"""Common API dependencies for dependency injection."""

from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.base import PaginationParams

# Type alias for database session dependency
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_pagination_params(
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of items to return"),
) -> PaginationParams:
    """Common pagination parameters dependency."""
    return PaginationParams(offset=offset, limit=limit)


# Type alias for pagination dependency
Pagination = Annotated[PaginationParams, Depends(get_pagination_params)]


def get_tags_filter(
    tags: list[str] | None = Query(default=None, description="Filter by tags (comma-separated)"),
) -> list[str] | None:
    """Parse tags filter from query parameter."""
    return tags


# Type alias for tags filter dependency
TagsFilter = Annotated[list[str] | None, Depends(get_tags_filter)]


def get_status_filter(
    status: str | None = Query(
        default=None,
        pattern="^(active|archived)$",
        description="Filter by status",
    ),
) -> str | None:
    """Parse status filter from query parameter."""
    return status


# Type alias for status filter dependency
StatusFilter = Annotated[str | None, Depends(get_status_filter)]
