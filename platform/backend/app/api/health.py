"""Health check endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    version: str


class HealthDetailResponse(HealthResponse):
    """Detailed health check response."""

    database_type: str
    storage_path: str


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Check if the service and database are healthy."""
    db_status = "disconnected"

    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        pass

    response = HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        version=settings.APP_VERSION,
    )

    return response


@router.get(
    "/health/detail",
    response_model=HealthDetailResponse,
    responses={
        200: {"description": "Detailed health information"},
    },
)
async def health_check_detail(db: AsyncSession = Depends(get_db)) -> HealthDetailResponse:
    """Get detailed health information."""
    db_status = "disconnected"
    db_type = "unknown"

    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
        db_type = "sqlite" if settings.is_sqlite else "postgresql"
    except Exception:
        pass

    return HealthDetailResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        version=settings.APP_VERSION,
        database_type=db_type,
        storage_path=settings.STORAGE_PATH,
    )
