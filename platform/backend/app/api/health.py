"""Health check endpoint."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.test_set import TestSet

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


class DashboardStats(BaseModel):
    """Dashboard statistics."""

    projects: int
    knowledge_bases: int
    test_sets: int
    evaluations: int
    completed_evaluations: int
    running_evaluations: int


class RecentActivityItem(BaseModel):
    """Recent activity item."""

    id: str
    type: str  # 'project', 'evaluation', 'knowledge_base', 'test_set'
    action: str  # 'created', 'completed', 'started', etc.
    name: str
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class RecentActivityResponse(BaseModel):
    """Recent activity response."""

    items: list[RecentActivityItem]


@router.get(
    "/stats",
    response_model=DashboardStats,
    responses={
        200: {"description": "Dashboard statistics"},
    },
)
async def get_stats(db: AsyncSession = Depends(get_db)) -> DashboardStats:
    """Get dashboard statistics."""
    # Count projects
    projects_result = await db.execute(select(func.count()).select_from(Project))
    projects_count = projects_result.scalar() or 0

    # Count knowledge bases
    kb_result = await db.execute(select(func.count()).select_from(KnowledgeBase))
    kb_count = kb_result.scalar() or 0

    # Count test sets
    ts_result = await db.execute(select(func.count()).select_from(TestSet))
    ts_count = ts_result.scalar() or 0

    # Count evaluations
    eval_result = await db.execute(select(func.count()).select_from(Evaluation))
    eval_count = eval_result.scalar() or 0

    # Count completed evaluations
    completed_result = await db.execute(
        select(func.count()).select_from(Evaluation).where(Evaluation.status == "completed")
    )
    completed_count = completed_result.scalar() or 0

    # Count running evaluations
    running_result = await db.execute(
        select(func.count()).select_from(Evaluation).where(Evaluation.status == "running")
    )
    running_count = running_result.scalar() or 0

    return DashboardStats(
        projects=projects_count,
        knowledge_bases=kb_count,
        test_sets=ts_count,
        evaluations=eval_count,
        completed_evaluations=completed_count,
        running_evaluations=running_count,
    )


@router.get(
    "/recent-activity",
    response_model=RecentActivityResponse,
    responses={
        200: {"description": "Recent activity"},
    },
)
async def get_recent_activity(
    db: AsyncSession = Depends(get_db),
    limit: int = 10,
) -> RecentActivityResponse:
    """Get recent activity across all entities."""
    items: list[RecentActivityItem] = []

    # Recent projects
    projects_query = select(Project).order_by(Project.created_at.desc()).limit(limit)
    projects_result = await db.execute(projects_query)
    for p in projects_result.scalars():
        items.append(
            RecentActivityItem(
                id=str(p.id),
                type="project",
                action="created",
                name=p.name,
                timestamp=p.created_at,
            )
        )

    # Recent evaluations (both created and completed)
    evals_query = select(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit)
    evals_result = await db.execute(evals_query)
    for e in evals_result.scalars():
        action = "started"
        timestamp = e.created_at
        if e.status == "completed" and e.completed_at:
            action = "completed"
            timestamp = e.completed_at
        elif e.status == "failed":
            action = "failed"
        elif e.status == "running":
            action = "running"

        items.append(
            RecentActivityItem(
                id=str(e.id),
                type="evaluation",
                action=action,
                name=f"Evaluation #{str(e.id)[:8]}",
                timestamp=timestamp,
                metadata={
                    "status": e.status,
                    "pass_rate": e.pass_rate,
                },
            )
        )

    # Recent knowledge bases
    kb_query = select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).limit(limit)
    kb_result = await db.execute(kb_query)
    for kb in kb_result.scalars():
        items.append(
            RecentActivityItem(
                id=str(kb.id),
                type="knowledge_base",
                action="created" if kb.status == "pending" else kb.status,
                name=kb.name,
                timestamp=kb.created_at,
                metadata={"status": kb.status},
            )
        )

    # Sort by timestamp descending and limit
    items.sort(key=lambda x: x.timestamp, reverse=True)
    items = items[:limit]

    return RecentActivityResponse(items=items)
