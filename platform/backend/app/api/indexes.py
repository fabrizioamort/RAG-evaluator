"""Knowledge Base Index API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.knowledge_base import KnowledgeBase
from app.models.rag_config import RAGConfig
from app.schemas.knowledge_base_index import (
    KnowledgeBaseIndexCreate,
    KnowledgeBaseIndexList,
    KnowledgeBaseIndexResponse,
    IndexArchiveRequest,
    IndexRetryRequest,
)
from app.services.index_build_service import IndexBuildService
from app.services.job_event_log import JobEventLog, get_job_event_log
from app.utils.exceptions import NotFoundError, BadRequestError
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Indexes"])
logger = get_logger(__name__)


def get_index_build_service(
    db: DbSession,
) -> IndexBuildService:
    # Use global event log or create new one
    event_log = get_job_event_log()
    return IndexBuildService(db, event_log)


def _index_to_response(index: KnowledgeBaseIndex) -> KnowledgeBaseIndexResponse:
    """Convert KnowledgeBaseIndex model to response schema."""
    return KnowledgeBaseIndexResponse(
        id=index.id,
        knowledge_base_id=index.knowledge_base_id,
        kb_version_id=index.kb_version_id,
        rag_config_id=index.rag_config_id,
        name=index.name,
        description=index.description,
        status=index.status,
        physical_id=index.physical_id,
        storage_type=index.storage_type,
        config_snapshot=index.config_snapshot,
        document_count=index.document_count,
        chunk_count=index.chunk_count,
        embedding_model=index.embedding_model,
        build_started_at=index.build_started_at,
        build_completed_at=index.build_completed_at,
        build_duration_seconds=index.build_duration_seconds,
        error_message=index.error_message,
        created_at=index.created_at,
        knowledge_base_name=index.knowledge_base.name if index.knowledge_base else None,
        rag_config_name=index.rag_config.name if index.rag_config else None,
        project_id=index.knowledge_base.project_id if index.knowledge_base else None,
    )


@router.get(
    "/indexes",
    response_model=KnowledgeBaseIndexList,
    summary="List indexes",
    description="Retrieve a paginated list of indexes, optionally filtered.",
)
async def list_indexes(
    db: DbSession,
    pagination: Pagination,
    kb_id: UUID | None = None,
    project_id: UUID | None = None,
    status: str | None = None,
) -> KnowledgeBaseIndexList:
    """List all indexes."""
    query = select(KnowledgeBaseIndex).options(
        selectinload(KnowledgeBaseIndex.knowledge_base),
        selectinload(KnowledgeBaseIndex.rag_config),
    )

    if kb_id:
        query = query.where(KnowledgeBaseIndex.knowledge_base_id == kb_id)
    
    if project_id:
        query = query.join(KnowledgeBase).where(KnowledgeBase.project_id == project_id)

    if status:
        query = query.where(KnowledgeBaseIndex.status == status)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(KnowledgeBaseIndex.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    result = await db.execute(query)
    indexes = result.scalars().all()

    return KnowledgeBaseIndexList(
        items=[_index_to_response(idx) for idx in indexes],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.post(
    "/knowledge-bases/{kb_id}/indexes",
    response_model=KnowledgeBaseIndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create and build index",
    description="Create a new index record and start the background build process.",
)
async def create_index(
    db: DbSession,
    kb_id: UUID,
    request: KnowledgeBaseIndexCreate,
    background_tasks: BackgroundTasks,
    # In a real app, we'd use a dependency to get the service factory
) -> KnowledgeBaseIndexResponse:
    """Create a new index."""
    # We construct services here. In production, use Depends()
    event_log = get_job_event_log()
    service = IndexBuildService(db, event_log)

    try:
        index = await service.create_index(
            kb_id=kb_id,
            rag_config_id=request.rag_config_id,
            name=request.name,
            description=request.description,
        )
    except ValueError as e:
        raise BadRequestError(detail=str(e))

    # Trigger build in background
    from app.database import get_db_context
    
    async def run_build_task(index_id: UUID) -> None:
        async with get_db_context() as task_db:
            task_event_log = get_job_event_log() # Use singleton/default
            task_service = IndexBuildService(task_db, task_event_log)
            await task_service.build_index(index_id)

    background_tasks.add_task(run_build_task, index.id)

    # Re-fetch with relationships for response
    query = (
        select(KnowledgeBaseIndex)
        .where(KnowledgeBaseIndex.id == index.id)
        .options(
            selectinload(KnowledgeBaseIndex.knowledge_base),
            selectinload(KnowledgeBaseIndex.rag_config),
        )
    )
    result = await db.execute(query)
    refreshed_index = result.scalar_one()

    return _index_to_response(refreshed_index)


@router.get(
    "/indexes/{index_id}",
    response_model=KnowledgeBaseIndexResponse,
    summary="Get index details",
    description="Retrieve details of a specific index.",
)
async def get_index(
    db: DbSession,
    index_id: UUID,
) -> KnowledgeBaseIndexResponse:
    """Get index details."""
    query = (
        select(KnowledgeBaseIndex)
        .where(KnowledgeBaseIndex.id == index_id)
        .options(
            selectinload(KnowledgeBaseIndex.knowledge_base),
            selectinload(KnowledgeBaseIndex.rag_config),
        )
    )
    result = await db.execute(query)
    index = result.scalar_one_or_none()

    if not index:
        raise NotFoundError(detail="Index not found")

    return _index_to_response(index)


@router.delete(
    "/indexes/{index_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete index",
    description="Delete an index and its physical storage. Fails if evaluations exist.",
)
async def delete_index(
    db: DbSession,
    index_id: UUID,
) -> None:
    """Delete an index."""
    event_log = get_job_event_log()
    service = IndexBuildService(db, event_log)

    try:
        await service.delete_index(index_id)
    except ValueError as e:
        # Check if it's the dependency error
        if "evaluations" in str(e):
            raise BadRequestError(detail=str(e)) # Should be 409 Conflict ideally
        raise NotFoundError(detail=str(e))


@router.get(
    "/indexes/{index_id}/stream",
    summary="Stream build progress",
    description="SSE stream for index build progress.",
)
async def stream_index_build(
    request: Request,
    db: DbSession,
    index_id: UUID,
) -> EventSourceResponse:
    """Stream build events."""
    # Verify index exists
    query = select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == index_id)
    result = await db.execute(query)
    if not result.scalar_one_or_none():
        raise NotFoundError(detail="Index not found")

    event_log = get_job_event_log()
    
    return EventSourceResponse(
        event_log.subscribe(index_id)
    )


@router.post(
    "/indexes/{index_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry index build",
    description="Retry a failed index build.",
)
async def retry_index_build(
    db: DbSession,
    index_id: UUID,
    background_tasks: BackgroundTasks,
    request: IndexRetryRequest | None = None,
) -> dict[str, str]:
    """Retry index build."""
    query = select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == index_id)
    result = await db.execute(query)
    index = result.scalar_one_or_none()

    if not index:
        raise NotFoundError(detail="Index not found")

    if index.status not in ["failed", "pending"]:
         raise BadRequestError(detail=f"Cannot retry index in '{index.status}' status")

    from app.database import get_db_context
    
    async def run_build_task(idx_id: UUID) -> None:
        async with get_db_context() as task_db:
            task_event_log = get_job_event_log()
            task_service = IndexBuildService(task_db, task_event_log)
            await task_service.build_index(idx_id)

    background_tasks.add_task(run_build_task, index.id)

    return {"status": "accepted", "message": "Build retry started"}


@router.post(
    "/indexes/{index_id}/archive",
    response_model=KnowledgeBaseIndexResponse,
    summary="Archive index",
    description="Archive an index (soft delete).",
)
async def archive_index(
    db: DbSession,
    index_id: UUID,
    request: IndexArchiveRequest | None = None,
) -> KnowledgeBaseIndexResponse:
    """Archive an index."""
    query = (
        select(KnowledgeBaseIndex)
        .where(KnowledgeBaseIndex.id == index_id)
        .options(
            selectinload(KnowledgeBaseIndex.knowledge_base),
            selectinload(KnowledgeBaseIndex.rag_config),
        )
    )
    result = await db.execute(query)
    index = result.scalar_one_or_none()

    if not index:
        raise NotFoundError(detail="Index not found")

    # Update status to archived
    index.status = "archived"
    await db.commit()
    await db.refresh(index)

    return _index_to_response(index)