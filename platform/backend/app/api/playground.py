"""Playground API endpoints for interactive RAG testing and comparison."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession, Pagination
from app.schemas.playground import (
    PlaygroundIndexList,
    PlaygroundQueryDetail,
    PlaygroundQueryHistoryList,
    PlaygroundQueryRequest,
    PlaygroundQueryResponse,
)
from app.services.playground_service import get_playground_service
from app.utils.logging_config import get_logger

router = APIRouter(prefix="/playground", tags=["Playground"])
logger = get_logger(__name__)


@router.get(
    "/indexes",
    response_model=PlaygroundIndexList,
    summary="Get available indexes for playground",
)
async def get_playground_indexes(
    db: DbSession,
    project_id: UUID | None = Query(default=None, description="Filter by project ID"),
    kb_id: UUID | None = Query(default=None, description="Filter by knowledge base ID"),
) -> PlaygroundIndexList:
    """Get list of indexes available for playground queries.

    Only returns indexes with status='ready' that can be queried.
    """
    service = get_playground_service(db)
    indexes = await service.get_available_indexes(project_id=project_id, kb_id=kb_id)

    return PlaygroundIndexList(indexes=indexes)


@router.post(
    "/query",
    response_model=PlaygroundQueryResponse,
    summary="Execute a playground query",
)
async def execute_playground_query(
    db: DbSession,
    request: PlaygroundQueryRequest,
) -> PlaygroundQueryResponse:
    """Execute a query against one or more RAG indexes.

    Supports querying up to 4 indexes simultaneously for comparison.
    Results include the generated answer, retrieved chunks, and retrieval trace
    for each index.

    The query is automatically saved to history for later review.
    """
    service = get_playground_service(db)

    try:
        response = await service.execute_query(
            question=request.question,
            index_ids=request.index_ids,
            top_k=request.top_k,
            query_overrides=request.query_overrides.model_dump(exclude_none=True)
            if request.query_overrides
            else None,
        )
        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.exception("Playground query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        ) from e


@router.get(
    "/history",
    response_model=PlaygroundQueryHistoryList,
    summary="Get query history",
)
async def get_query_history(
    db: DbSession,
    pagination: Pagination,
) -> PlaygroundQueryHistoryList:
    """Get paginated list of past playground queries.

    Returns summary information for each query including the question,
    indexes queried, and success rate.
    """
    service = get_playground_service(db)
    items, total = await service.get_query_history(
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return PlaygroundQueryHistoryList(
        items=items,
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get(
    "/history/{query_id}",
    response_model=PlaygroundQueryDetail,
    summary="Get query detail",
)
async def get_query_detail(
    db: DbSession,
    query_id: UUID,
) -> PlaygroundQueryDetail:
    """Get full details of a saved playground query.

    Returns the complete query results including answers, chunks, and traces
    for all indexes that were queried.
    """
    service = get_playground_service(db)
    detail = await service.get_query_detail(query_id)

    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found",
        )

    return detail


@router.delete(
    "/history/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a query from history",
)
async def delete_query(
    db: DbSession,
    query_id: UUID,
) -> None:
    """Delete a query from the history.

    This permanently removes the query and its results.
    """
    service = get_playground_service(db)
    deleted = await service.delete_query(query_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found",
        )
