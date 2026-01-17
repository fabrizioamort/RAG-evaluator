"""Comparisons API endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination
from app.models.comparison import Comparison
from app.models.evaluation import Evaluation
from app.schemas.comparison import (
    AggregateMetrics,
    ComparisonCreate,
    ComparisonDetail,
    ComparisonList,
    ComparisonResponse,
    PerQuestionDelta,
)
from app.services.comparison_service import get_comparison_service
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Comparisons"])
logger = get_logger(__name__)


async def _get_comparison_or_404(db: DbSession, comparison_id: UUID) -> Comparison:
    """Get comparison by ID or raise 404."""
    query = (
        select(Comparison)
        .where(Comparison.id == comparison_id)
        .options(selectinload(Comparison.baseline_evaluation))
    )
    result = await db.execute(query)
    comparison = result.scalar_one_or_none()
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison with id {comparison_id} not found",
        )
    return comparison


def _comparison_to_response(comparison: Comparison) -> ComparisonResponse:
    """Convert Comparison model to ComparisonResponse schema."""
    aggregate_metrics = None
    if comparison.aggregate_metrics:
        aggregate_metrics = AggregateMetrics(**comparison.aggregate_metrics)

    # Convert string UUIDs to UUID objects
    compared_ids = [UUID(id_str) for id_str in comparison.compared_evaluation_ids]

    return ComparisonResponse(
        id=comparison.id,
        project_id=comparison.project_id,
        name=comparison.name,
        description=comparison.description,
        baseline_evaluation_id=comparison.baseline_evaluation_id,
        compared_evaluation_ids=compared_ids,
        aggregate_metrics=aggregate_metrics,
        created_at=comparison.created_at,
    )


def _comparison_to_detail(comparison: Comparison) -> ComparisonDetail:
    """Convert Comparison model to ComparisonDetail schema with per-question deltas."""
    aggregate_metrics = None
    if comparison.aggregate_metrics:
        aggregate_metrics = AggregateMetrics(**comparison.aggregate_metrics)

    per_question_deltas = None
    if comparison.per_question_deltas:
        per_question_deltas = [PerQuestionDelta(**d) for d in comparison.per_question_deltas]

    # Convert string UUIDs to UUID objects
    compared_ids = [UUID(id_str) for id_str in comparison.compared_evaluation_ids]

    return ComparisonDetail(
        id=comparison.id,
        project_id=comparison.project_id,
        name=comparison.name,
        description=comparison.description,
        baseline_evaluation_id=comparison.baseline_evaluation_id,
        compared_evaluation_ids=compared_ids,
        aggregate_metrics=aggregate_metrics,
        per_question_deltas=per_question_deltas,
        created_at=comparison.created_at,
    )


@router.post(
    "/comparisons",
    response_model=ComparisonResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new comparison",
)
async def create_comparison(
    db: DbSession,
    comparison_data: ComparisonCreate,
) -> ComparisonResponse:
    """Create a comparison between evaluations.

    Compares one or more evaluations against a baseline evaluation.
    All evaluations must belong to the same project and be completed.
    """
    # 1. Get baseline evaluation
    baseline_query = (
        select(Evaluation)
        .where(Evaluation.id == comparison_data.baseline_evaluation_id)
        .options(selectinload(Evaluation.rag_config))
    )
    baseline_result = await db.execute(baseline_query)
    baseline_eval = baseline_result.scalar_one_or_none()

    if not baseline_eval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline evaluation {comparison_data.baseline_evaluation_id} not found",
        )

    if baseline_eval.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Baseline evaluation must be completed",
        )

    # 2. Get compared evaluations
    compared_query = (
        select(Evaluation)
        .where(Evaluation.id.in_(comparison_data.compared_evaluation_ids))
        .options(selectinload(Evaluation.rag_config))
    )
    compared_result = await db.execute(compared_query)
    compared_evals = list(compared_result.scalars().all())

    if len(compared_evals) != len(comparison_data.compared_evaluation_ids):
        found_ids = {e.id for e in compared_evals}
        missing_ids = [
            str(id) for id in comparison_data.compared_evaluation_ids if id not in found_ids
        ]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluations not found: {', '.join(missing_ids)}",
        )

    # 3. Validate all evaluations belong to same project and are completed
    project_id = baseline_eval.project_id
    for eval in compared_evals:
        if eval.project_id != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation {eval.id} belongs to a different project",
            )
        if eval.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation {eval.id} is not completed",
            )

    # 4. Check baseline is not in compared list
    if baseline_eval.id in comparison_data.compared_evaluation_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Baseline evaluation cannot be in the compared list",
        )

    # 5. Create comparison
    service = get_comparison_service()
    comparison = await service.create_comparison(
        db=db,
        baseline_eval=baseline_eval,
        compared_evals=compared_evals,
        name=comparison_data.name,
        description=comparison_data.description,
    )

    logger.info(
        "Created comparison",
        comparison_id=str(comparison.id),
        project_id=str(project_id),
    )

    return _comparison_to_response(comparison)


@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonDetail,
    summary="Get comparison details",
)
async def get_comparison(
    db: DbSession,
    comparison_id: UUID,
) -> ComparisonDetail:
    """Get detailed comparison results including per-question deltas."""
    comparison = await _get_comparison_or_404(db, comparison_id)
    return _comparison_to_detail(comparison)


@router.delete(
    "/comparisons/{comparison_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a comparison",
)
async def delete_comparison(
    db: DbSession,
    comparison_id: UUID,
) -> None:
    """Delete a comparison."""
    comparison = await _get_comparison_or_404(db, comparison_id)
    await db.delete(comparison)
    await db.commit()
    logger.info("Deleted comparison", comparison_id=str(comparison_id))


@router.get(
    "/projects/{project_id}/comparisons",
    response_model=ComparisonList,
    summary="List comparisons in a project",
)
async def list_project_comparisons(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
) -> ComparisonList:
    """List all comparisons in a project."""
    # Build query
    query = select(Comparison).where(Comparison.project_id == project_id)

    # Get total count
    count_query = select(func.count(Comparison.id)).where(Comparison.project_id == project_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(Comparison.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    comparisons = result.scalars().all()

    return ComparisonList(
        items=[_comparison_to_response(c) for c in comparisons],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get(
    "/evaluations/{evaluation_id}/comparisons",
    response_model=ComparisonList,
    summary="List comparisons involving an evaluation",
)
async def list_evaluation_comparisons(
    db: DbSession,
    evaluation_id: UUID,
    pagination: Pagination,
) -> ComparisonList:
    """List all comparisons where this evaluation is baseline or compared."""
    # For comparisons where this evaluation is in compared list,
    # we need to check the JSON array. Using a simple approach that
    # fetches all comparisons and filters in Python.
    all_comparisons_query = select(Comparison)
    all_result = await db.execute(all_comparisons_query)
    all_comparisons = all_result.scalars().all()

    # Filter to find ones that include this evaluation
    matching_comparisons: list[Comparison] = []
    for c in all_comparisons:
        if c.baseline_evaluation_id == evaluation_id:
            matching_comparisons.append(c)
        elif str(evaluation_id) in c.compared_evaluation_ids:
            matching_comparisons.append(c)

    # Remove duplicates (in case same comparison matched both conditions)
    seen_ids: set[UUID] = set()
    unique_comparisons: list[Comparison] = []
    for c in matching_comparisons:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            unique_comparisons.append(c)

    # Sort by created_at desc
    unique_comparisons.sort(key=lambda x: x.created_at, reverse=True)

    # Apply pagination manually
    total = len(unique_comparisons)
    start = pagination.offset
    end = start + pagination.limit
    paginated = unique_comparisons[start:end]

    return ComparisonList(
        items=[_comparison_to_response(c) for c in paginated],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )
