"""Evaluations API endpoints."""

import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.api.deps import DbSession, Pagination
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_version import KnowledgeBaseVersion
from app.models.rag_config import RAGConfig
from app.models.run_manifest import RunManifest
from app.models.test_set import TestSet
from app.schemas.evaluation import (
    CostMetrics,
    EvaluationCreate,
    EvaluationList,
    EvaluationResponse,
    EvaluationResultList,
    EvaluationResultWithTestCase,
    PerformanceMetrics,
    SetBaselineRequest,
    SummaryMetrics,
)
from app.services.artifact_store import get_artifact_store
from app.services.evaluation_runner import EvaluationRunner, get_evaluation_runner
from app.services.job_event_log import get_job_event_log
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Evaluations"])
logger = get_logger(__name__)


async def _get_evaluation_or_404(db: DbSession, evaluation_id: UUID) -> Evaluation:
    """Get evaluation by ID or raise 404."""
    query = (
        select(Evaluation)
        .where(Evaluation.id == evaluation_id)
        .options(
            selectinload(Evaluation.rag_config),
            selectinload(Evaluation.knowledge_base),
            selectinload(Evaluation.test_set),
        )
    )
    result = await db.execute(query)
    evaluation = result.scalar_one_or_none()
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with id {evaluation_id} not found",
        )
    return evaluation


def _evaluation_to_response(eval_model: Evaluation, result_count: int = 0) -> EvaluationResponse:
    """Convert Evaluation model to EvaluationResponse schema."""
    return EvaluationResponse(
        id=eval_model.id,
        project_id=eval_model.project_id,
        knowledge_base_id=eval_model.knowledge_base_id,
        kb_version_id=eval_model.kb_version_id,
        test_set_id=eval_model.test_set_id,
        rag_config_id=eval_model.rag_config_id,
        run_manifest_id=eval_model.run_manifest_id,
        status=eval_model.status,
        started_at=eval_model.started_at,
        completed_at=eval_model.completed_at,
        summary_metrics=SummaryMetrics(**eval_model.summary_metrics)
        if eval_model.summary_metrics
        else None,
        cost_metrics=CostMetrics(**eval_model.cost_metrics) if eval_model.cost_metrics else None,
        performance_metrics=PerformanceMetrics(**eval_model.performance_metrics)
        if eval_model.performance_metrics
        else None,
        pass_rate=eval_model.pass_rate,
        is_baseline=eval_model.is_baseline,
        baseline_reason=eval_model.baseline_reason,
        notes=eval_model.notes,
        tags=eval_model.tags if isinstance(eval_model.tags, list) else [],
        error_message=eval_model.error_message,
        created_at=eval_model.created_at,
        result_count=result_count,
    )


async def _run_evaluation_background(evaluation_id: UUID) -> None:
    """Background task to run the evaluation."""
    from app.database import get_db_context

    async with get_db_context() as db:
        try:
            runner = get_evaluation_runner(db, evaluation_id)
            await runner.run()
        except Exception:
            logger.exception("Background evaluation failed", evaluation_id=str(evaluation_id))


@router.post(
    "/evaluations",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new evaluation",
)
async def create_evaluation(
    db: DbSession,
    background_tasks: BackgroundTasks,
    evaluation_data: EvaluationCreate,
) -> EvaluationResponse:
    """Create and start a new evaluation."""
    # 1. Get RAG Config and associated project
    query = select(RAGConfig).where(RAGConfig.id == evaluation_data.rag_config_id)
    result = await db.execute(query)
    rag_config: RAGConfig | None = result.scalar_one_or_none()
    if not rag_config:
        raise HTTPException(status_code=404, detail="RAG Config not found")

    project_id = rag_config.project_id

    # 2. Get KB and check if it belongs to project
    kb_query = select(KnowledgeBase).where(
        KnowledgeBase.id == evaluation_data.knowledge_base_id,
        KnowledgeBase.project_id == project_id,
    )
    kb_result = await db.execute(kb_query)
    kb: KnowledgeBase | None = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge Base not found in project")

    # 3. Get Test Set and check if it belongs to project
    ts_query = select(TestSet).where(
        TestSet.id == evaluation_data.test_set_id, TestSet.project_id == project_id
    )
    ts_result = await db.execute(ts_query)
    test_set: TestSet | None = ts_result.scalar_one_or_none()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test Set not found in project")

    # 4. Create Run Manifest (Snapshot)
    # Get current KB version
    kb_v_query = select(KnowledgeBaseVersion).where(
        KnowledgeBaseVersion.knowledge_base_id == kb.id,
        KnowledgeBaseVersion.version_number == kb.current_version,
    )
    kb_v_result = await db.execute(kb_v_query)
    kb_version: KnowledgeBaseVersion | None = kb_v_result.scalar_one_or_none()

    manifest = RunManifest(
        rag_config_snapshot={
            "id": str(rag_config.id),
            "name": rag_config.name,
            "rag_type": rag_config.rag_type,
            "parameters": rag_config.parameters,
            "llm_provider": rag_config.llm_provider,
            "llm_model": rag_config.llm_model,
        },
        kb_version_snapshot={
            "version_number": kb_version.version_number if kb_version else 0,
            "document_snapshot": kb_version.document_snapshot if kb_version else [],
        },
        generation_model=rag_config.llm_model,
        eval_judge_model=rag_config.llm_model,  # Using same model as judge for now
        rag_evaluator_version="0.1.0",
        platform_version="0.1.0",
    )
    db.add(manifest)
    await db.flush()

    # 5. Create Evaluation
    evaluation = Evaluation(
        project_id=project_id,
        knowledge_base_id=kb.id,
        kb_version_id=kb_version.id if kb_version else None,
        test_set_id=test_set.id,
        rag_config_id=rag_config.id,
        run_manifest_id=manifest.id,
        status="pending",
        notes=evaluation_data.notes,
        tags=evaluation_data.tags,
    )
    db.add(evaluation)
    await db.commit()
    await db.refresh(evaluation)

    # 6. Start background task
    background_tasks.add_task(_run_evaluation_background, evaluation.id)

    logger.info(
        "Started evaluation",
        evaluation_id=str(evaluation.id),
        project_id=str(project_id),
        rag_config=rag_config.name,
    )

    return _evaluation_to_response(evaluation)


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationResponse,
    summary="Get evaluation details",
)
async def get_evaluation(
    db: DbSession,
    evaluation_id: UUID,
) -> EvaluationResponse:
    """Get evaluation details."""
    evaluation = await _get_evaluation_or_404(db, evaluation_id)

    # Count results
    query = select(func.count(EvaluationResult.id)).where(
        EvaluationResult.evaluation_id == evaluation_id
    )
    result = await db.execute(query)
    result_count = result.scalar() or 0

    return _evaluation_to_response(evaluation, result_count=result_count)


@router.get(
    "/evaluations/{evaluation_id}/results",
    response_model=EvaluationResultList,
    summary="Get evaluation results",
)
async def get_evaluation_results(
    db: DbSession,
    evaluation_id: UUID,
    pagination: Pagination,
) -> EvaluationResultList:
    """Get paginated results for an evaluation."""
    # Verify evaluation exists
    await _get_evaluation_or_404(db, evaluation_id)

    # Build query
    query = (
        select(EvaluationResult)
        .where(EvaluationResult.evaluation_id == evaluation_id)
        .options(selectinload(EvaluationResult.test_case))
    )

    # Get total count
    count_query = select(func.count(EvaluationResult.id)).where(
        EvaluationResult.evaluation_id == evaluation_id
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(EvaluationResult.created_at.asc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    results = result.scalars().all()

    items = []
    for r in results:
        # Map to schema - EvaluationResultWithTestCase if test case is loaded
        items.append(
            EvaluationResultWithTestCase(
                **r.__dict__,
                question=r.test_case.question if r.test_case else None,
                expected_answer=r.test_case.expected_answer if r.test_case else None,
                difficulty=r.test_case.difficulty if r.test_case else None,
                category=r.test_case.category if r.test_case else None,
            )
        )

    return EvaluationResultList(
        items=items,
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get(
    "/evaluations/{evaluation_id}/trace/{result_id}",
    summary="Get retrieval trace for a result",
)
async def get_evaluation_result_trace(
    db: DbSession,
    evaluation_id: UUID,
    result_id: UUID,
) -> Any:
    """Get the retrieval trace artifact for a specific result."""
    # 1. Get the result
    query = select(EvaluationResult).where(
        EvaluationResult.id == result_id, EvaluationResult.evaluation_id == evaluation_id
    )
    result_set = await db.execute(query)
    eval_result = result_set.scalar_one_or_none()

    if not eval_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found in evaluation {evaluation_id}",
        )

    if not eval_result.retrieval_trace_artifact_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No retrieval trace artifact associated with result {result_id}",
        )

    # 2. Get the artifact content
    store = get_artifact_store()
    trace = await store.retrieve_json_by_id(db, eval_result.retrieval_trace_artifact_id)

    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval trace artifact content not found on disk",
        )

    return trace


@router.get(
    "/evaluations/{evaluation_id}/stream",
    summary="SSE stream for evaluation progress",
)
async def stream_evaluation_progress(
    evaluation_id: UUID,
    last_event_id: str | None = Query(default=None, alias="last-event-id"),
) -> EventSourceResponse:
    """Stream evaluation progress events via SSE."""
    event_log = get_job_event_log()

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        async for event in event_log.subscribe(evaluation_id, last_event_id):
            yield {
                "event": event.get("event_type", "message"),
                "id": str(datetime.now(timezone.utc).timestamp()),
                "data": json.dumps(event),
            }

    return EventSourceResponse(event_generator())


@router.post(
    "/evaluations/{evaluation_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel running evaluation",
)
async def cancel_evaluation(
    db: DbSession,
    evaluation_id: UUID,
) -> None:
    """Signal evaluation to cancel."""
    await _get_evaluation_or_404(db, evaluation_id)
    EvaluationRunner.cancel(evaluation_id)
    logger.info("Cancelled evaluation", evaluation_id=str(evaluation_id))


@router.post(
    "/evaluations/{evaluation_id}/pause",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Pause running evaluation",
)
async def pause_evaluation(
    db: DbSession,
    evaluation_id: UUID,
) -> None:
    """Signal evaluation to pause."""
    await _get_evaluation_or_404(db, evaluation_id)
    EvaluationRunner.pause(evaluation_id)
    logger.info("Paused evaluation", evaluation_id=str(evaluation_id))


@router.post(
    "/evaluations/{evaluation_id}/resume",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Resume paused evaluation",
)
async def resume_evaluation(
    db: DbSession,
    background_tasks: BackgroundTasks,
    evaluation_id: UUID,
) -> None:
    """Resume a paused evaluation."""
    evaluation = await _get_evaluation_or_404(db, evaluation_id)

    if evaluation.status != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume evaluation in status {evaluation.status}",
        )

    EvaluationRunner.resume(evaluation_id)
    background_tasks.add_task(_run_evaluation_background, evaluation.id)
    logger.info("Resumed evaluation", evaluation_id=str(evaluation_id))


@router.post(
    "/evaluations/{evaluation_id}/retry",
    response_model=EvaluationResponse,
    summary="Retry failed evaluation",
)
async def retry_evaluation(
    db: DbSession,
    background_tasks: BackgroundTasks,
    evaluation_id: UUID,
) -> EvaluationResponse:
    """Retry a failed or cancelled evaluation."""
    evaluation = await _get_evaluation_or_404(db, evaluation_id)

    if evaluation.status not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry evaluation in status {evaluation.status}",
        )

    # Reset evaluation status
    evaluation.status = "pending"
    evaluation.error_message = None
    evaluation.started_at = None
    evaluation.completed_at = None

    # Delete results to start fresh if needed, or runner can handle resume
    # For retry, we usually want to start fresh or resume from last successful
    # EvaluationRunner currently handles resume from job.progress_current

    await db.commit()
    await db.refresh(evaluation)

    background_tasks.add_task(_run_evaluation_background, evaluation.id)
    logger.info("Retrying evaluation", evaluation_id=str(evaluation_id))

    return _evaluation_to_response(evaluation)


@router.post(
    "/evaluations/{evaluation_id}/set-baseline",
    response_model=EvaluationResponse,
    summary="Mark evaluation as baseline",
)
async def set_baseline(
    db: DbSession,
    evaluation_id: UUID,
    request: SetBaselineRequest,
) -> EvaluationResponse:
    """Mark an evaluation as the baseline for its project."""
    evaluation = await _get_evaluation_or_404(db, evaluation_id)

    if evaluation.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed evaluations can be marked as baseline",
        )

    # Unset existing baseline for this project
    from sqlalchemy import update

    await db.execute(
        update(Evaluation)
        .where(Evaluation.project_id == evaluation.project_id, Evaluation.is_baseline)
        .values(is_baseline=False, baseline_reason=None)
    )

    # Set new baseline
    evaluation.is_baseline = True
    evaluation.baseline_reason = request.reason

    await db.commit()
    await db.refresh(evaluation)

    logger.info("Set baseline evaluation", evaluation_id=str(evaluation_id))

    return _evaluation_to_response(evaluation)


@router.get(
    "/projects/{project_id}/evaluations",
    response_model=EvaluationList,
    summary="List evaluations in a project",
)
async def list_evaluations(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
    status: str | None = Query(default=None),
) -> EvaluationList:
    """List all evaluations in a project."""
    # Build query
    query = select(Evaluation).where(Evaluation.project_id == project_id)

    if status:
        query = query.where(Evaluation.status == status)

    # Get total count
    count_query = select(func.count(Evaluation.id)).where(Evaluation.project_id == project_id)
    if status:
        count_query = count_query.where(Evaluation.status == status)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(Evaluation.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    evaluations = result.scalars().all()

    return EvaluationList(
        items=[_evaluation_to_response(e) for e in evaluations],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )
