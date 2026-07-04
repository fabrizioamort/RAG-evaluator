"""Comparisons API endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination
from app.models.comparison import Comparison
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.schemas.comparison import (
    AggregateMetrics,
    ComparisonCreate,
    ComparisonDetail,
    ComparisonList,
    ComparisonResponse,
    PerQuestionDelta,
)
from app.services.artifact_store import get_artifact_store
from app.services.comparison_service import get_comparison_service
from app.services.evaluation_exporter import (
    HEADLINE_COLUMNS,
    ExportMember,
    build_markdown_report,
    build_question_record,
    headline_rows,
    per_question_jsonl,
    taxonomy_columns,
    taxonomy_rows,
    to_csv,
)
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


def _member_label(eval_model: Evaluation) -> str:
    """Match the frontend label: notes, then RAG config name, then short id."""
    if eval_model.notes and eval_model.notes.strip():
        return eval_model.notes.strip()
    if eval_model.rag_config and eval_model.rag_config.name:
        return eval_model.rag_config.name
    return f"#{str(eval_model.id)[:8]}"


def _rag_type_of(eval_model: Evaluation) -> str | None:
    """Frozen index snapshot wins over the (mutable) RAG config."""
    if eval_model.index and eval_model.index.config_snapshot:
        rag_type = eval_model.index.config_snapshot.get("rag_type")
        if rag_type:
            return str(rag_type)
    if eval_model.rag_config:
        return eval_model.rag_config.rag_type
    return None


def _manifest_dict(eval_model: Evaluation) -> dict | None:
    manifest = eval_model.run_manifest
    if manifest is None:
        return None
    return {
        "rag_config_snapshot": manifest.rag_config_snapshot,
        "build_config_snapshot": manifest.build_config_snapshot,
        "query_overrides": manifest.query_overrides,
        "effective_config_snapshot": manifest.effective_config_snapshot,
        "kb_version_snapshot": manifest.kb_version_snapshot,
        "generation_model": manifest.generation_model,
        "eval_judge_model": manifest.eval_judge_model,
        "prompt_templates": manifest.prompt_templates,
        "rag_evaluator_version": manifest.rag_evaluator_version,
        "platform_version": manifest.platform_version,
    }


def _to_export_member(eval_model: Evaluation) -> ExportMember:
    legal = (eval_model.summary_metrics or {}).get("legal_rag_bench")
    return ExportMember(
        label=_member_label(eval_model),
        rag_config_name=eval_model.rag_config.name if eval_model.rag_config else None,
        rag_type=_rag_type_of(eval_model),
        pass_rate=eval_model.pass_rate,
        summary_metrics=eval_model.summary_metrics,
        performance_metrics=eval_model.performance_metrics,
        legal_rag_bench=legal,
        manifest=_manifest_dict(eval_model),
    )


async def _load_member_evaluations(
    db: DbSession, comparison: Comparison
) -> list[Evaluation]:
    """Load member evaluations (baseline first, then compared in stored order)."""
    ordered_ids = [comparison.baseline_evaluation_id] + [
        UUID(id_str) for id_str in comparison.compared_evaluation_ids
    ]
    query = (
        select(Evaluation)
        .where(Evaluation.id.in_(ordered_ids))
        .options(
            selectinload(Evaluation.rag_config),
            selectinload(Evaluation.index),
            selectinload(Evaluation.run_manifest),
        )
    )
    result = await db.execute(query)
    by_id = {e.id: e for e in result.scalars().all()}
    return [by_id[eid] for eid in ordered_ids if eid in by_id]


async def _build_question_records(
    db: DbSession, evaluations: list[Evaluation]
) -> list[dict]:
    """Assemble per-question JSONL records, pulling legal payloads from artifacts."""
    store = get_artifact_store()
    records: list[dict] = []
    for eval_model in evaluations:
        member_label = _member_label(eval_model)
        rag_type = _rag_type_of(eval_model)
        rag_config_name = eval_model.rag_config.name if eval_model.rag_config else None

        query = (
            select(EvaluationResult)
            .where(EvaluationResult.evaluation_id == eval_model.id)
            .options(selectinload(EvaluationResult.test_case))
            .order_by(EvaluationResult.created_at.asc())
        )
        result = await db.execute(query)
        rows = result.scalars().all()
        artifact_ids = [r.raw_metrics_artifact_id for r in rows if r.raw_metrics_artifact_id]
        payloads = await store.retrieve_json_by_ids(db, artifact_ids)
        for r in rows:
            legal = None
            raw = payloads.get(r.raw_metrics_artifact_id) if r.raw_metrics_artifact_id else None
            if raw:
                legal = raw.get("legal_rag_bench")
            test_case = r.test_case
            records.append(
                build_question_record(
                    member_label=member_label,
                    evaluation_id=str(eval_model.id),
                    rag_type=rag_type,
                    rag_config_name=rag_config_name,
                    question=test_case.question if test_case else None,
                    expected_answer=test_case.expected_answer if test_case else None,
                    generated_answer=r.generated_answer,
                    scores={
                        "faithfulness": r.faithfulness_score,
                        "relevancy": r.relevancy_score,
                        "precision": r.precision_score,
                        "recall": r.recall_score,
                        "g_eval": r.g_eval_score,
                    },
                    latency_seconds=r.latency_seconds,
                    legal_rag_bench=legal,
                )
            )
    return records


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


@router.get(
    "/comparisons/{comparison_id}/export",
    summary="Export comparison as article-ready CSV / Markdown / JSONL",
)
async def export_comparison(
    db: DbSession,
    comparison_id: UUID,
    format: str = Query("markdown", pattern="^(markdown|csv|jsonl)$"),
    table: str = Query("headline", pattern="^(headline|taxonomy)$"),
) -> Response:
    """Export Legal RAG Bench comparison artifacts.

    - ``markdown``: full report (headline + taxonomy tables + run manifests).
    - ``csv``: a single table (``table=headline`` or ``table=taxonomy``).
    - ``jsonl``: one reproducibility record per question per evaluation.
    """
    comparison = await _get_comparison_or_404(db, comparison_id)
    evaluations = await _load_member_evaluations(db, comparison)
    members = [_to_export_member(e) for e in evaluations]

    stem = f"comparison-{str(comparison_id)[:8]}"

    if format == "markdown":
        title = comparison.name or "Legal RAG Bench - Architecture Comparison"
        content = build_markdown_report(members, title=title)
        return _file_response(content, "text/markdown", f"{stem}.md")

    if format == "csv":
        if table == "taxonomy":
            content = to_csv(taxonomy_rows(members), taxonomy_columns())
        else:
            content = to_csv(headline_rows(members), HEADLINE_COLUMNS)
        return _file_response(content, "text/csv", f"{stem}-{table}.csv")

    records = await _build_question_records(db, evaluations)
    content = per_question_jsonl(records)
    return _file_response(content, "application/x-ndjson", f"{stem}-questions.jsonl")


def _file_response(content: str, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
