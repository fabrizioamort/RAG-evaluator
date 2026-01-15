"""Test Sets API endpoints."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.test_case import TestCase
from app.models.test_generation_job import TestGenerationJob
from app.models.test_set import TestSet
from app.schemas.test_set import (
    TestCaseBulkCreate,
    TestCaseBulkReview,
    TestCaseCreate,
    TestCaseResponse,
    TestCaseUpdate,
    TestGenerationConfig,
    TestGenerationJobResponse,
    TestGenerationStatusResponse,
    TestSetCreate,
    TestSetExport,
    TestSetImport,
    TestSetList,
    TestSetResponse,
    TestSetUpdate,
    TestSetWithCases,
)
from app.services.test_generator_service import (
    GenerationConfig,
    get_test_generator_service,
)
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Test Sets"])
logger = get_logger(__name__)


def _test_set_to_response(test_set: TestSet) -> TestSetResponse:
    """Convert TestSet model to TestSetResponse schema."""
    return TestSetResponse(
        id=test_set.id,
        project_id=test_set.project_id,
        name=test_set.name,
        description=test_set.description,
        tags=test_set.tags if isinstance(test_set.tags, list) else [],
        test_case_count=test_set.test_case_count,
        created_at=test_set.created_at,
    )


def _test_case_to_response(test_case: TestCase) -> TestCaseResponse:
    """Convert TestCase model to TestCaseResponse schema."""
    return TestCaseResponse(
        id=test_case.id,
        test_set_id=test_case.test_set_id,
        template_id=test_case.template_id,
        question=test_case.question,
        expected_answer=test_case.expected_answer,
        ground_truth_context=test_case.ground_truth_context
        if isinstance(test_case.ground_truth_context, list)
        else [],
        difficulty=test_case.difficulty,
        category=test_case.category,
        question_type=test_case.question_type,
        is_generated=test_case.is_generated,
        is_reviewed=test_case.is_reviewed,
        quality_score=test_case.quality_score,
        provenance_artifact_id=test_case.provenance_artifact_id,
        created_at=test_case.created_at,
    )


async def _get_project_or_404(db: DbSession, project_id: UUID) -> Project:
    """Get project by ID or raise 404."""
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )
    return project


async def _get_test_set_or_404(
    db: DbSession, test_set_id: UUID, include_cases: bool = False
) -> TestSet:
    """Get test set by ID or raise 404."""
    query = select(TestSet).where(TestSet.id == test_set_id)
    if include_cases:
        query = query.options(selectinload(TestSet.test_cases))

    result = await db.execute(query)
    test_set = result.scalar_one_or_none()
    if not test_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test set with id {test_set_id} not found",
        )
    return test_set


# =============================================================================
# Test Set CRUD Endpoints
# =============================================================================


@router.get(
    "/projects/{project_id}/test-sets",
    response_model=TestSetList,
    summary="List test sets in a project",
    description="Retrieve a paginated list of test sets for a specific project.",
)
async def list_test_sets(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
) -> TestSetList:
    """List all test sets in a project."""
    await _get_project_or_404(db, project_id)

    query = (
        select(TestSet)
        .where(TestSet.project_id == project_id)
        .options(selectinload(TestSet.test_cases))
    )

    count_query = select(func.count()).select_from(TestSet).where(TestSet.project_id == project_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(TestSet.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    result = await db.execute(query)
    test_sets = result.scalars().all()

    logger.info("Listed test sets", project_id=str(project_id), count=len(test_sets), total=total)

    return TestSetList(
        items=[_test_set_to_response(ts) for ts in test_sets],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "/projects/{project_id}/test-sets",
    response_model=TestSetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a test set",
    description="Create a new test set within a project.",
)
async def create_test_set(
    db: DbSession,
    project_id: UUID,
    test_set_data: TestSetCreate,
) -> TestSetResponse:
    """Create a new test set."""
    await _get_project_or_404(db, project_id)

    test_set = TestSet(
        project_id=project_id,
        name=test_set_data.name,
        description=test_set_data.description,
        tags=test_set_data.tags,
    )
    db.add(test_set)
    await db.commit()
    # Re-fetch with test_cases to avoid lazy loading
    test_set = await _get_test_set_or_404(db, test_set.id, include_cases=True)
    return _test_set_to_response(test_set)


@router.get(
    "/test-sets/{test_set_id}",
    response_model=TestSetWithCases,
    summary="Get test set details",
    description="Retrieve details of a test set including its test cases.",
)
async def get_test_set(
    db: DbSession,
    test_set_id: UUID,
) -> TestSetWithCases:
    """Get a test set by ID with test cases."""
    test_set = await _get_test_set_or_404(db, test_set_id, include_cases=True)

    return TestSetWithCases(
        id=test_set.id,
        project_id=test_set.project_id,
        name=test_set.name,
        description=test_set.description,
        tags=test_set.tags if isinstance(test_set.tags, list) else [],
        test_case_count=test_set.test_case_count,
        created_at=test_set.created_at,
        test_cases=[_test_case_to_response(tc) for tc in test_set.test_cases],
    )


@router.put(
    "/test-sets/{test_set_id}",
    response_model=TestSetResponse,
    summary="Update a test set",
    description="Update test set metadata (name, description, tags).",
)
async def update_test_set(
    db: DbSession,
    test_set_id: UUID,
    test_set_data: TestSetUpdate,
) -> TestSetResponse:
    """Update a test set."""
    test_set = await _get_test_set_or_404(db, test_set_id)

    update_data = test_set_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(test_set, field, value)

    await db.commit()
    # Re-fetch with test_cases to avoid lazy loading
    test_set = await _get_test_set_or_404(db, test_set_id, include_cases=True)
    return _test_set_to_response(test_set)


@router.delete(
    "/test-sets/{test_set_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a test set",
    description="Delete a test set and all its test cases.",
)
async def delete_test_set(
    db: DbSession,
    test_set_id: UUID,
) -> None:
    """Delete a test set and all its cases."""
    test_set = await _get_test_set_or_404(db, test_set_id)

    await db.delete(test_set)
    await db.commit()

    logger.info("Deleted test set", test_set_id=str(test_set_id))


# =============================================================================
# Test Case Management Endpoints
# =============================================================================


@router.post(
    "/test-sets/{test_set_id}/cases",
    response_model=TestCaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a test case",
    description="Add a single test case to a test set.",
)
async def add_test_case(
    db: DbSession,
    test_set_id: UUID,
    test_case_data: TestCaseCreate,
) -> TestCaseResponse:
    """Add a test case to a test set."""
    await _get_test_set_or_404(db, test_set_id)

    test_case = TestCase(
        test_set_id=test_set_id,
        template_id=test_case_data.template_id,
        question=test_case_data.question,
        expected_answer=test_case_data.expected_answer,
        ground_truth_context=test_case_data.ground_truth_context,
        difficulty=test_case_data.difficulty,
        category=test_case_data.category,
        question_type=test_case_data.question_type,
        is_generated=False,
        is_reviewed=True,
    )
    db.add(test_case)
    await db.commit()
    await db.refresh(test_case)

    logger.info("Added test case", test_case_id=str(test_case.id), test_set_id=str(test_set_id))

    return _test_case_to_response(test_case)


@router.post(
    "/test-sets/{test_set_id}/cases/bulk",
    response_model=list[TestCaseResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Bulk add test cases",
    description="Add multiple test cases to a test set in a single request.",
)
async def bulk_add_test_cases(
    db: DbSession,
    test_set_id: UUID,
    bulk_data: TestCaseBulkCreate,
) -> list[TestCaseResponse]:
    """Add multiple test cases to a test set."""
    await _get_test_set_or_404(db, test_set_id)

    test_cases = []
    for tc_data in bulk_data.test_cases:
        test_case = TestCase(
            test_set_id=test_set_id,
            template_id=tc_data.template_id,
            question=tc_data.question,
            expected_answer=tc_data.expected_answer,
            ground_truth_context=tc_data.ground_truth_context,
            difficulty=tc_data.difficulty,
            category=tc_data.category,
            question_type=tc_data.question_type,
            is_generated=False,
            is_reviewed=True,
        )
        db.add(test_case)
        test_cases.append(test_case)

    await db.commit()
    for tc in test_cases:
        await db.refresh(tc)

    logger.info("Bulk added test cases", count=len(test_cases), test_set_id=str(test_set_id))

    return [_test_case_to_response(tc) for tc in test_cases]


@router.put(
    "/test-sets/{test_set_id}/cases/{case_id}",
    response_model=TestCaseResponse,
    summary="Update a test case",
    description="Update an existing test case's details.",
)
async def update_test_case(
    db: DbSession,
    test_set_id: UUID,
    case_id: UUID,
    test_case_data: TestCaseUpdate,
) -> TestCaseResponse:
    """Update a test case."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    # Find the test case
    query = select(TestCase).where(TestCase.id == case_id, TestCase.test_set_id == test_set_id)
    result = await db.execute(query)
    test_case = result.scalar_one_or_none()

    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case with id {case_id} not found in test set {test_set_id}",
        )

    # Update only provided fields
    update_data = test_case_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(test_case, field, value)

    await db.commit()
    await db.refresh(test_case)

    logger.info("Updated test case", case_id=str(case_id), updated_fields=list(update_data.keys()))

    return _test_case_to_response(test_case)


@router.delete(
    "/test-sets/{test_set_id}/cases/{case_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a test case",
    description="Remove a test case from a test set.",
)
async def delete_test_case(
    db: DbSession,
    test_set_id: UUID,
    case_id: UUID,
) -> None:
    """Delete a test case from a test set."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    # Find the test case
    query = select(TestCase).where(TestCase.id == case_id, TestCase.test_set_id == test_set_id)
    result = await db.execute(query)
    test_case = result.scalar_one_or_none()

    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case with id {case_id} not found in test set {test_set_id}",
        )

    await db.delete(test_case)
    await db.commit()

    logger.info("Deleted test case", case_id=str(case_id), test_set_id=str(test_set_id))


@router.post(
    "/test-sets/{test_set_id}/cases/bulk-review",
    status_code=status.HTTP_200_OK,
    summary="Bulk review test cases",
    description="Bulk approve or reject (delete) generated test cases.",
)
async def bulk_review_test_cases(
    db: DbSession,
    test_set_id: UUID,
    review_data: TestCaseBulkReview,
) -> dict[str, str | int]:
    """Bulk approve or reject test cases."""
    await _get_test_set_or_404(db, test_set_id)

    if review_data.action == "approve":
        # Mark as reviewed
        query = (
            select(TestCase)
            .where(TestCase.id.in_(review_data.test_case_ids))
            .where(TestCase.test_set_id == test_set_id)
        )
        result = await db.execute(query)
        test_cases = result.scalars().all()

        for tc in test_cases:
            tc.is_reviewed = True

        await db.commit()
        return {"status": "success", "action": "approved", "count": len(test_cases)}

    elif review_data.action == "reject":
        # Delete rejected cases
        query = (
            select(TestCase)
            .where(TestCase.id.in_(review_data.test_case_ids))
            .where(TestCase.test_set_id == test_set_id)
        )
        result = await db.execute(query)
        test_cases = result.scalars().all()

        count = len(test_cases)
        for tc in test_cases:
            await db.delete(tc)

        await db.commit()
        return {"status": "success", "action": "rejected", "count": count}

    return {"status": "error", "message": "Invalid action"}


# =============================================================================
# Import/Export Endpoints
# =============================================================================


@router.post(
    "/projects/{project_id}/test-sets/import",
    response_model=TestSetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import test set",
    description="Create a new test set by importing from JSON.",
)
async def import_test_set(
    db: DbSession,
    project_id: UUID,
    import_data: TestSetImport,
) -> TestSetResponse:
    """Import a test set from JSON."""
    await _get_project_or_404(db, project_id)

    # Create the test set
    test_set = TestSet(
        project_id=project_id,
        name=import_data.name,
        description=import_data.description,
        tags=import_data.tags,
    )
    db.add(test_set)
    await db.flush()

    # Add the test cases
    for tc_data in import_data.test_cases:
        test_case = TestCase(
            test_set_id=test_set.id,
            template_id=tc_data.template_id,
            question=tc_data.question,
            expected_answer=tc_data.expected_answer,
            ground_truth_context=tc_data.ground_truth_context,
            difficulty=tc_data.difficulty,
            category=tc_data.category,
            question_type=tc_data.question_type,
            is_generated=False,
            is_reviewed=True,
        )
        db.add(test_case)

    await db.commit()
    # Re-fetch with test_cases to avoid lazy loading
    test_set = await _get_test_set_or_404(db, test_set.id, include_cases=True)
    return _test_set_to_response(test_set)


@router.get(
    "/test-sets/{test_set_id}/export",
    response_model=TestSetExport,
    summary="Export test set",
    description="Export a test set and all its cases to JSON.",
)
async def export_test_set(
    db: DbSession,
    test_set_id: UUID,
) -> TestSetExport:
    """Export a test set to JSON."""
    test_set = await _get_test_set_or_404(db, test_set_id, include_cases=True)

    test_cases_data = []
    for tc in test_set.test_cases:
        test_cases_data.append(
            {
                "question": tc.question,
                "expected_answer": tc.expected_answer,
                "ground_truth_context": tc.ground_truth_context,
                "difficulty": tc.difficulty,
                "category": tc.category,
                "question_type": tc.question_type,
                "is_generated": tc.is_generated,
                "is_reviewed": tc.is_reviewed,
                "quality_score": tc.quality_score,
            }
        )

    return TestSetExport(
        id=test_set.id,
        name=test_set.name,
        description=test_set.description,
        tags=test_set.tags if isinstance(test_set.tags, list) else [],
        created_at=test_set.created_at,
        test_cases=test_cases_data,
        metadata={
            "project_id": str(test_set.project_id),
            "export_version": "1.0",
        },
    )


# =============================================================================
# Test Generation Endpoints
# =============================================================================


def _generation_job_to_response(job: TestGenerationJob) -> TestGenerationJobResponse:
    """Convert TestGenerationJob model to response schema."""
    return TestGenerationJobResponse(
        id=job.id,
        test_set_id=job.test_set_id,
        knowledge_base_id=job.knowledge_base_id,
        status=job.status,
        config=job.config if isinstance(job.config, dict) else {},
        questions_generated=job.questions_generated,
        questions_total=job.questions_total,
        questions_rejected=job.questions_rejected,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        created_at=job.created_at,
    )


async def _run_generation_task(
    job_id: UUID,
    config: GenerationConfig,
) -> None:
    """Background task to run test generation.

    Uses the application's database session context.
    """
    from app.database import get_db_context

    async with get_db_context() as session:
        try:
            service = get_test_generator_service(db=session)
            await service.generate_and_save(job_id, config)
        except Exception as e:
            logger.error("Generation task failed", job_id=str(job_id), error=str(e))
            # Update job status to failed
            query = select(TestGenerationJob).where(TestGenerationJob.id == job_id)
            result = await session.execute(query)
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                await session.commit()


@router.post(
    "/test-sets/{test_set_id}/generate",
    response_model=TestGenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start test generation",
    description="Start generating test cases from a knowledge base using LLM.",
)
async def start_generation(
    db: DbSession,
    test_set_id: UUID,
    generation_config: TestGenerationConfig,
    background_tasks: BackgroundTasks,
) -> TestGenerationJobResponse:
    """Start test case generation for a test set."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    # Verify knowledge base exists
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == generation_config.knowledge_base_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base {generation_config.knowledge_base_id} not found",
        )

    # Check if there's already a running generation job
    running_query = (
        select(TestGenerationJob)
        .where(TestGenerationJob.test_set_id == test_set_id)
        .where(TestGenerationJob.status.in_(["pending", "running"]))
    )
    running_result = await db.execute(running_query)
    existing_job = running_result.scalar_one_or_none()
    if existing_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A generation job is already running for this test set",
        )

    # Create generation job
    job = TestGenerationJob(
        test_set_id=test_set_id,
        knowledge_base_id=generation_config.knowledge_base_id,
        status="pending",
        config=generation_config.model_dump(mode="json"),
        questions_total=generation_config.target_count,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Convert to service config
    service_config = GenerationConfig(
        target_count=generation_config.target_count,
        questions_per_chunk=generation_config.questions_per_chunk,
        difficulty_distribution=generation_config.difficulty_distribution,
        template_ids=generation_config.template_ids,
        llm_model=generation_config.llm_model,
        skip_semantic_check=generation_config.skip_semantic_check,
    )

    # Get database URL for background task

    # Schedule background generation
    background_tasks.add_task(
        _run_generation_task,
        job.id,
        service_config,
    )

    logger.info(
        "Started generation job",
        job_id=str(job.id),
        test_set_id=str(test_set_id),
        target_count=generation_config.target_count,
    )

    return _generation_job_to_response(job)


@router.get(
    "/test-sets/{test_set_id}/generation-status",
    response_model=TestGenerationStatusResponse,
    summary="Get generation status",
    description="Get the status of the current or most recent generation job.",
)
async def get_generation_status(
    db: DbSession,
    test_set_id: UUID,
) -> TestGenerationStatusResponse:
    """Get the status of test generation for a test set."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    # Get the most recent generation job
    query = (
        select(TestGenerationJob)
        .where(TestGenerationJob.test_set_id == test_set_id)
        .order_by(TestGenerationJob.created_at.desc())
        .limit(1)
    )
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No generation job found for this test set",
        )

    # Calculate progress
    progress = 0.0
    if job.questions_total > 0:
        progress = min(1.0, job.questions_generated / job.questions_total)
    if job.status == "completed":
        progress = 1.0

    return TestGenerationStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=progress,
        questions_generated=job.questions_generated,
        questions_total=job.questions_total,
        questions_rejected=job.questions_rejected,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.delete(
    "/test-sets/{test_set_id}/generation",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel generation",
    description="Cancel an ongoing test generation job.",
)
async def cancel_generation(
    db: DbSession,
    test_set_id: UUID,
) -> None:
    """Cancel an ongoing generation job."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    # Find running job
    query = (
        select(TestGenerationJob)
        .where(TestGenerationJob.test_set_id == test_set_id)
        .where(TestGenerationJob.status.in_(["pending", "running"]))
    )
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active generation job to cancel",
        )

    # Mark as cancelled
    job.status = "cancelled"
    await db.commit()

    logger.info("Cancelled generation job", job_id=str(job.id), test_set_id=str(test_set_id))


@router.get(
    "/test-sets/{test_set_id}/generation-jobs",
    response_model=list[TestGenerationJobResponse],
    summary="List generation jobs",
    description="List all generation jobs for a test set.",
)
async def list_generation_jobs(
    db: DbSession,
    test_set_id: UUID,
) -> list[TestGenerationJobResponse]:
    """List all generation jobs for a test set."""
    # Verify test set exists
    await _get_test_set_or_404(db, test_set_id)

    query = (
        select(TestGenerationJob)
        .where(TestGenerationJob.test_set_id == test_set_id)
        .order_by(TestGenerationJob.created_at.desc())
    )
    result = await db.execute(query)
    jobs = result.scalars().all()

    return [_generation_job_to_response(job) for job in jobs]
