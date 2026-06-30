"""Tests for JobCheckpointService."""

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.models.project import Project
from app.services.job_checkpoint_service import JobCheckpointService


@pytest.fixture
async def sample_evaluation(db_session: AsyncSession) -> Evaluation:
    """Fixture for a sample evaluation."""
    project = Project(name="Test Project")
    db_session.add(project)
    await db_session.flush()

    evaluation = Evaluation(project_id=project.id, status="pending")
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)
    return evaluation


@pytest.mark.asyncio
async def test_create_and_get_job(db_session: AsyncSession, sample_evaluation: Evaluation) -> None:
    """Test creating and retrieving a job."""
    service = JobCheckpointService(db_session)

    # Create
    job = await service.create_job(sample_evaluation.id, 100)
    assert job.evaluation_id == sample_evaluation.id
    assert job.progress_total == 100
    assert job.state == "created"

    # Get
    retrieved = await service.get_job(sample_evaluation.id)
    assert retrieved is not None
    assert retrieved.id == job.id


@pytest.mark.asyncio
async def test_update_progress(db_session: AsyncSession, sample_evaluation: Evaluation) -> None:
    """Test updating job progress."""
    service = JobCheckpointService(db_session)
    await service.create_job(sample_evaluation.id, 100)

    await service.update_progress(sample_evaluation.id, 10, state="running")

    job = await service.get_job(sample_evaluation.id)
    assert job is not None
    assert job.progress_current == 10
    assert job.state == "running"


@pytest.mark.asyncio
async def test_save_checkpoint(db_session: AsyncSession, sample_evaluation: Evaluation) -> None:
    """Test saving a checkpoint."""
    service = JobCheckpointService(db_session)
    await service.create_job(sample_evaluation.id, 100)

    checkpoint_data = {"last_seen_id": "item_123"}
    await service.save_checkpoint(sample_evaluation.id, 50, checkpoint_data)

    job = await service.get_job(sample_evaluation.id)
    assert job is not None
    assert job.last_checkpoint == 50
    assert job.checkpoint_data == checkpoint_data


@pytest.mark.asyncio
async def test_complete_job(db_session: AsyncSession, sample_evaluation: Evaluation) -> None:
    """Test completing a job."""
    service = JobCheckpointService(db_session)
    await service.create_job(sample_evaluation.id, 100)

    metrics: dict[str, Any] = {"faithfulness_avg": 0.8}
    await service.complete_job(sample_evaluation.id, metrics, 0.75)

    # Check evaluation
    await db_session.refresh(sample_evaluation)
    assert sample_evaluation.status == "completed"
    assert sample_evaluation.summary_metrics == metrics
    assert sample_evaluation.pass_rate == 0.75

    # Check job
    job = await service.get_job(sample_evaluation.id)
    assert job is not None
    assert job.state == "completed"


@pytest.mark.asyncio
async def test_fail_job(db_session: AsyncSession, sample_evaluation: Evaluation) -> None:
    """Test failing a job."""
    service = JobCheckpointService(db_session)
    await service.create_job(sample_evaluation.id, 100)

    await service.fail_job(sample_evaluation.id, "Connection timeout")

    # Check evaluation
    await db_session.refresh(sample_evaluation)
    assert sample_evaluation.status == "failed"
    assert sample_evaluation.error_message == "Connection timeout"

    # Check job
    job = await service.get_job(sample_evaluation.id)
    assert job is not None
    assert job.state == "failed"
    assert job.error_message == "Connection timeout"


@pytest.mark.asyncio
async def test_reconcile_orphaned_evaluations(db_session: AsyncSession) -> None:
    """Orphaned running/pending evaluations become failed-but-recoverable."""
    project = Project(name="Test Project")
    db_session.add(project)
    await db_session.flush()

    running = Evaluation(project_id=project.id, status="running")
    pending = Evaluation(project_id=project.id, status="pending")
    paused = Evaluation(project_id=project.id, status="paused")
    completed = Evaluation(project_id=project.id, status="completed")
    db_session.add_all([running, pending, paused, completed])
    await db_session.flush()

    service = JobCheckpointService(db_session)
    await service.create_job(running.id, 3)
    await service.update_progress(running.id, 2, state="running")
    # Two test cases already have results saved for the running evaluation.
    db_session.add_all(
        [
            EvaluationResult(evaluation_id=running.id),
            EvaluationResult(evaluation_id=running.id),
        ]
    )
    await db_session.commit()

    reconciled = await service.reconcile_orphaned_evaluations()
    assert reconciled == 2

    for evaluation in (running, pending, paused, completed):
        await db_session.refresh(evaluation)

    # Orphaned active evaluations are now retryable.
    assert running.status == "failed"
    assert running.error_message is not None
    assert running.completed_at is not None
    assert pending.status == "failed"
    # Deliberate / terminal states are untouched.
    assert paused.status == "paused"
    assert completed.status == "completed"

    # The linked job is marked failed too.
    job = await service.get_job(running.id)
    assert job is not None
    assert job.state == "failed"

    # Partial results are preserved so retry can resume from where it stopped.
    results = await db_session.execute(
        EvaluationResult.__table__.select().where(
            EvaluationResult.evaluation_id == running.id
        )
    )
    assert len(results.all()) == 2
