"""Integration tests for the evaluation lifecycle."""

import uuid
from contextlib import ExitStack
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import LogCaptureFixture
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import Artifact
from app.models.evaluation import Evaluation
from app.models.evaluation_job import EvaluationJob
from app.models.evaluation_result import EvaluationResult
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.models.test_case import TestCase
from app.models.test_set import TestSet
from app.services.evaluation_runner import EvaluationRunner


async def create_setup_data(
    db_session: AsyncSession, case_count: int = 3
) -> tuple[Evaluation, list[uuid.UUID]]:
    """Helper to create test data including dummy artifacts for FK constraints."""
    project = Project(name="Lifecycle Test Project")
    db_session.add(project)
    await db_session.flush()

    rag_config = RAGConfig(
        project_id=project.id,
        name="Lifecycle RAG",
        rag_type="vector_semantic",
        llm_model="gpt-4o-mini",
        parameters={},
    )
    db_session.add(rag_config)

    test_set = TestSet(project_id=project.id, name="Lifecycle Test Set")
    db_session.add(test_set)
    await db_session.flush()

    for i in range(case_count):
        test_case = TestCase(
            test_set_id=test_set.id,
            question=f"Question {i}?",
            expected_answer=f"Answer {i}",
            ground_truth_context=[f"Context {i}"],
        )
        db_session.add(test_case)

    evaluation = Evaluation(
        project_id=project.id,
        test_set_id=test_set.id,
        rag_config_id=rag_config.id,
        status="pending",
    )
    db_session.add(evaluation)

    # Create dummy artifacts for EvaluationResult FKs
    artifact_ids = []
    for kind in ["context", "trace", "metrics"]:
        art = Artifact(
            kind=kind,
            storage_key=str(uuid.uuid4()),
            content_type="application/json",
            size_bytes=100,
        )
        db_session.add(art)
        await db_session.flush()
        artifact_ids.append(art.id)

    await db_session.commit()
    await db_session.refresh(evaluation)
    return evaluation, artifact_ids


def get_mock_patches() -> list[Any]:
    """Returns list of patches for common runner dependencies."""
    return [
        patch("app.services.evaluation_runner.get_rag_adapter_service"),
        patch("app.services.evaluation_runner.FaithfulnessMetric"),
        patch("app.services.evaluation_runner.AnswerRelevancyMetric"),
        patch("app.services.evaluation_runner.ContextualPrecisionMetric"),
        patch("app.services.evaluation_runner.ContextualRecallMetric"),
        patch("app.services.evaluation_runner.get_job_event_log"),
        patch("app.services.cost_tracker.get_cost_tracker"),
        patch("app.services.evaluation_runner.get_artifact_store"),
        patch("app.services.evaluation_runner.get_checkpoint_service"),
    ]


def setup_common_mocks(
    stack: ExitStack, patches: list[Any], mock_rag: MagicMock, artifact_ids: list[uuid.UUID]
) -> tuple[MagicMock, MagicMock]:
    """Configures common mocks for tests."""
    mock_get_rag = stack.enter_context(patches[0])
    mock_faith = stack.enter_context(patches[1])
    mock_rel = stack.enter_context(patches[2])
    mock_prec = stack.enter_context(patches[3])
    mock_recall = stack.enter_context(patches[4])
    mock_get_event_log = stack.enter_context(patches[5])
    mock_get_cost_tracker = stack.enter_context(patches[6])
    mock_get_artifact_store = stack.enter_context(patches[7])
    mock_get_checkpoint_service = stack.enter_context(patches[8])

    # RAG Adapter
    mock_adapter = mock_get_rag.return_value
    mock_adapter.get_or_create_rag.return_value = mock_rag
    # Default query_with_trace is the one from mock_rag
    mock_adapter.query_with_trace = mock_rag.query_with_trace

    # Metrics
    for m_class, name in [
        (mock_faith, "FaithfulnessMetric"),
        (mock_rel, "AnswerRelevancyMetric"),
        (mock_prec, "ContextualPrecisionMetric"),
        (mock_recall, "ContextualRecallMetric"),
    ]:
        inst = MagicMock()
        inst.__class__.__name__ = name
        inst.score = 0.8
        inst.measure = MagicMock()
        inst.reason = "Mock Reason"  # Explicit string
        m_class.return_value = inst

    # Event Log
    mock_event_log = mock_get_event_log.return_value
    mock_event_log.log_event = AsyncMock()

    # Cost Tracker
    mock_cost_tracker = mock_get_cost_tracker.return_value
    mock_cost_tracker.calculate_cost.return_value = Decimal("0.001")

    # Artifact Store
    mock_artifact_store = mock_get_artifact_store.return_value
    mock_artifact_store.store_json = AsyncMock()
    # Cycle through our real artifact IDs
    mock_artifact_store.store_json.side_effect = [
        MagicMock(id=artifact_ids[0]),  # trace
        MagicMock(id=artifact_ids[1]),  # context
        MagicMock(id=artifact_ids[2]),  # metrics
    ] * 20  # Repeat for multiple test cases

    # Checkpoint Service
    mock_checkpoint = mock_get_checkpoint_service.return_value
    mock_checkpoint.get_job = AsyncMock()
    mock_checkpoint.get_job.return_value = None  # Force creation
    mock_checkpoint.create_job = AsyncMock(
        side_effect=lambda eid, total: EvaluationJob(
            evaluation_id=eid, progress_total=total, progress_current=0
        )
    )
    mock_checkpoint.update_evaluation_status = AsyncMock()
    mock_checkpoint.update_progress = AsyncMock()
    mock_checkpoint.save_checkpoint = AsyncMock()
    mock_checkpoint.complete_job = AsyncMock()
    mock_checkpoint.fail_job = AsyncMock()

    return mock_adapter, mock_checkpoint


@pytest.mark.asyncio
async def test_evaluation_full_lifecycle(db_session: AsyncSession, caplog: LogCaptureFixture) -> None:
    """Test that the evaluation runner completes successfully for a full set."""
    evaluation, artifact_ids = await create_setup_data(db_session, case_count=3)

    mock_rag = MagicMock()
    mock_rag.query_with_trace = AsyncMock(
        return_value={
            "answer": "Mock Answer",
            "context": ["Mock Context"],
            "metadata": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            "retrieval_trace": {"strategy": "vector"},
        }
    )

    patches = get_mock_patches()
    with ExitStack() as stack:
        mock_adapter, mock_checkpoint = setup_common_mocks(stack, patches, mock_rag, artifact_ids)

        runner = EvaluationRunner(db_session, evaluation.id)
        await runner.run()

        # Check if fail_job was called (debugging)
        if mock_checkpoint.fail_job.called:
            args = mock_checkpoint.fail_job.call_args
            # Print captured logs
            print("\nCaptured Logs:")
            for record in caplog.records:
                print(f"{record.levelname}: {record.message}")
            pytest.fail(f"Evaluation failed with: {args}")

        # Refresh and verify
        await db_session.refresh(evaluation)
        # Since we mocked checkpoint_service, we need to check if complete_job was called
        mock_checkpoint.complete_job.assert_called_once()

        # Check results were added to session
        from sqlalchemy import select

        res = await db_session.execute(
            select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation.id)
        )
        results = res.scalars().all()
        assert len(results) == 3


@pytest.mark.asyncio
async def test_evaluation_cancel_lifecycle(db_session: AsyncSession, caplog: LogCaptureFixture) -> None:
    """Test cancellation mid-run."""
    evaluation, artifact_ids = await create_setup_data(db_session, case_count=4)

    mock_rag = MagicMock()
    original_return = {
        "answer": "Mock Answer",
        "context": ["Mock Context"],
        "metadata": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        "retrieval_trace": {"strategy": "vector"},
    }

    call_count = 0

    async def query_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            EvaluationRunner.cancel(evaluation.id)
        return original_return

    mock_rag.query_with_trace = AsyncMock(side_effect=query_side_effect)

    patches = get_mock_patches()
    with ExitStack() as stack:
        mock_adapter, mock_checkpoint = setup_common_mocks(stack, patches, mock_rag, artifact_ids)

        # Override query_with_trace with our side effect
        mock_adapter.query_with_trace = mock_rag.query_with_trace

        runner = EvaluationRunner(db_session, evaluation.id)
        await runner.run()

        if mock_checkpoint.fail_job.called:
            fail_args = mock_checkpoint.fail_job.call_args
            print("\nCaptured Logs:")
            for record in caplog.records:
                print(f"{record.levelname}: {record.message}")
            pytest.fail(f"Evaluation failed with: {fail_args}")

        # Check results: processed 2, cancelled before 3rd
        from sqlalchemy import select

        res = await db_session.execute(
            select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation.id)
        )
        results = res.scalars().all()
        assert len(results) == 2

        # Verify cancellation was reported to checkpoint
        mock_checkpoint.update_evaluation_status.assert_any_call(evaluation.id, "cancelled")


@pytest.mark.asyncio
async def test_evaluation_pause_resume_lifecycle(db_session: AsyncSession, caplog: LogCaptureFixture) -> None:
    """Test pause and resume cycle."""
    evaluation, artifact_ids = await create_setup_data(db_session, case_count=4)

    mock_rag = MagicMock()
    original_return = {
        "answer": "Mock Answer",
        "context": ["Mock Context"],
        "metadata": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        "retrieval_trace": {"strategy": "vector"},
    }

    call_count = 0

    async def query_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            EvaluationRunner.pause(evaluation.id)
        return original_return

    mock_rag.query_with_trace = AsyncMock(side_effect=query_side_effect)

    patches = get_mock_patches()
    with ExitStack() as stack:
        mock_adapter, mock_checkpoint = setup_common_mocks(stack, patches, mock_rag, artifact_ids)
        mock_adapter.query_with_trace = mock_rag.query_with_trace

        # Part 1: Run until pause
        runner1 = EvaluationRunner(db_session, evaluation.id)
        await runner1.run()

        if mock_checkpoint.fail_job.called:
            args = mock_checkpoint.fail_job.call_args
            print("\nCaptured Logs (Run 1):")
            for record in caplog.records:
                print(f"{record.levelname}: {record.message}")
            pytest.fail(f"Run 1 failed with: {args}")

        # Should have 2 results
        from sqlalchemy import select

        res = await db_session.execute(
            select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation.id)
        )
        results = res.scalars().all()
        assert len(results) == 2

        # Verify pause reported
        mock_checkpoint.update_evaluation_status.assert_any_call(evaluation.id, "paused")

        # Part 2: Resume
        EvaluationRunner.resume(evaluation.id)

        # Configure checkpoint mock to return job with progress 2
        mock_job = EvaluationJob(evaluation_id=evaluation.id, progress_current=2, progress_total=4)
        mock_checkpoint.get_job.return_value = mock_job

        # Reset side effect for second run
        async def query_side_effect_resume(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return original_return

        mock_adapter.query_with_trace = AsyncMock(side_effect=query_side_effect_resume)

        runner2 = EvaluationRunner(db_session, evaluation.id)
        await runner2.run()

        if mock_checkpoint.fail_job.called:
            args = mock_checkpoint.fail_job.call_args
            print("\nCaptured Logs (Run 2):")
            for record in caplog.records:
                print(f"{record.levelname}: {record.message}")
            pytest.fail(f"Run 2 failed with: {args}")

        # Total results should be 4
        res = await db_session.execute(
            select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation.id)
        )
        results = res.scalars().all()
        assert len(results) == 4

        # Verify completion reported
        mock_checkpoint.complete_job.assert_called_once()
