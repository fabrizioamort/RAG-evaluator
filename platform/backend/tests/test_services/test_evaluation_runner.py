"""Tests for EvaluationRunner service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.models.test_case import TestCase
from app.models.test_set import TestSet
from app.services.evaluation_runner import EvaluationRunner


@pytest.fixture
async def setup_data(db_session: AsyncSession) -> Evaluation:
    """Setup basic data for evaluation tests."""
    project = Project(name="Test Project")
    db_session.add(project)
    await db_session.flush()

    rag_config = RAGConfig(
        project_id=project.id,
        name="Test RAG",
        rag_type="vector_semantic",
        llm_model="gpt-4o-mini",
        parameters={},
    )
    db_session.add(rag_config)

    test_set = TestSet(project_id=project.id, name="Test Set")
    db_session.add(test_set)
    await db_session.flush()

    test_case = TestCase(
        test_set_id=test_set.id,
        question="What is unit testing?",
        expected_answer="Testing small units of code.",
        ground_truth_context=["Unit testing is..."],
    )
    db_session.add(test_case)

    evaluation = Evaluation(
        project_id=project.id,
        test_set_id=test_set.id,
        rag_config_id=rag_config.id,
        status="pending",
    )
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)

    return evaluation


@pytest.mark.asyncio
async def test_evaluation_runner_success(db_session: AsyncSession, setup_data: Evaluation) -> None:
    """Test that the evaluation runner completes successfully."""
    evaluation = setup_data

    # Mock RAG adapter result
    mock_rag_instance = MagicMock()
    mock_rag_instance.query_with_trace = AsyncMock(
        return_value={
            "answer": "Testing small units of code in isolation.",
            "context": ["Unit testing definition..."],
            "metadata": {
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "cost": 0.001,
            },
            "retrieval_trace": {"strategy": "vector", "steps": []},
        }
    )

    # Mock DeepEval metrics
    with (
        patch("app.services.evaluation_runner.get_rag_adapter_service") as mock_get_rag_service,
        patch("app.services.evaluation_runner.FaithfulnessMetric") as mock_faith,
        patch("app.services.evaluation_runner.AnswerRelevancyMetric") as mock_rel,
        patch("app.services.evaluation_runner.ContextualPrecisionMetric") as mock_prec,
        patch("app.services.evaluation_runner.ContextualRecallMetric") as mock_recall,
        patch("app.services.evaluation_runner.GEval") as mock_geval,
        patch("app.services.evaluation_runner.get_job_event_log") as mock_get_event_log,
    ):
        # Setup RAG Adapter mock
        mock_adapter = mock_get_rag_service.return_value
        mock_adapter.get_or_create_rag.return_value = mock_rag_instance
        mock_adapter.query_with_trace = mock_rag_instance.query_with_trace

        # Setup metric mocks
        for m_class, name in [
            (mock_faith, "FaithfulnessMetric"),
            (mock_rel, "AnswerRelevancyMetric"),
            (mock_prec, "ContextualPrecisionMetric"),
            (mock_recall, "ContextualRecallMetric"),
            (mock_geval, "GEval"),
        ]:
            inst = MagicMock()
            inst.__class__.__name__ = name
            inst.score = 0.9
            inst.reason = "Good"
            inst.a_measure = AsyncMock()
            inst.measure = MagicMock()
            m_class.return_value = inst

        mock_event_log = mock_get_event_log.return_value
        mock_event_log.log_event = AsyncMock()

        # Mock checkpoint service methods if needed, but they are real for now
        # unless we patch them too. They use self.db which is db_session.

        # Instantiate runner INSIDE the patch to ensure it gets the mock adapter
        runner = EvaluationRunner(db_session, evaluation.id)

        # Run evaluation
        await runner.run()

        # Verify evaluation status
        await db_session.refresh(evaluation)
        assert evaluation.status == "completed", f"Evaluation failed: {evaluation.error_message}"
        assert evaluation.pass_rate == 1.0, f"Pass rate is {evaluation.pass_rate}, expected 1.0"
        assert evaluation.summary_metrics is not None
        assert evaluation.summary_metrics["overall_avg"] == 0.9

        # Verify job status
        from sqlalchemy import select

        from app.models.evaluation_job import EvaluationJob

        res = await db_session.execute(
            select(EvaluationJob).where(EvaluationJob.evaluation_id == evaluation.id)
        )
        job = res.scalar_one()
        assert job.state == "completed"
        assert job.progress_current == 1

        # Verify results were saved
        from app.models.evaluation_result import EvaluationResult

        res_count = await db_session.execute(
            select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation.id)
        )
        results = res_count.scalars().all()
        assert len(results) == 1
        assert results[0].faithfulness_score == 0.9
        assert results[0].retrieved_context_artifact_id is not None
        assert results[0].retrieval_trace_artifact_id is not None
        assert results[0].raw_metrics_artifact_id is not None


@pytest.mark.asyncio
async def test_evaluation_runner_cancellation(
    db_session: AsyncSession, setup_data: Evaluation
) -> None:
    """Test that the evaluation runner respects cancellation."""
    evaluation = setup_data

    # Signal cancellation immediately
    EvaluationRunner.cancel(evaluation.id)

    # Mock RAG adapter return for initialization
    mock_adapter_service = MagicMock()
    mock_adapter_service.get_or_create_rag.return_value = MagicMock()
    mock_adapter_service.query = AsyncMock()

    # Mock event log
    mock_event_log = MagicMock()
    mock_event_log.log_event = AsyncMock()

    with (
        patch(
            "app.services.evaluation_runner.get_rag_adapter_service",
            return_value=mock_adapter_service,
        ),
        patch("app.services.evaluation_runner.get_job_event_log", return_value=mock_event_log),
        patch(
            "app.services.evaluation_runner.EvaluationRunner._initialize_metrics"
        ) as mock_init_metrics,
    ):
        mock_init_metrics.return_value = []

        # Instantiate runner INSIDE the patch
        runner = EvaluationRunner(db_session, evaluation.id)

        await runner.run()

        # Verify status
        await db_session.refresh(evaluation)
        assert evaluation.status == "cancelled", (
            f"Evaluation status: {evaluation.status}, Error: {evaluation.error_message}"
        )


@pytest.mark.asyncio
async def test_evaluation_runner_ready_index_loads_without_prepare(
    db_session: AsyncSession,
) -> None:
    """Ready index evaluation should use load path and pass effective top_k."""
    project = Project(name="Indexed Project")
    db_session.add(project)
    await db_session.flush()

    kb = KnowledgeBase(
        project_id=project.id,
        name="KB",
        status="ready",
        storage_path="./storage/documents",
    )
    db_session.add(kb)

    rag_config = RAGConfig(
        project_id=project.id,
        name="RLM",
        rag_type="rlm_rag",
        llm_model="gpt-build",
        parameters={"worker_model": "gpt-worker"},
    )
    db_session.add(rag_config)

    test_set = TestSet(project_id=project.id, name="Test Set")
    db_session.add(test_set)
    await db_session.flush()

    test_case = TestCase(
        test_set_id=test_set.id,
        question="What is indexed?",
        expected_answer="The answer.",
        ground_truth_context=[],
    )
    db_session.add(test_case)
    await db_session.flush()

    index = KnowledgeBaseIndex(
        knowledge_base_id=kb.id,
        rag_config_id=rag_config.id,
        name="Ready Index",
        status="ready",
        physical_id="idx_runner_ready",
        storage_type="filesystem",
        config_snapshot={
            "rag_type": "rlm_rag",
            "parameters": {
                "worker_model": "gpt-worker",
                "orchestrator_model": "gpt-build",
            },
            "llm_provider": "openai",
            "llm_model": "gpt-build",
            "embedding_model": "text-embedding-3-small",
        },
        document_count=1,
    )
    db_session.add(index)
    await db_session.flush()

    evaluation = Evaluation(
        project_id=project.id,
        knowledge_base_id=kb.id,
        knowledge_base_index_id=index.id,
        rag_config_id=rag_config.id,
        test_set_id=test_set.id,
        status="pending",
        query_overrides={
            "llm_model": "gpt-query",
            "top_k": 11,
            "parameters": {"orchestrator_model": "gpt-query"},
        },
        eval_judge_model="gpt-judge",
        metric_config={"metrics": []},
    )
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)

    mock_rag = MagicMock()
    mock_effective = MagicMock(top_k=11, generation_model="gpt-query")
    mock_adapter = MagicMock()
    mock_adapter.load_rag_for_index_query.return_value = (mock_rag, mock_effective)
    mock_adapter.prepare_documents = AsyncMock()
    mock_adapter.query_with_trace = AsyncMock(
        return_value={
            "answer": "The answer.",
            "context": [],
            "metadata": {"token_usage": {"prompt_tokens": 3, "completion_tokens": 2}},
            "retrieval_trace": {"strategy": "vector", "steps": []},
        }
    )
    mock_event_log = MagicMock()
    mock_event_log.log_event = AsyncMock()

    with (
        patch("app.services.evaluation_runner.get_rag_adapter_service", return_value=mock_adapter),
        patch("app.services.evaluation_runner.get_job_event_log", return_value=mock_event_log),
        patch(
            "app.services.evaluation_runner.EvaluationRunner._initialize_metrics",
            return_value=[],
        ) as mock_init_metrics,
    ):
        runner = EvaluationRunner(db_session, evaluation.id)
        await runner.run()

    mock_adapter.load_rag_for_index_query.assert_called_once()
    mock_adapter.prepare_documents.assert_not_called()
    mock_adapter.query_with_trace.assert_awaited_once_with(mock_rag, "What is indexed?", 11)
    mock_init_metrics.assert_called_once_with("gpt-judge")
