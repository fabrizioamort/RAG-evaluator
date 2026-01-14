"""Tests for evaluations API endpoints."""

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.models.test_case import TestCase
from app.models.test_set import TestSet


@pytest.fixture
async def sample_project(db_session: AsyncSession) -> Project:
    """Create a sample project."""
    project = Project(name="Eval Test Project")
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def sample_kb(db_session: AsyncSession, sample_project: Project) -> KnowledgeBase:
    """Create a sample knowledge base."""
    kb = KnowledgeBase(
        project_id=sample_project.id,
        name="Test KB",
        status="ready",
        current_version=1,
    )
    db_session.add(kb)
    await db_session.commit()
    await db_session.refresh(kb)
    return kb


@pytest.fixture
async def sample_test_set(db_session: AsyncSession, sample_project: Project) -> TestSet:
    """Create a sample test set with cases."""
    test_set = TestSet(project_id=sample_project.id, name="Test Set")
    db_session.add(test_set)
    await db_session.flush()

    case = TestCase(
        test_set_id=test_set.id,
        question="What is 1+1?",
        expected_answer="2",
    )
    db_session.add(case)
    await db_session.commit()
    await db_session.refresh(test_set)
    return test_set


@pytest.fixture
async def sample_rag_config(db_session: AsyncSession, sample_project: Project) -> RAGConfig:
    """Create a sample RAG config."""
    config = RAGConfig(
        project_id=sample_project.id,
        name="Test RAG",
        rag_type="vector_semantic",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)
    return config


class TestCreateEvaluation:
    """Tests for POST /api/v1/evaluations."""

    @pytest.mark.asyncio
    async def test_create_evaluation_success(
        self,
        client: AsyncClient,
        sample_kb: KnowledgeBase,
        sample_test_set: TestSet,
        sample_rag_config: RAGConfig,
    ) -> None:
        """Test starting a new evaluation."""
        payload = {
            "knowledge_base_id": str(sample_kb.id),
            "test_set_id": str(sample_test_set.id),
            "rag_config_id": str(sample_rag_config.id),
            "notes": "Test evaluation",
            "tags": ["test"],
        }

        response = await client.post("/api/v1/evaluations", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        assert data["knowledge_base_id"] == str(sample_kb.id)
        assert data["test_set_id"] == str(sample_test_set.id)
        assert data["rag_config_id"] == str(sample_rag_config.id)
        assert "run_manifest_id" in data
        assert data["notes"] == "Test evaluation"
        assert data["tags"] == ["test"]

    @pytest.mark.asyncio
    async def test_create_evaluation_invalid_ids(self, client: AsyncClient) -> None:
        """Test with non-existent IDs."""
        payload = {
            "knowledge_base_id": str(uuid4()),
            "test_set_id": str(uuid4()),
            "rag_config_id": str(uuid4()),
        }

        response = await client.post("/api/v1/evaluations", json=payload)
        assert response.status_code == 404


@pytest.fixture
async def sample_evaluation(
    db_session: AsyncSession,
    sample_project: Project,
    sample_kb: KnowledgeBase,
    sample_test_set: TestSet,
    sample_rag_config: RAGConfig,
) -> Evaluation:
    """Create a sample evaluation."""
    evaluation = Evaluation(
        project_id=sample_project.id,
        knowledge_base_id=sample_kb.id,
        test_set_id=sample_test_set.id,
        rag_config_id=sample_rag_config.id,
        status="completed",
        pass_rate=0.8,
    )
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)
    return evaluation


class TestGetEvaluation:
    """Tests for GET /api/v1/evaluations endpoints."""

    @pytest.mark.asyncio
    async def test_get_evaluation_detail(
        self, client: AsyncClient, sample_evaluation: Evaluation
    ) -> None:
        """Test getting evaluation detail."""
        response = await client.get(f"/api/v1/evaluations/{sample_evaluation.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_evaluation.id)
        assert data["status"] == "completed"
        assert data["pass_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_list_evaluations(
        self, client: AsyncClient, sample_project: Project, sample_evaluation: Evaluation
    ) -> None:
        """Test listing evaluations for a project."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}/evaluations")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == str(sample_evaluation.id)


class TestEvaluationControl:
    """Tests for evaluation control endpoints."""

    @pytest.fixture
    async def running_evaluation(
        self,
        db_session: AsyncSession,
        sample_project: Project,
        sample_kb: KnowledgeBase,
        sample_test_set: TestSet,
        sample_rag_config: RAGConfig,
    ) -> Evaluation:
        """Create a running evaluation."""
        evaluation = Evaluation(
            project_id=sample_project.id,
            knowledge_base_id=sample_kb.id,
            test_set_id=sample_test_set.id,
            rag_config_id=sample_rag_config.id,
            status="running",
        )
        db_session.add(evaluation)
        await db_session.commit()
        await db_session.refresh(evaluation)
        return evaluation

    @pytest.mark.asyncio
    async def test_cancel_evaluation(
        self, client: AsyncClient, running_evaluation: Evaluation
    ) -> None:
        """Test cancelling an evaluation."""
        response = await client.post(f"/api/v1/evaluations/{running_evaluation.id}/cancel")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_pause_evaluation(
        self, client: AsyncClient, running_evaluation: Evaluation
    ) -> None:
        """Test pausing an evaluation."""
        response = await client.post(f"/api/v1/evaluations/{running_evaluation.id}/pause")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_resume_evaluation_wrong_status(
        self, client: AsyncClient, running_evaluation: Evaluation
    ) -> None:
        """Test resuming an evaluation that is not paused."""
        response = await client.post(f"/api/v1/evaluations/{running_evaluation.id}/resume")
        assert response.status_code == 400
        assert "Cannot resume" in response.json()["detail"]


class TestSetBaseline:
    """Tests for POST /api/v1/evaluations/{id}/set-baseline."""

    @pytest.mark.asyncio
    async def test_set_baseline_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_project: Project,
        sample_kb: KnowledgeBase,
        sample_test_set: TestSet,
        sample_rag_config: RAGConfig,
    ) -> None:
        """Test setting an evaluation as baseline."""
        # Create a completed evaluation
        evaluation = Evaluation(
            project_id=sample_project.id,
            knowledge_base_id=sample_kb.id,
            test_set_id=sample_test_set.id,
            rag_config_id=sample_rag_config.id,
            status="completed",
        )
        db_session.add(evaluation)
        await db_session.commit()

        payload = {"reason": "Initial golden baseline"}
        response = await client.post(
            f"/api/v1/evaluations/{evaluation.id}/set-baseline", json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_baseline"] is True
        assert data["baseline_reason"] == "Initial golden baseline"

        # Verify old baseline is cleared (if we had one)
        # Create another completed evaluation
        eval2 = Evaluation(
            project_id=sample_project.id,
            status="completed",
        )
        db_session.add(eval2)
        await db_session.commit()

        await client.post(
            f"/api/v1/evaluations/{eval2.id}/set-baseline", json={"reason": "New baseline"}
        )

        # Check first eval
        await db_session.refresh(evaluation)
        assert evaluation.is_baseline is False


class TestGetEvaluationManifest:
    """Tests for GET /api/v1/evaluations/{id}/manifest."""

    @pytest.mark.asyncio
    async def test_get_manifest_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_evaluation: Evaluation,
    ) -> None:
        """Test getting evaluation manifest."""
        from app.models.run_manifest import RunManifest

        # Create a manifest
        manifest = RunManifest(
            rag_config_snapshot={"test": "config"},
            kb_version_snapshot={"key": "val"},
            generation_model="gpt-4",
            prompt_templates={"p": "t"},
        )
        db_session.add(manifest)
        await db_session.flush()

        # Link to evaluation
        sample_evaluation.run_manifest_id = manifest.id
        await db_session.commit()

        response = await client.get(f"/api/v1/evaluations/{sample_evaluation.id}/manifest")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(manifest.id)
        assert data["rag_config_snapshot"] == {"test": "config"}
        assert data["generation_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_manifest_not_found(self, client: AsyncClient) -> None:
        """Test with non-existent evaluation."""
        response = await client.get(f"/api/v1/evaluations/{uuid4()}/manifest")
        assert response.status_code == 404
