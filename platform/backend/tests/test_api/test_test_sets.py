"""Tests for test sets API endpoints."""

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.models.project import Project
from app.models.test_case import TestCase
from app.models.test_generation_job import TestGenerationJob
from app.models.test_set import TestSet


@pytest.fixture
async def sample_project(db_session: AsyncSession) -> Project:
    """Create a sample project for testing."""
    project = Project(
        name="Test Project",
        description="A test project for unit tests",
        status="active",
        tags=["test"],
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def sample_test_set(db_session: AsyncSession, sample_project: Project) -> TestSet:
    """Create a sample test set for testing."""
    test_set = TestSet(
        project_id=sample_project.id,
        name="Test Set",
        description="A test set for unit tests",
        tags=["test"],
    )
    db_session.add(test_set)
    await db_session.commit()
    await db_session.refresh(test_set)
    return test_set


@pytest.fixture
async def sample_test_case(db_session: AsyncSession, sample_test_set: TestSet) -> TestCase:
    """Create a sample test case for testing."""
    test_case = TestCase(
        test_set_id=sample_test_set.id,
        question="What is the capital of France?",
        expected_answer="Paris",
        ground_truth_context=["Paris is the capital of France."],
        difficulty="easy",
        question_type="factual",
        is_reviewed=True,
    )
    db_session.add(test_case)
    await db_session.commit()
    await db_session.refresh(test_case)
    return test_case


class TestListTestSets:
    """Tests for GET /api/v1/projects/{project_id}/test-sets endpoint."""

    @pytest.mark.asyncio
    async def test_list_test_sets_empty(self, client: AsyncClient, sample_project: Project) -> None:
        """Test listing test sets when none exist."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}/test-sets")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_test_sets_with_data(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test listing test sets with existing data."""
        response = await client.get(f"/api/v1/projects/{sample_test_set.project_id}/test-sets")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Test Set"


class TestCreateTestSet:
    """Tests for POST /api/v1/projects/{project_id}/test-sets endpoint."""

    @pytest.mark.asyncio
    async def test_create_test_set_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test creating a test set successfully."""
        payload = {
            "name": "New Test Set",
            "description": "A newly created test set",
            "tags": ["new"],
        }

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/test-sets", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Test Set"
        assert data["project_id"] == str(sample_project.id)


class TestGetTestSet:
    """Tests for GET /api/v1/test-sets/{test_set_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_test_set_success(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test getting a test set with cases."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_test_set.id)
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["question"] == sample_test_case.question

    @pytest.mark.asyncio
    async def test_get_test_set_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent test set."""
        response = await client.get(f"/api/v1/test-sets/{uuid4()}")
        assert response.status_code == 404


class TestUpdateTestSet:
    """Tests for PUT /api/v1/test-sets/{test_set_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_test_set_success(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test updating a test set."""
        payload = {"name": "Updated Name"}
        response = await client.put(f"/api/v1/test-sets/{sample_test_set.id}", json=payload)

        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"


class TestDeleteTestSet:
    """Tests for DELETE /api/v1/test-sets/{test_set_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_test_set_success(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test deleting a test set."""
        response = await client.delete(f"/api/v1/test-sets/{sample_test_set.id}")
        assert response.status_code == 204

        # Verify deletion
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}")
        assert response.status_code == 404


class TestTestCaseCRUD:
    """Tests for test case management endpoints."""

    @pytest.mark.asyncio
    async def test_add_test_case_success(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test adding a test case."""
        payload = {
            "question": "What is 2+2?",
            "expected_answer": "4",
            "ground_truth_context": ["Basic arithmetic."],
            "difficulty": "easy",
            "question_type": "factual",
        }
        response = await client.post(f"/api/v1/test-sets/{sample_test_set.id}/cases", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["question"] == "What is 2+2?"
        assert data["test_set_id"] == str(sample_test_set.id)

    @pytest.mark.asyncio
    async def test_bulk_add_test_cases(self, client: AsyncClient, sample_test_set: TestSet) -> None:
        """Test bulk adding test cases."""
        payload = {
            "test_cases": [
                {
                    "question": "Q1",
                    "expected_answer": "A1",
                    "ground_truth_context": ["C1"],
                },
                {
                    "question": "Q2",
                    "expected_answer": "A2",
                    "ground_truth_context": ["C2"],
                },
            ]
        }
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/cases/bulk", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 2
        assert data[0]["question"] == "Q1"
        assert data[1]["question"] == "Q2"

    @pytest.mark.asyncio
    async def test_update_test_case(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test updating a test case."""
        payload = {"question": "Updated Question"}
        response = await client.put(
            f"/api/v1/test-sets/{sample_test_set.id}/cases/{sample_test_case.id}", json=payload
        )

        assert response.status_code == 200
        assert response.json()["question"] == "Updated Question"

    @pytest.mark.asyncio
    async def test_delete_test_case(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test deleting a test case."""
        response = await client.delete(
            f"/api/v1/test-sets/{sample_test_set.id}/cases/{sample_test_case.id}"
        )
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_bulk_review_approve(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test bulk approving test cases."""
        # Un-review the case first
        sample_test_case.is_reviewed = False
        # (Already in DB from fixture, but we can update it in test if needed,
        # but let's just use it as is if we assume it's generated)

        payload = {"test_case_ids": [str(sample_test_case.id)], "action": "approve"}
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/cases/bulk-review", json=payload
        )

        assert response.status_code == 200
        assert response.json()["action"] == "approved"
        assert response.json()["count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_review_reject(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test bulk rejecting test cases."""
        payload = {"test_case_ids": [str(sample_test_case.id)], "action": "reject"}
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/cases/bulk-review", json=payload
        )

        assert response.status_code == 200
        assert response.json()["action"] == "rejected"
        assert response.json()["count"] == 1


class TestImportExport:
    """Tests for import/export endpoints."""

    @pytest.mark.asyncio
    async def test_import_test_set_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test importing a test set."""
        payload = {
            "name": "Imported Set",
            "description": "Imported desc",
            "tags": ["imported"],
            "test_cases": [
                {
                    "question": "IQ1",
                    "expected_answer": "IA1",
                    "ground_truth_context": ["IC1"],
                }
            ],
        }
        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/test-sets/import", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Imported Set"

        # Verify cases were created
        get_response = await client.get(f"/api/v1/test-sets/{data['id']}")
        assert len(get_response.json()["test_cases"]) == 1

    @pytest.mark.asyncio
    async def test_export_test_set_success(
        self, client: AsyncClient, sample_test_set: TestSet, sample_test_case: TestCase
    ) -> None:
        """Test exporting a test set."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}/export")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_test_set.name
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["question"] == sample_test_case.question


# =============================================================================
# Test Generation API Tests
# =============================================================================


@pytest.fixture
async def sample_knowledge_base(db_session: AsyncSession, sample_project: Project) -> KnowledgeBase:
    """Create a sample knowledge base for testing."""
    kb = KnowledgeBase(
        project_id=sample_project.id,
        name="Test KB",
        description="A test knowledge base",
        status="ready",
    )
    db_session.add(kb)
    await db_session.commit()
    await db_session.refresh(kb)
    return kb


@pytest.fixture
async def sample_generation_job(
    db_session: AsyncSession, sample_test_set: TestSet, sample_knowledge_base: KnowledgeBase
) -> TestGenerationJob:
    """Create a sample generation job for testing."""
    job = TestGenerationJob(
        test_set_id=sample_test_set.id,
        knowledge_base_id=sample_knowledge_base.id,
        status="completed",
        config={"target_count": 10},
        questions_generated=8,
        questions_total=10,
        questions_rejected=2,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    return job


class TestGenerationAPI:
    """Tests for test generation API endpoints."""

    @pytest.mark.asyncio
    async def test_start_generation_success(
        self,
        client: AsyncClient,
        sample_test_set: TestSet,
        sample_knowledge_base: KnowledgeBase,
    ) -> None:
        """Test starting a generation job."""
        payload = {
            "knowledge_base_id": str(sample_knowledge_base.id),
            "target_count": 10,
            "questions_per_chunk": 2,
            "llm_model": "gpt-4o-mini",
        }
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/generate", json=payload
        )

        assert response.status_code == 202
        data = response.json()
        assert data["test_set_id"] == str(sample_test_set.id)
        assert data["knowledge_base_id"] == str(sample_knowledge_base.id)
        assert data["status"] == "pending"
        assert data["questions_total"] == 10

    @pytest.mark.asyncio
    async def test_start_generation_kb_not_found(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test starting generation with non-existent KB."""
        payload = {
            "knowledge_base_id": str(uuid4()),
            "target_count": 10,
        }
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/generate", json=payload
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_start_generation_test_set_not_found(
        self, client: AsyncClient, sample_knowledge_base: KnowledgeBase
    ) -> None:
        """Test starting generation with non-existent test set."""
        payload = {
            "knowledge_base_id": str(sample_knowledge_base.id),
            "target_count": 10,
        }
        response = await client.post(f"/api/v1/test-sets/{uuid4()}/generate", json=payload)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_start_generation_conflict(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_test_set: TestSet,
        sample_knowledge_base: KnowledgeBase,
    ) -> None:
        """Test starting generation when one is already running."""
        # Create a running job
        running_job = TestGenerationJob(
            test_set_id=sample_test_set.id,
            knowledge_base_id=sample_knowledge_base.id,
            status="running",
            config={},
        )
        db_session.add(running_job)
        await db_session.commit()

        payload = {
            "knowledge_base_id": str(sample_knowledge_base.id),
            "target_count": 10,
        }
        response = await client.post(
            f"/api/v1/test-sets/{sample_test_set.id}/generate", json=payload
        )

        assert response.status_code == 409
        assert "already running" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_generation_status_success(
        self,
        client: AsyncClient,
        sample_test_set: TestSet,
        sample_generation_job: TestGenerationJob,
    ) -> None:
        """Test getting generation status."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}/generation-status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(sample_generation_job.id)
        assert data["status"] == "completed"
        assert data["questions_generated"] == 8
        assert data["questions_rejected"] == 2
        assert data["progress"] == 1.0  # Completed job has progress 1.0

    @pytest.mark.asyncio
    async def test_get_generation_status_not_found(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test getting status when no job exists."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}/generation-status")

        assert response.status_code == 404
        assert "no generation job" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_cancel_generation_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_test_set: TestSet,
        sample_knowledge_base: KnowledgeBase,
    ) -> None:
        """Test cancelling a running generation job."""
        # Create a running job
        running_job = TestGenerationJob(
            test_set_id=sample_test_set.id,
            knowledge_base_id=sample_knowledge_base.id,
            status="running",
            config={},
        )
        db_session.add(running_job)
        await db_session.commit()

        response = await client.delete(f"/api/v1/test-sets/{sample_test_set.id}/generation")

        assert response.status_code == 204

        # Verify job is cancelled
        await db_session.refresh(running_job)
        assert running_job.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_generation_no_active_job(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test cancelling when no active job exists."""
        response = await client.delete(f"/api/v1/test-sets/{sample_test_set.id}/generation")

        assert response.status_code == 404
        assert "no active" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_generation_jobs_success(
        self,
        client: AsyncClient,
        sample_test_set: TestSet,
        sample_generation_job: TestGenerationJob,
    ) -> None:
        """Test listing generation jobs."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}/generation-jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == str(sample_generation_job.id)
        assert data[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_generation_jobs_empty(
        self, client: AsyncClient, sample_test_set: TestSet
    ) -> None:
        """Test listing generation jobs when none exist."""
        response = await client.get(f"/api/v1/test-sets/{sample_test_set.id}/generation-jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
