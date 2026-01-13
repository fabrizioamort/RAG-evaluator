"""Tests for projects API endpoints."""

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project


@pytest.fixture
async def sample_project(db_session: AsyncSession) -> Project:
    """Create a sample project for testing."""
    project = Project(
        name="Test Project",
        description="A test project for unit tests",
        status="active",
        tags=["test", "sample"],
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def multiple_projects(db_session: AsyncSession) -> list[Project]:
    """Create multiple projects for testing pagination and filtering."""
    projects = [
        Project(name="Project Alpha", status="active", tags=["alpha", "test"]),
        Project(name="Project Beta", status="active", tags=["beta", "test"]),
        Project(name="Project Gamma", status="archived", tags=["gamma"]),
        Project(name="Project Delta", status="active", tags=["delta"]),
        Project(name="Project Epsilon", status="archived", tags=["epsilon"]),
    ]
    for project in projects:
        db_session.add(project)
    await db_session.commit()
    for project in projects:
        await db_session.refresh(project)
    return projects


class TestListProjects:
    """Tests for GET /api/v1/projects endpoint."""

    @pytest.mark.asyncio
    async def test_list_projects_empty(self, client: AsyncClient) -> None:
        """Test listing projects when none exist."""
        response = await client.get("/api/v1/projects")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["offset"] == 0
        assert data["limit"] == 20

    @pytest.mark.asyncio
    async def test_list_projects_with_data(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test listing projects with existing data."""
        response = await client.get("/api/v1/projects")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Test Project"
        assert data["items"][0]["status"] == "active"
        assert data["items"][0]["tags"] == ["test", "sample"]

    @pytest.mark.asyncio
    async def test_list_projects_pagination(
        self, client: AsyncClient, multiple_projects: list[Project]
    ) -> None:
        """Test pagination works correctly."""
        # Get first page with limit 2
        response = await client.get("/api/v1/projects?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["offset"] == 0
        assert data["limit"] == 2

        # Get second page
        response = await client.get("/api/v1/projects?limit=2&offset=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["offset"] == 2

    @pytest.mark.asyncio
    async def test_list_projects_status_filter(
        self, client: AsyncClient, multiple_projects: list[Project]
    ) -> None:
        """Test filtering by status."""
        # Filter active projects
        response = await client.get("/api/v1/projects?status=active")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        for item in data["items"]:
            assert item["status"] == "active"

        # Filter archived projects
        response = await client.get("/api/v1/projects?status=archived")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert item["status"] == "archived"

    @pytest.mark.asyncio
    async def test_list_projects_invalid_status_filter(self, client: AsyncClient) -> None:
        """Test that invalid status filter returns validation error."""
        response = await client.get("/api/v1/projects?status=invalid")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_list_projects_response_includes_counts(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test that project response includes relationship counts."""
        response = await client.get("/api/v1/projects")

        assert response.status_code == 200
        data = response.json()
        project_data = data["items"][0]
        assert "knowledge_base_count" in project_data
        assert "test_set_count" in project_data
        assert "rag_config_count" in project_data
        assert "evaluation_count" in project_data
        assert project_data["knowledge_base_count"] == 0


class TestCreateProject:
    """Tests for POST /api/v1/projects endpoint."""

    @pytest.mark.asyncio
    async def test_create_project_success(self, client: AsyncClient) -> None:
        """Test creating a project successfully."""
        payload = {
            "name": "New Project",
            "description": "A newly created project",
            "tags": ["new", "created"],
        }

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Project"
        assert data["description"] == "A newly created project"
        assert data["status"] == "active"
        assert data["tags"] == ["new", "created"]
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_project_minimal(self, client: AsyncClient) -> None:
        """Test creating a project with minimal data."""
        payload = {"name": "Minimal Project"}

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Project"
        assert data["description"] is None
        assert data["status"] == "active"
        assert data["tags"] == []

    @pytest.mark.asyncio
    async def test_create_project_empty_name(self, client: AsyncClient) -> None:
        """Test that empty name fails validation."""
        payload = {"name": ""}

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_project_missing_name(self, client: AsyncClient) -> None:
        """Test that missing name fails validation."""
        payload = {"description": "No name provided"}

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_project_name_too_long(self, client: AsyncClient) -> None:
        """Test that name exceeding max length fails validation."""
        payload = {"name": "x" * 256}  # Max is 255

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 422  # Validation error


class TestGetProject:
    """Tests for GET /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_project_success(self, client: AsyncClient, sample_project: Project) -> None:
        """Test getting a project by ID."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_project.id)
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project for unit tests"
        assert data["status"] == "active"
        assert data["tags"] == ["test", "sample"]

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent project."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/projects/{fake_id}")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_project_invalid_uuid(self, client: AsyncClient) -> None:
        """Test getting a project with invalid UUID format."""
        response = await client.get("/api/v1/projects/not-a-uuid")

        assert response.status_code == 422  # Validation error


class TestUpdateProject:
    """Tests for PUT /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_project_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test updating a project successfully."""
        payload = {
            "name": "Updated Project Name",
            "description": "Updated description",
        }

        response = await client.put(f"/api/v1/projects/{sample_project.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Project Name"
        assert data["description"] == "Updated description"
        # Status and tags should remain unchanged
        assert data["status"] == "active"
        assert data["tags"] == ["test", "sample"]

    @pytest.mark.asyncio
    async def test_update_project_partial(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test partial update with only some fields."""
        payload = {"name": "Only Name Updated"}

        response = await client.put(f"/api/v1/projects/{sample_project.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Only Name Updated"
        # Other fields should remain unchanged
        assert data["description"] == "A test project for unit tests"

    @pytest.mark.asyncio
    async def test_update_project_status(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test updating project status."""
        payload = {"status": "archived"}

        response = await client.put(f"/api/v1/projects/{sample_project.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"

    @pytest.mark.asyncio
    async def test_update_project_invalid_status(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test that invalid status fails validation."""
        payload = {"status": "invalid_status"}

        response = await client.put(f"/api/v1/projects/{sample_project.id}", json=payload)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_update_project_not_found(self, client: AsyncClient) -> None:
        """Test updating a non-existent project."""
        fake_id = uuid4()
        payload = {"name": "Updated"}

        response = await client.put(f"/api/v1/projects/{fake_id}", json=payload)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_project_tags(self, client: AsyncClient, sample_project: Project) -> None:
        """Test updating project tags."""
        payload = {"tags": ["updated", "new-tags"]}

        response = await client.put(f"/api/v1/projects/{sample_project.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == ["updated", "new-tags"]


class TestDeleteProject:
    """Tests for DELETE /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_project_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test deleting a project successfully."""
        project_id = sample_project.id

        response = await client.delete(f"/api/v1/projects/{project_id}")

        assert response.status_code == 204

        # Verify project is actually deleted
        get_response = await client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_not_found(self, client: AsyncClient) -> None:
        """Test deleting a non-existent project."""
        fake_id = uuid4()

        response = await client.delete(f"/api/v1/projects/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_invalid_uuid(self, client: AsyncClient) -> None:
        """Test deleting with invalid UUID format."""
        response = await client.delete("/api/v1/projects/not-a-uuid")

        assert response.status_code == 422


class TestArchiveProject:
    """Tests for POST /api/v1/projects/{project_id}/archive endpoint."""

    @pytest.mark.asyncio
    async def test_archive_project_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test archiving a project successfully."""
        response = await client.post(f"/api/v1/projects/{sample_project.id}/archive")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"
        assert data["name"] == "Test Project"  # Other fields unchanged

    @pytest.mark.asyncio
    async def test_archive_project_already_archived(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test archiving an already archived project."""
        # Create an archived project
        project = Project(name="Already Archived", status="archived")
        db_session.add(project)
        await db_session.commit()
        await db_session.refresh(project)

        response = await client.post(f"/api/v1/projects/{project.id}/archive")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "archived"

    @pytest.mark.asyncio
    async def test_archive_project_not_found(self, client: AsyncClient) -> None:
        """Test archiving a non-existent project."""
        fake_id = uuid4()

        response = await client.post(f"/api/v1/projects/{fake_id}/archive")

        assert response.status_code == 404


class TestProjectRelationshipCounts:
    """Tests for verifying relationship counts in project responses."""

    @pytest.mark.asyncio
    async def test_new_project_has_zero_counts(self, client: AsyncClient) -> None:
        """Test that a newly created project has zero relationship counts."""
        payload = {"name": "Fresh Project"}

        response = await client.post("/api/v1/projects", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["knowledge_base_count"] == 0
        assert data["test_set_count"] == 0
        assert data["rag_config_count"] == 0
        assert data["evaluation_count"] == 0
