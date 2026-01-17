"""Tests for test templates API endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.test_template import TestTemplate


@pytest.fixture
async def sample_template(db_session: AsyncSession) -> TestTemplate:
    """Create a sample custom template for testing."""
    template = TestTemplate(
        name="Custom Template",
        description="A custom template for testing",
        category="custom",
        question_template="What is {x}?",
        answer_template="{x} is y.",
        entity_types=["x"],
        complexity_level="medium",
        is_builtin=False,
    )
    db_session.add(template)
    await db_session.commit()
    await db_session.refresh(template)
    return template


@pytest.fixture
async def builtin_template(db_session: AsyncSession) -> TestTemplate:
    """Create a sample builtin template for testing."""
    template = TestTemplate(
        name="Builtin Template",
        description="A builtin template",
        category="factual",
        question_template="What is {entity}?",
        answer_template="{entity} is ...",
        entity_types=["entity"],
        complexity_level="easy",
        is_builtin=True,
    )
    db_session.add(template)
    await db_session.commit()
    await db_session.refresh(template)
    return template


class TestListTestTemplates:
    """Tests for GET /api/v1/test-templates endpoint."""

    @pytest.mark.asyncio
    async def test_list_templates_with_data(
        self, client: AsyncClient, sample_template: TestTemplate, builtin_template: TestTemplate
    ) -> None:
        """Test listing templates including builtin and custom."""
        response = await client.get("/api/v1/test-templates")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2

        # Verify both are present
        names = [item["name"] for item in data["items"]]
        assert "Custom Template" in names
        assert "Builtin Template" in names


class TestCreateTestTemplate:
    """Tests for POST /api/v1/test-templates endpoint."""

    @pytest.mark.asyncio
    async def test_create_template_success(self, client: AsyncClient) -> None:
        """Test creating a custom template."""
        payload = {
            "name": "New Custom Template",
            "question_template": "How to {action}?",
            "category": "procedural",
        }

        response = await client.post("/api/v1/test-templates", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Custom Template"
        assert data["is_builtin"] is False


class TestGetTestTemplate:
    """Tests for GET /api/v1/test-templates/{template_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_template_success(
        self, client: AsyncClient, sample_template: TestTemplate
    ) -> None:
        """Test getting template details."""
        response = await client.get(f"/api/v1/test-templates/{sample_template.id}")

        assert response.status_code == 200
        assert response.json()["name"] == "Custom Template"


class TestUpdateTestTemplate:
    """Tests for PUT /api/v1/test-templates/{template_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_custom_template_success(
        self, client: AsyncClient, sample_template: TestTemplate
    ) -> None:
        """Test updating a custom template."""
        payload = {"name": "Updated Custom Name"}
        response = await client.put(f"/api/v1/test-templates/{sample_template.id}", json=payload)

        assert response.status_code == 200
        assert response.json()["name"] == "Updated Custom Name"

    @pytest.mark.asyncio
    async def test_update_builtin_template_fails(
        self, client: AsyncClient, builtin_template: TestTemplate
    ) -> None:
        """Test that builtin templates cannot be updated."""
        payload = {"name": "Try to Update Builtin"}
        response = await client.put(f"/api/v1/test-templates/{builtin_template.id}", json=payload)

        assert response.status_code == 403
        assert "builtin templates cannot be modified" in response.json()["detail"].lower()


class TestDeleteTestTemplate:
    """Tests for DELETE /api/v1/test-templates/{template_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_custom_template_success(
        self, client: AsyncClient, sample_template: TestTemplate
    ) -> None:
        """Test deleting a custom template."""
        response = await client.delete(f"/api/v1/test-templates/{sample_template.id}")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_builtin_template_fails(
        self, client: AsyncClient, builtin_template: TestTemplate
    ) -> None:
        """Test that builtin templates cannot be deleted."""
        response = await client.delete(f"/api/v1/test-templates/{builtin_template.id}")

        assert response.status_code == 403
        assert "builtin templates cannot be deleted" in response.json()["detail"].lower()
