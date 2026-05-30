"""Tests for RAG configurations API endpoints."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from rag_evaluator.rag_implementations.registry import get_parameter_schema
from sqlalchemy.ext.asyncio import AsyncSession

import app.api.rag_configs as rag_configs_api
from app.models.project import Project
from app.models.rag_config import RAGConfig


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
async def sample_rag_config(db_session: AsyncSession, sample_project: Project) -> RAGConfig:
    """Create a sample RAG config for testing."""
    config = RAGConfig(
        project_id=sample_project.id,
        name="Semantic Search Default",
        rag_type="vector_semantic",
        parameters={"collection_name": "test_collection"},
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)
    return config


class TestDiscoveryEndpoints:
    """Tests for metadata discovery endpoints."""

    @pytest.mark.asyncio
    async def test_list_rag_types(self, client: AsyncClient) -> None:
        """Test listing available RAG types."""
        response = await client.get("/api/v1/rag-types")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify structure of one item
        rag_type = data[0]
        assert "name" in rag_type
        assert "display_name" in rag_type
        assert "parameters" in rag_type
        assert isinstance(rag_type["parameters"], list)

        rlm_type = next(t for t in data if t["name"] == "rlm_rag")
        assert rlm_type["display_name"] == "RLM-RAG"
        rlm_params = {p["name"]: p for p in rlm_type["parameters"]}
        assert rlm_params["security_mode"]["choices"] == ["lite", "full"]
        assert rlm_params["security_mode"]["phase"] == "query"
        assert rlm_params["worker_model"]["phase"] == "build"
        assert rlm_params["max_repl_steps"]["default"] == 15
        assert rlm_params["max_repl_steps"]["min_value"] == 1
        assert rlm_params["max_repl_steps"]["max_value"] == 50

    @pytest.mark.asyncio
    async def test_rag_types_phase_data_matches_core_registry(self, client: AsyncClient) -> None:
        """The API should expose phase metadata from the canonical core registry."""
        response = await client.get("/api/v1/rag-types")

        assert response.status_code == 200
        data = response.json()
        for rag_type in data:
            core = get_parameter_schema(rag_type["name"])["properties"]
            for parameter in rag_type["parameters"]:
                assert parameter["phase"] == core[parameter["name"]]["phase"]

    @pytest.mark.asyncio
    async def test_get_rag_type_parameters_success(self, client: AsyncClient) -> None:
        """Test getting parameters for a specific RAG type."""
        response = await client.get("/api/v1/rag-types/vector_semantic/parameters")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Should have collection_name parameter
        param_names = [p["name"] for p in data]
        assert "collection_name" in param_names

    @pytest.mark.asyncio
    async def test_get_rag_type_parameters_not_found(self, client: AsyncClient) -> None:
        """Test getting parameters for a non-existent RAG type."""
        response = await client.get("/api/v1/rag-types/invalid_type/parameters")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_llm_providers(self, client: AsyncClient) -> None:
        """Test listing available LLM providers."""
        response = await client.get("/api/v1/llm-providers")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        provider_names = [p["name"] for p in data]
        assert "openai" in provider_names
        assert "ollama" in provider_names


class TestRAGConfigCRUD:
    """Tests for RAG configuration CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_rag_config_success(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test creating a RAG config successfully."""
        payload = {
            "name": "New Hybrid Config",
            "rag_type": "vector_hybrid",
            "parameters": {"collection_name": "hybrid_docs"},
            "llm_provider": "anthropic",
            "llm_model": "claude-3-haiku-20240307",
        }

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/rag-configs", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Hybrid Config"
        assert data["rag_type"] == "vector_hybrid"
        assert data["project_id"] == str(sample_project.id)
        assert data["llm_provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_create_graph_rag_config_blank_params_use_env_fallback(
        self, client: AsyncClient, sample_project: Project, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank Neo4j params should fallback to backend settings values."""
        monkeypatch.setattr(rag_configs_api.settings, "NEO4J_URI", "bolt://env-host:7687")
        monkeypatch.setattr(rag_configs_api.settings, "NEO4J_USERNAME", "env-user")
        monkeypatch.setattr(rag_configs_api.settings, "NEO4J_PASSWORD", "env-pass")

        payload = {
            "name": "Graph Config",
            "rag_type": "graph_rag",
            "parameters": {
                "neo4j_uri": "   ",
                "neo4j_username": "",
                "neo4j_password": "   ",
            },
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        }

        with patch("app.api.rag_configs._test_neo4j_connection") as mock_connection_test:
            response = await client.post(
                f"/api/v1/projects/{sample_project.id}/rag-configs", json=payload
            )

        assert response.status_code == 201
        mock_connection_test.assert_called_once_with("bolt://env-host:7687", "env-user", "env-pass")

    @pytest.mark.asyncio
    async def test_create_rag_config_invalid_project(self, client: AsyncClient) -> None:
        """Test creating a RAG config for a non-existent project."""
        fake_id = uuid4()
        payload = {
            "name": "Should Fail",
            "rag_type": "vector_semantic",
        }

        response = await client.post(f"/api/v1/projects/{fake_id}/rag-configs", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_rag_config_invalid_type(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test creating a RAG config with an invalid type."""
        payload = {
            "name": "Invalid Type Config",
            "rag_type": "invalid_rag_type",
        }

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/rag-configs", json=payload
        )

        assert response.status_code == 400
        assert "invalid rag type" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_rag_configs(
        self, client: AsyncClient, sample_project: Project, sample_rag_config: RAGConfig
    ) -> None:
        """Test listing RAG configs for a project."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}/rag-configs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == sample_rag_config.name

    @pytest.mark.asyncio
    async def test_get_rag_config_success(
        self, client: AsyncClient, sample_rag_config: RAGConfig
    ) -> None:
        """Test getting a specific RAG config by ID."""
        response = await client.get(f"/api/v1/rag-configs/{sample_rag_config.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_rag_config.id)
        assert data["name"] == sample_rag_config.name

    @pytest.mark.asyncio
    async def test_get_rag_config_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent RAG config."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/rag-configs/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_rag_config_success(
        self, client: AsyncClient, sample_rag_config: RAGConfig
    ) -> None:
        """Test updating a RAG config."""
        payload = {
            "name": "Updated Name",
            "llm_model": "gpt-4o",
        }

        response = await client.put(f"/api/v1/rag-configs/{sample_rag_config.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["llm_model"] == "gpt-4o"
        # Others should be unchanged
        assert data["rag_type"] == sample_rag_config.rag_type

    @pytest.mark.asyncio
    async def test_update_graph_rag_config_validates_neo4j_connection(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_project: Project,
    ) -> None:
        """Updating graph_rag connection params should validate connectivity."""
        graph_config = RAGConfig(
            project_id=sample_project.id,
            name="Graph Config",
            rag_type="graph_rag",
            parameters={},
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        db_session.add(graph_config)
        await db_session.commit()
        await db_session.refresh(graph_config)

        payload = {
            "parameters": {
                "neo4j_uri": "bolt://bad-host:7687",
                "neo4j_username": "neo4j",
                "neo4j_password": "wrong-password",
            }
        }

        with patch(
            "app.api.rag_configs._test_neo4j_connection",
            side_effect=RuntimeError("connection refused"),
        ) as mock_connection_test:
            response = await client.put(f"/api/v1/rag-configs/{graph_config.id}", json=payload)

        assert response.status_code == 422
        assert "cannot connect to neo4j" in response.json()["detail"].lower()
        mock_connection_test.assert_called_once_with(
            "bolt://bad-host:7687", "neo4j", "wrong-password"
        )

    @pytest.mark.asyncio
    async def test_delete_rag_config_success(
        self, client: AsyncClient, sample_rag_config: RAGConfig
    ) -> None:
        """Test deleting a RAG config."""
        response = await client.delete(f"/api/v1/rag-configs/{sample_rag_config.id}")

        assert response.status_code == 204

        # Verify deletion
        get_response = await client.get(f"/api/v1/rag-configs/{sample_rag_config.id}")
        assert get_response.status_code == 404
