"""Tests for knowledge bases API endpoints."""

import io
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_version import KnowledgeBaseVersion
from app.models.project import Project
from app.models.rag_config import RAGConfig


@pytest.fixture
async def sample_project(db_session: AsyncSession) -> Project:
    """Create a sample project for testing."""
    project = Project(
        name="Test Project",
        description="A test project for KB tests",
        status="active",
        tags=["test"],
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def sample_kb(db_session: AsyncSession, sample_project: Project) -> KnowledgeBase:
    """Create a sample knowledge base for testing."""
    kb = KnowledgeBase(
        project_id=sample_project.id,
        name="Test Knowledge Base",
        description="A test KB for unit tests",
        status="pending",
        current_version=1,
        storage_path="./storage/documents",
        # index_path removed/deprecated
        metadata_={"source": "test"},
    )
    db_session.add(kb)
    await db_session.commit()
    await db_session.refresh(kb)

    # Create initial version
    version = KnowledgeBaseVersion(
        knowledge_base_id=kb.id,
        version_number=1,
        change_type="initial",
        document_snapshot=[],
        change_description="Initial version",
    )
    db_session.add(version)
    await db_session.commit()

    return kb


@pytest.fixture
async def sample_document(db_session: AsyncSession, sample_kb: KnowledgeBase) -> Document:
    """Create a sample document for testing."""
    doc = Document(
        knowledge_base_id=sample_kb.id,
        filename="test_doc.txt",
        file_path="./storage/documents/test_doc.txt",
        content_type="text/plain",
        size_bytes=1024,
        checksum="abc123def456",
        status="uploaded",
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    return doc


@pytest.fixture
async def sample_rag_config(db_session: AsyncSession, sample_project: Project) -> RAGConfig:
    """Create a sample RAG configuration for testing."""
    config = RAGConfig(
        project_id=sample_project.id,
        name="Test RAG Config",
        rag_type="vector_semantic",
        parameters={"collection_name": "test_collection"},
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)
    return config


class TestListKnowledgeBases:
    """Tests for GET /api/v1/projects/{project_id}/knowledge-bases endpoint."""

    @pytest.mark.asyncio
    async def test_list_kbs_empty(self, client: AsyncClient, sample_project: Project) -> None:
        """Test listing KBs when none exist."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}/knowledge-bases")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["offset"] == 0
        assert data["limit"] == 20

    @pytest.mark.asyncio
    async def test_list_kbs_with_data(
        self, client: AsyncClient, sample_project: Project, sample_kb: KnowledgeBase
    ) -> None:
        """Test listing KBs with existing data."""
        response = await client.get(f"/api/v1/projects/{sample_project.id}/knowledge-bases")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Test Knowledge Base"
        assert data["items"][0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_kbs_pagination(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test pagination works correctly."""
        # Note: We need multiple KBs, but simplified for now
        pass

    @pytest.mark.asyncio
    async def test_list_kbs_project_not_found(self, client: AsyncClient) -> None:
        """Test listing KBs for non-existent project."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/projects/{fake_id}/knowledge-bases")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestCreateKnowledgeBase:
    """Tests for POST /api/v1/projects/{project_id}/knowledge-bases endpoint."""

    @pytest.mark.asyncio
    async def test_create_kb_success(self, client: AsyncClient, sample_project: Project) -> None:
        """Test creating a KB successfully."""
        payload = {
            "name": "New Knowledge Base",
            "description": "A newly created KB",
            "metadata": {"source": "api_test"},
        }

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Knowledge Base"
        assert data["description"] == "A newly created KB"
        assert data["status"] == "pending"
        assert data["current_version"] == 1  # Initial version created
        assert data["document_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert data["project_id"] == str(sample_project.id)

    @pytest.mark.asyncio
    async def test_create_kb_minimal(self, client: AsyncClient, sample_project: Project) -> None:
        """Test creating a KB with minimal data."""
        payload = {"name": "Minimal KB"}

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases", json=payload
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal KB"
        assert data["description"] is None
        assert data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_create_kb_empty_name(self, client: AsyncClient, sample_project: Project) -> None:
        """Test that empty name fails validation."""
        payload = {"name": ""}

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases", json=payload
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_kb_missing_name(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test that missing name fails validation."""
        payload = {"description": "No name provided"}

        response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases", json=payload
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_kb_project_not_found(self, client: AsyncClient) -> None:
        """Test creating KB for non-existent project."""
        fake_id = uuid4()
        payload = {"name": "Test KB"}

        response = await client.post(f"/api/v1/projects/{fake_id}/knowledge-bases", json=payload)

        assert response.status_code == 404


class TestGetKnowledgeBase:
    """Tests for GET /api/v1/knowledge-bases/{kb_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_kb_success(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test getting a KB by ID."""
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_kb.id)
        assert data["name"] == "Test Knowledge Base"
        assert data["description"] == "A test KB for unit tests"
        assert data["status"] == "pending"
        assert "documents" in data
        assert isinstance(data["documents"], list)

    @pytest.mark.asyncio
    async def test_get_kb_with_documents(
        self, client: AsyncClient, sample_kb: KnowledgeBase, sample_document: Document
    ) -> None:
        """Test getting a KB includes its documents."""
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "test_doc.txt"
        assert data["document_count"] == 1

    @pytest.mark.asyncio
    async def test_get_kb_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent KB."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/knowledge-bases/{fake_id}")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_kb_invalid_uuid(self, client: AsyncClient) -> None:
        """Test getting a KB with invalid UUID format."""
        response = await client.get("/api/v1/knowledge-bases/not-a-uuid")

        assert response.status_code == 422  # Validation error


class TestUpdateKnowledgeBase:
    """Tests for PUT /api/v1/knowledge-bases/{kb_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_kb_success(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test updating a KB successfully."""
        payload = {
            "name": "Updated KB Name",
            "description": "Updated description",
        }

        response = await client.put(f"/api/v1/knowledge-bases/{sample_kb.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated KB Name"
        assert data["description"] == "Updated description"
        # Status should remain unchanged
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_kb_partial(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test partial update with only some fields."""
        payload = {"name": "Only Name Updated"}

        response = await client.put(f"/api/v1/knowledge-bases/{sample_kb.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Only Name Updated"
        # Other fields should remain unchanged
        assert data["description"] == "A test KB for unit tests"

    @pytest.mark.asyncio
    async def test_update_kb_metadata(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test updating KB metadata."""
        payload = {"metadata": {"new_key": "new_value"}}

        response = await client.put(f"/api/v1/knowledge-bases/{sample_kb.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"] == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_update_kb_not_found(self, client: AsyncClient) -> None:
        """Test updating a non-existent KB."""
        fake_id = uuid4()
        payload = {"name": "Updated"}

        response = await client.put(f"/api/v1/knowledge-bases/{fake_id}", json=payload)

        assert response.status_code == 404


class TestDeleteKnowledgeBase:
    """Tests for DELETE /api/v1/knowledge-bases/{kb_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_kb_success(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test deleting a KB successfully."""
        kb_id = sample_kb.id

        response = await client.delete(f"/api/v1/knowledge-bases/{kb_id}")

        assert response.status_code == 204

        # Verify KB is actually deleted
        get_response = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_kb_not_found(self, client: AsyncClient) -> None:
        """Test deleting a non-existent KB."""
        fake_id = uuid4()

        response = await client.delete(f"/api/v1/knowledge-bases/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_kb_invalid_uuid(self, client: AsyncClient) -> None:
        """Test deleting with invalid UUID format."""
        response = await client.delete("/api/v1/knowledge-bases/not-a-uuid")

        assert response.status_code == 422


class TestDocumentUpload:
    """Tests for POST /api/v1/knowledge-bases/{kb_id}/documents endpoint."""

    @pytest.mark.asyncio
    async def test_upload_single_document(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test uploading a single document."""
        # Create a test file
        file_content = b"This is test content for the document."
        files = {"files": ("test_file.txt", io.BytesIO(file_content), "text/plain")}

        response = await client.post(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["uploaded"]) == 1
        assert data["failed"] == []
        assert data["total_size_bytes"] == len(file_content)
        assert data["uploaded"][0]["filename"] == "test_file.txt"
        assert data["uploaded"][0]["content_type"] == "text/plain"
        assert data["uploaded"][0]["checksum"] is not None

    @pytest.mark.asyncio
    async def test_upload_multiple_documents(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test uploading multiple documents."""
        files = [
            ("files", ("file1.txt", io.BytesIO(b"Content 1"), "text/plain")),
            ("files", ("file2.txt", io.BytesIO(b"Content 2 longer"), "text/plain")),
        ]

        response = await client.post(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["uploaded"]) == 2
        assert data["failed"] == []

    @pytest.mark.asyncio
    async def test_upload_creates_version(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test that uploading creates a new KB version."""
        files = {"files": ("test.txt", io.BytesIO(b"Test content"), "text/plain")}
        await client.post(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents",
            files=files,
        )

        # Check versions
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}/versions")
        assert response.status_code == 200
        versions = response.json()
        # Should have at least initial + new version
        assert len(versions) >= 1
        # Find the documents_added version
        added_versions = [v for v in versions if v["change_type"] == "documents_added"]
        assert len(added_versions) >= 1

    @pytest.mark.asyncio
    async def test_upload_empty_file_fails(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test that empty file upload is reported as failed."""
        files = {"files": ("empty.txt", io.BytesIO(b""), "text/plain")}

        response = await client.post(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents",
            files=files,
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data["uploaded"]) == 0
        assert len(data["failed"]) == 1
        assert data["failed"][0]["filename"] == "empty.txt"
        assert "empty" in data["failed"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_upload_kb_not_found(self, client: AsyncClient) -> None:
        """Test uploading to non-existent KB."""
        fake_id = uuid4()
        files = {"files": ("test.txt", io.BytesIO(b"Test"), "text/plain")}

        response = await client.post(
            f"/api/v1/knowledge-bases/{fake_id}/documents",
            files=files,
        )

        assert response.status_code == 404


class TestDeleteDocument:
    """Tests for DELETE /api/v1/knowledge-bases/{kb_id}/documents/{doc_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self,
        client: AsyncClient,
        sample_kb: KnowledgeBase,
        sample_document: Document,
    ) -> None:
        """Test deleting a document successfully."""
        response = await client.delete(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents/{sample_document.id}"
        )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test deleting a non-existent document."""
        fake_doc_id = uuid4()

        response = await client.delete(
            f"/api/v1/knowledge-bases/{sample_kb.id}/documents/{fake_doc_id}"
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_wrong_kb(
        self,
        client: AsyncClient,
        sample_project: Project,
        sample_kb: KnowledgeBase,
        sample_document: Document,
        db_session: AsyncSession,
    ) -> None:
        """Test deleting a document from wrong KB returns 404."""
        # Create another KB
        other_kb = KnowledgeBase(
            project_id=sample_project.id,
            name="Other KB",
            status="pending",
            current_version=0,
        )
        db_session.add(other_kb)
        await db_session.commit()
        await db_session.refresh(other_kb)

        # Try to delete document using wrong KB's ID
        response = await client.delete(
            f"/api/v1/knowledge-bases/{other_kb.id}/documents/{sample_document.id}"
        )

        assert response.status_code == 404


class TestListKBVersions:
    """Tests for GET /api/v1/knowledge-bases/{kb_id}/versions endpoint."""

    @pytest.mark.asyncio
    async def test_list_versions_success(
        self, client: AsyncClient, sample_kb: KnowledgeBase
    ) -> None:
        """Test listing KB versions."""
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}/versions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # Check version structure
        version = data[0]
        assert "id" in version
        assert "version_number" in version
        assert "change_type" in version
        assert "document_snapshot" in version
        assert "created_at" in version

    @pytest.mark.asyncio
    async def test_list_versions_kb_not_found(self, client: AsyncClient) -> None:
        """Test listing versions for non-existent KB."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/knowledge-bases/{fake_id}/versions")

        assert response.status_code == 404


class TestGetKBStatus:
    """Tests for GET /api/v1/knowledge-bases/{kb_id}/status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_success(self, client: AsyncClient, sample_kb: KnowledgeBase) -> None:
        """Test getting KB status."""
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_kb.id)
        assert data["status"] == "pending"
        assert data["current_version"] == sample_kb.current_version
        assert "document_count" in data
        assert "total_size_bytes" in data

    @pytest.mark.asyncio
    async def test_get_status_with_documents(
        self,
        client: AsyncClient,
        sample_kb: KnowledgeBase,
        sample_document: Document,
    ) -> None:
        """Test getting KB status includes document stats."""
        response = await client.get(f"/api/v1/knowledge-bases/{sample_kb.id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 1
        assert data["total_size_bytes"] == 1024  # From sample_document fixture

    @pytest.mark.asyncio
    async def test_get_status_kb_not_found(self, client: AsyncClient) -> None:
        """Test getting status for non-existent KB."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/knowledge-bases/{fake_id}/status")

        assert response.status_code == 404


class TestKBVersioning:
    """Tests for KB versioning functionality."""

    @pytest.mark.asyncio
    async def test_version_increments_on_document_add(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test that version increments when documents are added."""
        # Create a KB
        create_response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases",
            json={"name": "Version Test KB"},
        )
        assert create_response.status_code == 201
        kb_data = create_response.json()
        kb_id = kb_data["id"]
        initial_version = kb_data["current_version"]

        # Upload a document
        files = {"files": ("doc.txt", io.BytesIO(b"Content"), "text/plain")}
        await client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files=files)

        # Check version increased
        status_response = await client.get(f"/api/v1/knowledge-bases/{kb_id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["current_version"] > initial_version

    @pytest.mark.asyncio
    async def test_version_snapshot_contains_documents(
        self, client: AsyncClient, sample_project: Project
    ) -> None:
        """Test that version snapshot contains document information."""
        # Create KB and upload document
        create_response = await client.post(
            f"/api/v1/projects/{sample_project.id}/knowledge-bases",
            json={"name": "Snapshot Test KB"},
        )
        kb_id = create_response.json()["id"]

        files = {"files": ("snapshot_doc.txt", io.BytesIO(b"Snapshot content"), "text/plain")}
        await client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files=files)

        # Get versions and check snapshot
        versions_response = await client.get(f"/api/v1/knowledge-bases/{kb_id}/versions")
        assert versions_response.status_code == 200
        versions = versions_response.json()

        # Find the documents_added version
        doc_added_version = next(
            (v for v in versions if v["change_type"] == "documents_added"), None
        )
        assert doc_added_version is not None
        assert len(doc_added_version["document_snapshot"]) == 1
        assert doc_added_version["document_snapshot"][0]["filename"] == "snapshot_doc.txt"


class TestIndexCreation:
    """Tests for POST /api/v1/knowledge-bases/{kb_id}/indexes endpoint."""

    @pytest.mark.asyncio
    async def test_create_index_success(
        self,
        client: AsyncClient,
        sample_kb: KnowledgeBase,
        sample_document: Document,
        sample_rag_config: RAGConfig,
    ) -> None:
        """Test creating an index successfully."""
        from app.services.rag_adapter import get_rag_adapter_service
        
        mock_adapter = MagicMock()
        mock_adapter.create_rag_for_index.return_value = MagicMock()
        mock_adapter.prepare_documents = AsyncMock(return_value={"chunk_count": 10})
        
        from app.main import app
        app.dependency_overrides[get_rag_adapter_service] = lambda: mock_adapter
        
        try:
            response = await client.post(
                f"/api/v1/knowledge-bases/{sample_kb.id}/indexes",
                json={"rag_config_id": str(sample_rag_config.id)}
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "pending"
            assert data["knowledge_base_id"] == str(sample_kb.id)
            assert data["rag_config_id"] == str(sample_rag_config.id)
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_create_index_empty_kb_fails(
        self, client: AsyncClient, sample_kb: KnowledgeBase, sample_rag_config: RAGConfig
    ) -> None:
        """Test that indexing an empty KB fails."""
        # sample_kb has no documents in this test context
        
        response = await client.post(
            f"/api/v1/knowledge-bases/{sample_kb.id}/indexes",
            json={"rag_config_id": str(sample_rag_config.id)}
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
