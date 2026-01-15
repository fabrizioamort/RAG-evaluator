"""Tests for knowledge base indexing logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.knowledge_bases import _perform_indexing_task
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
        index_path="./storage/indexes",
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


@pytest.mark.asyncio
async def test_perform_indexing_task_success(
    sample_kb: KnowledgeBase,
    db_session: AsyncSession,
) -> None:
    """Test successful execution of indexing task."""
    mock_adapter = MagicMock()
    mock_adapter.get_or_create_rag.return_value = MagicMock()
    mock_adapter.prepare_documents = AsyncMock(return_value={"total_chunks": 10})

    # Mock get_db_context to return our test session
    with patch("app.database.get_db_context") as mock_ctx:
        mock_ctx.return_value.__aenter__.return_value = db_session
        
        await _perform_indexing_task(
            kb_id=sample_kb.id,
            rag_config_id=None,
            rag_adapter=mock_adapter,
        )

    # Verify KB status updated
    await db_session.refresh(sample_kb)
    assert sample_kb.status == "ready"
    assert sample_kb.current_version == 2
    
    # Verify adapter calls
    mock_adapter.get_or_create_rag.assert_called()
    mock_adapter.prepare_documents.assert_called()


@pytest.mark.asyncio
async def test_perform_indexing_task_failure(
    sample_kb: KnowledgeBase,
    db_session: AsyncSession,
) -> None:
    """Test indexing task failure handling."""
    mock_adapter = MagicMock()
    mock_adapter.get_or_create_rag.return_value = MagicMock()
    mock_adapter.prepare_documents = AsyncMock(side_effect=Exception("Indexing failed"))

    # Mock get_db_context to return our test session
    with patch("app.database.get_db_context") as mock_ctx:
        mock_ctx.return_value.__aenter__.return_value = db_session
        
        await _perform_indexing_task(
            kb_id=sample_kb.id,
            rag_config_id=None,
            rag_adapter=mock_adapter,
        )

    # Verify KB status updated to error
    await db_session.refresh(sample_kb)
    assert sample_kb.status == "error"
    # Version should not increment on error
    assert sample_kb.current_version == 1
