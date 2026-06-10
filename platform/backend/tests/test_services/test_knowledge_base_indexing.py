"""Tests for knowledge base indexing via IndexBuildService."""

from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index_checkpoint import (
    KnowledgeBaseIndexChunk,
    KnowledgeBaseIndexDocument,
)
from app.models.knowledge_base_version import KnowledgeBaseVersion
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.services.index_build_service import IndexBuildService
from app.services.job_event_log import JobEventLog


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
        llm_model="gpt-5.5",
        llm_reasoning_effort="high",
    )
    db_session.add(config)
    await db_session.commit()
    await db_session.refresh(config)
    return config


@pytest.fixture
def mock_event_log() -> JobEventLog:
    """Create a mock event log."""
    event_log = MagicMock(spec=JobEventLog)
    event_log.log_event = AsyncMock()
    return event_log


@pytest.mark.asyncio
async def test_create_index_success(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test successful creation of an index record."""
    service = IndexBuildService(db_session, mock_event_log)

    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
        name="Test Index",
        description="A test index",
    )

    assert index is not None
    assert index.status == "pending"
    assert index.knowledge_base_id == sample_kb.id
    assert index.rag_config_id == sample_rag_config.id
    assert index.name == "Test Index"
    assert index.document_count == 1
    assert index.physical_id.startswith("idx_")
    assert index.storage_type == "chroma"  # vector_semantic -> chroma
    assert "rag_type" in index.config_snapshot
    assert index.config_snapshot["llm_model"] == "gpt-5.5"
    assert index.config_snapshot["llm_reasoning_effort"] == "high"
    assert index.config_snapshot["embedding_model"] == "text-embedding-3-small"
    assert "build_parameters" in index.config_snapshot
    assert "query_default_parameters" in index.config_snapshot


@pytest.mark.asyncio
async def test_create_index_empty_kb_fails(
    sample_kb: KnowledgeBase,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test that creating index for empty KB fails."""
    service = IndexBuildService(db_session, mock_event_log)

    with pytest.raises(ValueError, match="Cannot index empty knowledge base"):
        await service.create_index(
            kb_id=sample_kb.id,
            rag_config_id=sample_rag_config.id,
        )


@pytest.mark.asyncio
async def test_create_index_kb_not_found(
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test that creating index for non-existent KB fails."""
    from uuid import uuid4

    service = IndexBuildService(db_session, mock_event_log)

    with pytest.raises(ValueError, match="not found"):
        await service.create_index(
            kb_id=uuid4(),
            rag_config_id=sample_rag_config.id,
        )


@pytest.mark.asyncio
async def test_build_index_success(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test successful build of an index."""
    # Create the index record first
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )

    # Mock the RAG adapter
    mock_rag = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.create_rag_for_index_build.return_value = mock_rag
    mock_adapter.prepare_documents = AsyncMock(return_value={"chunk_count": 10})

    with patch.object(service, "rag_adapter", mock_adapter):
        await service.build_index(index.id)

    # Refresh and verify
    await db_session.refresh(index)
    assert index.status == "ready"
    assert index.chunk_count == 10
    assert index.build_completed_at is not None
    assert index.build_duration_seconds is not None

    # Verify event log was called
    # Verify event log was called
    cast(AsyncMock, mock_event_log.log_event).assert_called()


@pytest.mark.asyncio
async def test_build_index_failure(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test index build failure handling."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )

    # Mock the RAG adapter to fail
    mock_rag = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.create_rag_for_index_build.return_value = mock_rag
    mock_adapter.prepare_documents = AsyncMock(side_effect=Exception("Indexing failed"))

    with patch.object(service, "rag_adapter", mock_adapter):
        await service.build_index(index.id)

    # Refresh and verify failure state
    await db_session.refresh(index)
    assert index.status == "failed"
    assert index.error_message == "Indexing failed"
    assert index.build_completed_at is not None


@pytest.mark.asyncio
async def test_archive_index(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test archiving an index."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )

    archived = await service.archive_index(index.id)

    assert archived.status == "archived"


@pytest.mark.asyncio
async def test_delete_index(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Test deleting an index."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )
    index_id = index.id

    await service.delete_index(index_id)

    # Verify deletion
    result = await service.get_index(index_id)
    assert result is None


@pytest.mark.asyncio
async def test_retry_build_preserves_checkpoints(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Normal retry should resume without clearing durable progress."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )
    index.status = "failed"
    index.chunk_count = 3

    doc_checkpoint = KnowledgeBaseIndexDocument(
        index_id=index.id,
        doc_key="doc_1",
        source_path="./storage/documents/test_doc.txt",
        checksum="abc123def456",
        status="completed",
        chunk_count=1,
        completed_chunks=1,
    )
    db_session.add(doc_checkpoint)
    await db_session.flush()
    db_session.add(
        KnowledgeBaseIndexChunk(
            index_id=index.id,
            document_id=doc_checkpoint.id,
            doc_key="doc_1",
            chunk_hash="chunkhash",
            storage_id="chunk_1",
            chunk_index=0,
            status="completed",
        )
    )
    await db_session.commit()

    retried = await service.retry_build(index.id)

    assert retried.status == "pending"
    assert retried.chunk_count == 3
    docs = (
        await db_session.execute(
            select(KnowledgeBaseIndexDocument).where(
                KnowledgeBaseIndexDocument.index_id == index.id
            )
        )
    ).scalars().all()
    chunks = (
        await db_session.execute(
            select(KnowledgeBaseIndexChunk).where(KnowledgeBaseIndexChunk.index_id == index.id)
        )
    ).scalars().all()
    assert len(docs) == 1
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_retry_build_force_clears_storage_and_checkpoints(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Forced retry performs the old clean rebuild behavior."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )
    index.status = "failed"
    index.chunk_count = 3

    doc_checkpoint = KnowledgeBaseIndexDocument(
        index_id=index.id,
        doc_key="doc_1",
        source_path="./storage/documents/test_doc.txt",
        checksum="abc123def456",
        status="completed",
    )
    db_session.add(doc_checkpoint)
    await db_session.flush()
    db_session.add(
        KnowledgeBaseIndexChunk(
            index_id=index.id,
            document_id=doc_checkpoint.id,
            doc_key="doc_1",
            chunk_hash="chunkhash",
            storage_id="chunk_1",
            chunk_index=0,
            status="completed",
        )
    )
    await db_session.commit()

    with patch.object(service, "_cleanup_storage", new=AsyncMock()) as cleanup:
        retried = await service.retry_build(index.id, force=True)

    cleanup.assert_awaited_once()
    assert retried.status == "pending"
    assert retried.chunk_count == 0
    doc_count = await db_session.scalar(
        select(KnowledgeBaseIndexDocument).where(
            KnowledgeBaseIndexDocument.index_id == index.id
        )
    )
    chunk_count = await db_session.scalar(
        select(KnowledgeBaseIndexChunk).where(KnowledgeBaseIndexChunk.index_id == index.id)
    )
    assert doc_count is None
    assert chunk_count is None


@pytest.mark.asyncio
async def test_reconcile_interrupted_builds_marks_stale_build_failed(
    sample_kb: KnowledgeBase,
    sample_document: Document,
    sample_rag_config: RAGConfig,
    db_session: AsyncSession,
    mock_event_log: JobEventLog,
) -> None:
    """Startup reconciliation should make stale building indexes retryable."""
    service = IndexBuildService(db_session, mock_event_log)
    index = await service.create_index(
        kb_id=sample_kb.id,
        rag_config_id=sample_rag_config.id,
    )
    index.status = "building"
    index.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=2)
    await db_session.commit()

    count = await service.reconcile_interrupted_builds(stale_after=timedelta(minutes=30))
    await db_session.refresh(index)

    assert count == 1
    assert index.status == "failed"
    assert "interrupted" in (index.error_message or "")
