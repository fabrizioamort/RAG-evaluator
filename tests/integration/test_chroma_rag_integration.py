"""Integration tests for ChromaDB Semantic Search RAG."""

import shutil
from pathlib import Path

import pytest

from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


@pytest.fixture
def test_chroma_db_path(tmp_path: Path) -> Path:
    """Create a temporary ChromaDB directory."""
    db_path = tmp_path / "test_chroma_db"
    db_path.mkdir()
    yield db_path
    # Cleanup - ignore errors on Windows where ChromaDB may hold file locks
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
        except (PermissionError, OSError):
            # ChromaDB on Windows may keep files locked - ignore cleanup errors
            pass


@pytest.fixture
def sample_documents_dir() -> Path:
    """Get path to sample documents."""
    return Path("data/raw")


@pytest.mark.skipif(
    not settings.openai_api_key or settings.openai_api_key == "",
    reason="OpenAI API key not configured",
)
class TestChromaRAGIntegration:
    """Integration tests for ChromaDB RAG (requires OpenAI API key)."""

    def test_end_to_end_workflow(
        self, test_chroma_db_path: Path, sample_documents_dir: Path
    ) -> None:
        """Test complete workflow: prepare documents and query."""
        # Skip if no sample documents
        if not sample_documents_dir.exists() or not list(sample_documents_dir.glob("*.txt")):
            pytest.skip("No sample documents found in data/raw")

        # Temporarily override chroma directory
        original_dir = settings.chroma_persist_directory
        settings.chroma_persist_directory = str(test_chroma_db_path)

        try:
            # Initialize RAG
            rag = ChromaSemanticRAG(collection_name="test_integration")

            # Prepare documents
            rag.prepare_documents(str(sample_documents_dir))

            # Verify documents were indexed
            metrics = rag.get_metrics()
            assert metrics["total_chunks"] > 0, "No chunks were indexed"

            # Query the system
            result = rag.query("What is RAG?", top_k=3)

            # Verify result structure
            assert "answer" in result
            assert "context" in result
            assert "metadata" in result

            # Verify we got an answer
            assert len(result["answer"]) > 0
            assert result["answer"] != "No answer generated"

            # Verify we retrieved context
            assert len(result["context"]) > 0
            assert result["metadata"]["chunks_retrieved"] > 0

            # Verify metrics were updated
            updated_metrics = rag.get_metrics()
            assert updated_metrics["total_queries"] == 1
            assert updated_metrics["avg_retrieval_time"] > 0

        finally:
            # Restore original setting
            settings.chroma_persist_directory = original_dir

    def test_query_without_documents(self, test_chroma_db_path: Path) -> None:
        """Test querying an empty collection."""
        original_dir = settings.chroma_persist_directory
        settings.chroma_persist_directory = str(test_chroma_db_path)

        try:
            rag = ChromaSemanticRAG(collection_name="test_empty")

            # Query without preparing documents
            result = rag.query("What is RAG?")

            # Should still return valid structure
            assert "answer" in result
            assert "context" in result
            assert result["metadata"]["chunks_retrieved"] == 0

        finally:
            settings.chroma_persist_directory = original_dir
