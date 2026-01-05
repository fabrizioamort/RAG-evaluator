"""Integration tests for Hybrid Search RAG with Qdrant.

These tests require:
1. Qdrant running locally (docker compose up -d)
2. Valid OpenAI API key in environment
"""

import os
import tempfile
from pathlib import Path

import pytest

# Skip all tests if SKIP_INTEGRATION_TESTS is set
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests disabled via SKIP_INTEGRATION_TESTS environment variable",
)


def is_qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def temp_docs_dir() -> Path:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_path = Path(tmpdir)

        # Create test documents
        (docs_path / "test1.txt").write_text(
            """Retrieval Augmented Generation (RAG) is a technique that combines
            information retrieval with text generation. RAG systems retrieve relevant
            documents from a knowledge base and use them to generate accurate responses.
            This approach helps reduce hallucination in large language models."""
        )

        (docs_path / "test2.txt").write_text(
            """Hybrid search combines semantic search with keyword search for better
            retrieval results. Semantic search uses dense vector embeddings to find
            conceptually similar content, while keyword search uses sparse vectors
            (like BM25 or SPLADE) to match specific terms. The combination provides
            both semantic understanding and precise term matching."""
        )

        yield docs_path


@pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
@pytest.mark.skipif(not has_openai_key(), reason="OpenAI API key not available")
class TestHybridRAGIntegration:
    """Integration tests for HybridSearchRAG."""

    def test_full_workflow(self, temp_docs_dir: Path) -> None:
        """Test the full workflow: prepare, query, metrics."""
        from rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag import HybridSearchRAG

        # Use a unique collection name to avoid conflicts
        import uuid

        collection_name = f"test_hybrid_{uuid.uuid4().hex[:8]}"

        try:
            # Initialize RAG
            rag = HybridSearchRAG(collection_name=collection_name)

            # Prepare documents
            rag.prepare_documents(str(temp_docs_dir))

            # Verify metrics after indexing
            metrics = rag.get_metrics()
            assert metrics["total_chunks"] > 0

            # Test semantic query
            result = rag.query("What is RAG?")

            assert "answer" in result
            assert "context" in result
            assert "metadata" in result
            assert len(result["context"]) > 0
            assert result["metadata"]["fusion_method"] == "RRF"

            # Test keyword-specific query (should benefit from sparse search)
            result = rag.query("What is SPLADE?")

            assert "answer" in result
            assert len(result["context"]) > 0

            # Verify metrics after queries
            metrics = rag.get_metrics()
            assert metrics["total_queries"] == 2
            assert metrics["avg_retrieval_time"] > 0

        finally:
            # Cleanup: delete the test collection
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(url="http://localhost:6333")
                client.delete_collection(collection_name)
            except Exception:
                pass

    def test_empty_results_handling(self) -> None:
        """Test handling of queries when collection is empty."""
        from rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag import HybridSearchRAG

        import uuid

        collection_name = f"test_empty_{uuid.uuid4().hex[:8]}"

        try:
            rag = HybridSearchRAG(collection_name=collection_name)

            # Query empty collection
            result = rag.query("What is RAG?")

            # Should return empty context but not crash
            assert "answer" in result
            assert "context" in result
            assert isinstance(result["context"], list)

        finally:
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(url="http://localhost:6333")
                client.delete_collection(collection_name)
            except Exception:
                pass

    def test_metrics_consistency(self, temp_docs_dir: Path) -> None:
        """Test that metrics are consistent and updated correctly."""
        from rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag import HybridSearchRAG

        import uuid

        collection_name = f"test_metrics_{uuid.uuid4().hex[:8]}"

        try:
            rag = HybridSearchRAG(collection_name=collection_name)

            # Initial metrics
            metrics = rag.get_metrics()
            assert metrics["total_queries"] == 0
            assert metrics["avg_retrieval_time"] == 0.0

            # Prepare documents
            rag.prepare_documents(str(temp_docs_dir))

            # Run multiple queries
            for _ in range(3):
                rag.query("What is hybrid search?")

            metrics = rag.get_metrics()
            assert metrics["total_queries"] == 3
            assert metrics["avg_retrieval_time"] > 0
            assert metrics["chunk_size"] == 700
            assert metrics["chunk_overlap"] == 100

        finally:
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(url="http://localhost:6333")
                client.delete_collection(collection_name)
            except Exception:
                pass
