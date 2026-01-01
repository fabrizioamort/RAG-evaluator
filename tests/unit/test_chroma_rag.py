"""Tests for ChromaDB Semantic Search RAG."""

from unittest.mock import MagicMock, patch

import pytest

from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings for testing."""
    with patch("rag_evaluator.rag_implementations.vector_semantic.chroma_rag.settings") as mock:
        mock.chroma_persist_directory = "./test_chroma_db"
        mock.openai_api_key = "test-api-key"
        mock.embedding_model = "text-embedding-3-small"
        mock.openai_model = "gpt-4"
        yield mock


@pytest.fixture
def mock_chromadb() -> MagicMock:
    """Mock ChromaDB client."""
    with patch(
        "rag_evaluator.rag_implementations.vector_semantic.chroma_rag.chromadb.PersistentClient"
    ) as mock:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock.return_value = mock_client
        yield mock


@pytest.fixture
def mock_openai() -> MagicMock:
    """Mock OpenAI client."""
    with patch("rag_evaluator.rag_implementations.vector_semantic.chroma_rag.OpenAI") as mock:
        yield mock


def test_chroma_rag_initialization(
    mock_settings: MagicMock, mock_chromadb: MagicMock, mock_openai: MagicMock
) -> None:
    """Test ChromaSemanticRAG initialization."""
    rag = ChromaSemanticRAG(collection_name="test_collection")

    assert rag.name == "ChromaDB Semantic Search"
    assert rag.collection_name == "test_collection"
    assert rag._total_chunks == 0
    assert len(rag._retrieval_times) == 0


def test_get_metrics_empty(
    mock_settings: MagicMock, mock_chromadb: MagicMock, mock_openai: MagicMock
) -> None:
    """Test get_metrics with no queries."""
    rag = ChromaSemanticRAG()
    metrics = rag.get_metrics()

    assert metrics["avg_retrieval_time"] == 0.0
    assert metrics["total_queries"] == 0
    assert "collection_name" in metrics


def test_prepare_documents_invalid_path(
    mock_settings: MagicMock, mock_chromadb: MagicMock, mock_openai: MagicMock
) -> None:
    """Test prepare_documents with invalid path."""
    rag = ChromaSemanticRAG()

    with pytest.raises(ValueError, match="Documents path does not exist"):
        rag.prepare_documents("/nonexistent/path")


def test_query_structure(
    mock_settings: MagicMock, mock_chromadb: MagicMock, mock_openai: MagicMock
) -> None:
    """Test that query returns correct structure."""
    rag = ChromaSemanticRAG()

    # Mock the collection query response
    rag.collection.query.return_value = {
        "documents": [["Test chunk 1", "Test chunk 2"]],
        "metadatas": [[{"source": "test.txt"}, {"source": "test.txt"}]],
    }

    # Mock OpenAI embedding
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    rag.openai_client.embeddings.create.return_value = mock_embedding_response

    # Mock OpenAI chat completion
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    rag.openai_client.chat.completions.create.return_value = mock_chat_response

    result = rag.query("What is RAG?")

    assert "answer" in result
    assert "context" in result
    assert "metadata" in result
    assert isinstance(result["context"], list)
    assert "retrieval_time" in result["metadata"]
    assert "chunks_retrieved" in result["metadata"]
