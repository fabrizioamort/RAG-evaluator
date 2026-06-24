"""Tests for Hybrid Search RAG with Qdrant."""

from unittest.mock import MagicMock, patch

import pytest

from rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag import HybridSearchRAG


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings for testing."""
    with patch("rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.settings") as mock:
        mock.qdrant_url = "http://localhost:6333"
        mock.qdrant_collection_name = "test_hybrid"
        mock.openai_api_key = "test-api-key"
        mock.embedding_model = "text-embedding-3-small"
        mock.openai_model = "gpt-4"
        mock.openai_timeout = 600
        mock.hybrid_chunk_size = 700
        mock.hybrid_chunk_overlap = 100
        mock.sparse_model_name = "prithvida/Splade_PP_en_v1"
        yield mock


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Mock Qdrant client."""
    with patch("rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.QdrantClient") as mock:
        mock_client = MagicMock()
        # Mock get_collections to return empty list (collection doesn't exist)
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        # Mock get_collection for metrics
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 0
        mock_client.get_collection.return_value = mock_collection_info
        mock.return_value = mock_client
        yield mock


@pytest.fixture
def mock_openai() -> MagicMock:
    """Mock OpenAI-compatible clients."""
    with (
        patch("rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.llm_client") as mock_llm,
        patch(
            "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.embedding_client"
        ) as mock_embedding,
    ):
        mock_llm_client = MagicMock()
        mock_embedding_client = MagicMock()
        mock_llm.return_value = mock_llm_client
        mock_embedding.return_value = mock_embedding_client
        yield mock_llm_client


@pytest.fixture
def mock_fastembed() -> MagicMock:
    """Mock FastEmbed sparse model."""
    with patch(
        "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.SparseTextEmbedding"
    ) as mock:
        mock_model = MagicMock()
        # Create a mock sparse embedding with numpy-like arrays
        mock_sparse = MagicMock()
        mock_sparse.indices = MagicMock()
        mock_sparse.indices.tolist.return_value = [1, 42, 100]
        mock_sparse.values = MagicMock()
        mock_sparse.values.tolist.return_value = [0.22, 0.8, 0.5]
        mock_model.embed.return_value = iter([mock_sparse])
        mock.return_value = mock_model
        yield mock


def test_hybrid_rag_initialization(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test HybridSearchRAG initialization."""
    rag = HybridSearchRAG(collection_name="test_collection")

    assert rag.name == "Hybrid Search (Semantic + Keyword)"
    assert rag.collection_name == "test_collection"
    assert rag._total_chunks == 0
    assert len(rag._retrieval_times) == 0


def test_hybrid_rag_default_collection(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test HybridSearchRAG uses default collection from settings."""
    rag = HybridSearchRAG()

    assert rag.collection_name == "test_hybrid"  # From mock_settings


def test_get_metrics_empty(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test get_metrics with no queries."""
    rag = HybridSearchRAG()
    metrics = rag.get_metrics()

    assert metrics["avg_retrieval_time"] == 0.0
    assert metrics["total_queries"] == 0
    assert "collection_name" in metrics
    assert metrics["fusion_method"] == "RRF"
    assert metrics["chunk_size"] == 700
    assert metrics["chunk_overlap"] == 100


def test_prepare_documents_invalid_path(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test prepare_documents with invalid path."""
    rag = HybridSearchRAG()

    with pytest.raises(ValueError, match="Documents path does not exist"):
        rag.prepare_documents("/nonexistent/path")


def test_query_structure(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test that query returns correct structure."""
    rag = HybridSearchRAG()

    # Mock the Qdrant query_points response
    mock_point1 = MagicMock()
    mock_point1.payload = {"text": "Test chunk 1", "source": "test.txt", "chunk_index": 0}
    mock_point1.score = 0.9

    mock_point2 = MagicMock()
    mock_point2.payload = {"text": "Test chunk 2", "source": "test.txt", "chunk_index": 1}
    mock_point2.score = 0.8

    mock_query_result = MagicMock()
    mock_query_result.points = [mock_point1, mock_point2]
    rag.client.query_points.return_value = mock_query_result

    # Mock OpenAI embedding
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    rag.embedding_client.embeddings.create.return_value = mock_embedding_response

    # Mock OpenAI chat completion
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    rag.openai_client.chat.completions.create.return_value = mock_chat_response

    result = rag.query("What is RAG?")

    assert "answer" in result
    assert "context" in result
    assert "metadata" in result
    assert isinstance(result["context"], list)
    assert len(result["context"]) == 2
    assert "retrieval_time" in result["metadata"]
    assert "chunks_retrieved" in result["metadata"]
    assert result["metadata"]["fusion_method"] == "RRF"


def test_collection_creation(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test that collection is created with correct config."""
    HybridSearchRAG()

    # Verify create_collection was called
    mock_qdrant.return_value.create_collection.assert_called_once()

    # Get the call arguments
    call_args = mock_qdrant.return_value.create_collection.call_args
    kwargs = call_args.kwargs

    # Verify collection name
    assert kwargs["collection_name"] == "test_hybrid"

    # Verify vectors_config has dense vector
    assert "dense" in kwargs["vectors_config"]

    # Verify sparse_vectors_config has sparse vector
    assert "sparse" in kwargs["sparse_vectors_config"]


def test_sparse_embedding_generation(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test sparse embedding generation."""
    rag = HybridSearchRAG()

    sparse_vec = rag._get_sparse_embedding("test text")

    # Verify the sparse model was called
    rag.sparse_model.embed.assert_called_once_with(["test text"])

    # Verify the sparse vector structure
    assert sparse_vec.indices == [1, 42, 100]
    assert sparse_vec.values == [0.22, 0.8, 0.5]


def test_dense_embedding_generation(
    mock_settings: MagicMock,
    mock_qdrant: MagicMock,
    mock_openai: MagicMock,
    mock_fastembed: MagicMock,
) -> None:
    """Test dense embedding generation."""
    rag = HybridSearchRAG()

    # Mock OpenAI embedding response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    rag.embedding_client.embeddings.create.return_value = mock_embedding_response

    dense_vec = rag._get_dense_embedding("test text")

    # Verify OpenAI was called
    rag.embedding_client.embeddings.create.assert_called_once()

    # Verify the dense vector length
    assert len(dense_vec) == 1536
