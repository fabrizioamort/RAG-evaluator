"""Tests for the Google Vertex AI Search RAG implementation."""

from unittest.mock import MagicMock, patch

import pytest

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.common.indexing import SourceDocument
from rag_evaluator.common.provider_interfaces import RetrievalTrace, RetrievedContext
from rag_evaluator.rag_implementations.google_vertex_search import client as vertex_client
from rag_evaluator.rag_implementations.google_vertex_search.google_vertex_rag import (
    GoogleVertexSearchRAG,
)

MODULE = "rag_evaluator.rag_implementations.google_vertex_search.google_vertex_rag"


@pytest.fixture
def mock_settings():
    """Mock the global settings used by the implementation."""
    with patch(f"{MODULE}.settings") as mock:
        mock.google_vertex_project_id = "test-project"
        mock.google_vertex_location = "global"
        mock.google_vertex_sa_key_path = None
        mock.google_vertex_data_store_id = ""
        mock.google_vertex_staging_bucket = "test-bucket"
        mock.google_vertex_generation_mode = "framework"
        mock.openai_model = "gpt-4o-mini"
        mock.openai_timeout = 600
        yield mock


@pytest.fixture
def mock_discoveryengine():
    """Mock the discoveryengine types module."""
    with patch(f"{MODULE}.discoveryengine") as mock:
        yield mock


@pytest.fixture
def mock_service_clients():
    """Mock the DiscoveryEngine client factories."""
    with (
        patch(f"{MODULE}.require_google_vertex"),
        patch(f"{MODULE}.get_data_store_service_client") as data_store_factory,
        patch(f"{MODULE}.get_document_service_client") as document_factory,
        patch(f"{MODULE}.get_search_service_client") as search_factory,
    ):
        yield {
            "data_store": data_store_factory.return_value,
            "document": document_factory.return_value,
            "search": search_factory.return_value,
        }


@pytest.fixture
def mock_llm_client():
    """Mock the OpenAI-compatible client used for framework generation."""
    with patch(f"{MODULE}.llm_client") as mock:
        yield mock.return_value


@pytest.fixture
def rag_factory(mock_settings, mock_discoveryengine, mock_service_clients, mock_llm_client):
    """Factory for building a GoogleVertexSearchRAG with all dependencies mocked."""

    def _make(**parameters) -> GoogleVertexSearchRAG:
        config = RAGConfig(name="Test Index", parameters=parameters)
        return GoogleVertexSearchRAG(config=config)

    return _make


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialization_generates_data_store_id(rag_factory):
    rag = rag_factory()
    assert rag.data_store_id  # auto-generated
    assert rag.location == "global"
    assert rag.reuse_existing_data_store is False
    assert rag.generation_mode == "framework"
    assert rag.num_previous_chunks == 2
    assert rag.num_next_chunks == 2


def test_initialization_uses_explicit_data_store_id(rag_factory):
    rag = rag_factory(data_store_id="my-existing-store", reuse_existing_data_store=True)
    assert rag.data_store_id == "my-existing-store"
    assert rag.reuse_existing_data_store is True


def test_chunk_window_is_clamped(rag_factory):
    rag = rag_factory(num_previous_chunks=10, num_next_chunks=-1)
    assert rag.num_previous_chunks == 3
    assert rag.num_next_chunks == 0


def test_missing_project_id_raises(mock_settings, mock_discoveryengine, mock_service_clients, mock_llm_client):
    mock_settings.google_vertex_project_id = ""
    config = RAGConfig(name="Test Index")
    with pytest.raises(ValueError, match="GOOGLE_VERTEX_PROJECT_ID"):
        GoogleVertexSearchRAG(config=config)


def test_constructor_requires_google_vertex_extra(mock_settings, mock_discoveryengine, mock_service_clients, mock_llm_client):
    with patch(f"{MODULE}.require_google_vertex", side_effect=ImportError("install extra")):
        with pytest.raises(ImportError, match="install extra"):
            GoogleVertexSearchRAG(config=RAGConfig(name="Test Index"))


# ---------------------------------------------------------------------------
# Data store lifecycle
# ---------------------------------------------------------------------------


def test_load_index_validates_existing_store(rag_factory, mock_service_clients):
    rag = rag_factory(data_store_id="existing-store")
    mock_service_clients["data_store"].get_data_store.return_value = MagicMock()

    rag.load_index()

    mock_service_clients["data_store"].get_data_store.assert_called_once()
    call_kwargs = mock_service_clients["data_store"].get_data_store.call_args.kwargs
    assert call_kwargs["name"].endswith("dataStores/existing-store")


def test_load_index_raises_when_not_found(rag_factory, mock_service_clients):
    rag = rag_factory(data_store_id="missing-store")
    mock_service_clients["data_store"].get_data_store.side_effect = vertex_client.NotFound("nope")

    with pytest.raises(ValueError, match="missing-store"):
        rag.load_index()


def test_ensure_data_store_creates_when_missing(rag_factory, mock_service_clients, mock_discoveryengine):
    rag = rag_factory(data_store_id="new-store")
    mock_service_clients["data_store"].get_data_store.side_effect = vertex_client.NotFound("nope")
    operation = MagicMock()
    mock_service_clients["data_store"].create_data_store.return_value = operation

    rag._ensure_data_store()

    mock_service_clients["data_store"].create_data_store.assert_called_once()
    operation.result.assert_called_once()
    assert rag._data_store_created is True
    assert rag._chunking_enabled is True


def test_ensure_data_store_reuses_when_present(rag_factory, mock_service_clients):
    rag = rag_factory(data_store_id="existing-store")
    mock_service_clients["data_store"].get_data_store.return_value = MagicMock()

    rag._ensure_data_store()

    mock_service_clients["data_store"].create_data_store.assert_not_called()
    assert rag._data_store_created is False


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_prepare_documents_reuse_existing_skips_import(rag_factory, mock_service_clients):
    rag = rag_factory(data_store_id="existing-store", reuse_existing_data_store=True)
    mock_service_clients["data_store"].get_data_store.return_value = MagicMock()

    rag.prepare_documents("data/raw")

    mock_service_clients["document"].import_documents.assert_not_called()
    mock_service_clients["data_store"].create_data_store.assert_not_called()


def test_prepare_documents_requires_staging_bucket(rag_factory, mock_settings):
    mock_settings.google_vertex_staging_bucket = ""
    rag = rag_factory(staging_bucket="")

    with pytest.raises(ValueError, match="staging bucket"):
        rag.prepare_documents("data/raw")


def test_prepare_documents_stages_and_imports(rag_factory, mock_service_clients, mock_discoveryengine):
    rag = rag_factory(data_store_id="new-store")
    mock_service_clients["data_store"].get_data_store.return_value = MagicMock()  # exists

    sources = [
        SourceDocument(
            doc_key="doc_a",
            source_path="/data/raw/a.txt",
            relative_path="a.txt",
            checksum="checksum-a",
        ),
        SourceDocument(
            doc_key="doc_b",
            source_path="/data/raw/b.txt",
            relative_path="b.txt",
            checksum="checksum-b",
        ),
    ]

    import_operation = MagicMock()
    mock_service_clients["document"].import_documents.return_value = import_operation

    with (
        patch(f"{MODULE}.discover_source_documents", return_value=sources),
        patch(f"{MODULE}.gcs_staging") as mock_gcs_staging,
    ):
        mock_gcs_staging.get_storage_client.return_value = MagicMock()
        mock_gcs_staging.gcs_blob_name.side_effect = lambda prefix, rel: f"{prefix}/{rel}"
        mock_gcs_staging.gcs_uri.side_effect = (
            lambda bucket, prefix, rel: f"gs://{bucket}/{prefix}/{rel}"
        )
        mock_gcs_staging.upload_file.return_value = "gs://uploaded"

        rag.prepare_documents("data/raw")

    assert mock_gcs_staging.upload_file.call_count == 2
    mock_service_clients["document"].import_documents.assert_called_once()
    import_operation.result.assert_called_once()
    assert rag._total_documents == 2


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieve_maps_chunks(rag_factory, mock_service_clients):
    rag = rag_factory()
    rag._chunking_enabled = True

    chunk_result = MagicMock()
    chunk_result.chunk.content = "chunk text"
    chunk_result.chunk.id = "chunk-1"
    chunk_result.chunk.relevance_score = 0.87
    chunk_result.chunk.document_metadata.uri = "gs://bucket/a.txt"
    chunk_result.chunk.document_metadata.title = "A Document"

    response = MagicMock()
    response.results = [chunk_result]
    mock_service_clients["search"].search.return_value = response

    context = rag.retrieve("What is in document A?", top_k=3)

    assert isinstance(context, RetrievedContext)
    assert context.chunks == ["chunk text"]
    chunk = context.chunk_details[0]
    assert chunk.chunk_id == "chunk-1"
    assert chunk.score == pytest.approx(0.87)
    assert chunk.source == "gs://bucket/a.txt"
    assert chunk.metadata["title"] == "A Document"

    assert isinstance(context.trace, RetrievalTrace)
    assert context.trace.strategy == "vector"
    assert context.trace.steps[0]["type"] == "vertex_search"
    assert context.trace.steps[0]["input"]["search_result_mode"] == "CHUNKS"


def test_retrieve_documents_fallback_when_chunking_disabled(rag_factory, mock_service_clients):
    rag = rag_factory()
    rag._chunking_enabled = False

    doc_result = MagicMock()
    doc_result.chunk = None
    doc_result.document.id = "doc-1"
    doc_result.document.derived_struct_data = {
        "link": "gs://bucket/a.txt",
        "snippets": [{"snippet": "snippet text"}],
    }

    response = MagicMock()
    response.results = [doc_result]
    mock_service_clients["search"].search.return_value = response

    context = rag.retrieve("What is in document A?", top_k=3)

    assert context.chunks == ["snippet text"]
    chunk = context.chunk_details[0]
    assert chunk.chunk_id == "doc-1"
    assert chunk.source == "gs://bucket/a.txt"
    assert chunk.score == 0.0


def test_retrieve_handles_empty_results(rag_factory, mock_service_clients):
    rag = rag_factory()
    response = MagicMock()
    response.results = []
    mock_service_clients["search"].search.return_value = response

    context = rag.retrieve("anything")

    assert context.chunks == []
    assert context.chunk_details == []


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_only_framework_mode(rag_factory, mock_llm_client):
    rag = rag_factory()

    completion = MagicMock()
    completion.choices[0].message.content = "The answer is 42."
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 10
    mock_llm_client.chat.completions.create.return_value = completion

    answer = rag._generate_only("What is the answer?", ["context chunk"])

    assert answer == "The answer is 42."
    assert rag._token_usage.prompt_tokens == 100
    assert rag._token_usage.completion_tokens == 10


def test_generate_grounded_mode(rag_factory, mock_discoveryengine):
    rag = rag_factory(generation_mode="google_grounded")

    with patch(f"{MODULE}.get_conversational_search_service_client") as factory:
        factory.return_value.answer_query.return_value.answer.answer_text = "Grounded answer"
        answer = rag._generate_grounded("What is the answer?")

    assert answer == "Grounded answer"


# ---------------------------------------------------------------------------
# Query / metrics
# ---------------------------------------------------------------------------


def test_query_composes_retrieve_and_generate(rag_factory, mock_service_clients, mock_llm_client):
    rag = rag_factory()

    response = MagicMock()
    response.results = []
    mock_service_clients["search"].search.return_value = response

    completion = MagicMock()
    completion.choices[0].message.content = "Answer"
    completion.usage.prompt_tokens = 5
    completion.usage.completion_tokens = 2
    mock_llm_client.chat.completions.create.return_value = completion

    result = rag.query("question", top_k=3)

    assert result["answer"] == "Answer"
    assert result["context"] == []
    assert result["metadata"]["data_store_id"] == rag.data_store_id
    assert result["metadata"]["generation_mode"] == "framework"
    assert result["metadata"]["token_usage"]["prompt_tokens"] == 5


def test_get_metrics_reports_data_store_info(rag_factory):
    rag = rag_factory(data_store_id="store-1", reuse_existing_data_store=True)

    metrics = rag.get_metrics()

    assert metrics["data_store_id"] == "store-1"
    assert metrics["reused_existing_data_store"] is True
    assert metrics["location"] == "global"
    assert "token_usage" in metrics


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_google_vertex_search():
    from rag_evaluator.common.base_rag import BaseRAG
    from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_rag_class

    assert "google_vertex_search" in RAG_TYPES
    cls = get_rag_class("google_vertex_search")
    assert issubclass(cls, BaseRAG)
    assert cls is GoogleVertexSearchRAG
