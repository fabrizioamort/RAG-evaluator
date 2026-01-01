"""Tests for BaseRAG abstract class."""

from typing import Any

import pytest

from rag_evaluator.common.base_rag import BaseRAG


class DummyRAG(BaseRAG):
    """Dummy RAG implementation for testing."""

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare documents."""
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG."""
        return {
            "answer": "test answer",
            "context": ["test context"],
            "metadata": {"retrieval_time": 0.1},
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics."""
        return {"test_metric": 1.0}


def test_base_rag_initialization() -> None:
    """Test that BaseRAG can be instantiated with a name."""
    rag = DummyRAG("Test RAG")
    assert rag.name == "Test RAG"


def test_base_rag_query() -> None:
    """Test that query returns expected structure."""
    rag = DummyRAG("Test RAG")
    result = rag.query("test question")

    assert "answer" in result
    assert "context" in result
    assert "metadata" in result
    assert result["answer"] == "test answer"


def test_base_rag_get_metrics() -> None:
    """Test that get_metrics returns expected structure."""
    rag = DummyRAG("Test RAG")
    metrics = rag.get_metrics()

    assert isinstance(metrics, dict)
    assert "test_metric" in metrics


def test_base_rag_cannot_instantiate_abstract() -> None:
    """Test that BaseRAG cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseRAG("Test")  # type: ignore
