"""Base class for RAG implementations."""

from abc import ABC, abstractmethod
from typing import Any


class BaseRAG(ABC):
    """Abstract base class for all RAG implementations."""

    def __init__(self, name: str) -> None:
        """Initialize the RAG implementation.

        Args:
            name: Name of the RAG implementation
        """
        self.name = name

    @abstractmethod
    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents for retrieval.

        Args:
            documents_path: Path to the directory containing documents
        """
        pass

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG system.

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: Retrieved context documents
                - metadata: Additional metadata (retrieval time, etc.)
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics for this RAG implementation.

        Returns:
            Dictionary containing performance metrics
        """
        pass
