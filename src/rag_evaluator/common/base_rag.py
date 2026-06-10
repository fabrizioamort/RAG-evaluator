"""Base class for RAG implementations.

This module defines the abstract base class for all RAG implementations,
providing a standardized interface for document preparation, retrieval,
generation, and querying.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.common.indexing import CheckpointStore
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    ProgressCallback,
    RetrievedContext,
)
from rag_evaluator.common.token_tracker import TokenUsage


@dataclass
class RAGConfig:
    """Configuration container for RAG implementations.

    Provides a standardized way to configure RAG systems
    with support for various parameters and LLM providers.
    """

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    storage_path: str = "./data/indexes"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    llm_reasoning_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "storage_path": self.storage_path,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "embedding_base_url": self.embedding_base_url,
            "llm_reasoning_effort": self.llm_reasoning_effort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGConfig":
        """Create RAGConfig from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            New RAGConfig instance
        """
        return cls(
            name=data.get("name", "Unknown"),
            parameters=data.get("parameters", {}),
            storage_path=data.get("storage_path", "./data/indexes"),
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            llm_base_url=data.get("llm_base_url"),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            embedding_provider=data.get("embedding_provider", "openai"),
            embedding_base_url=data.get("embedding_base_url"),
            llm_reasoning_effort=data.get("llm_reasoning_effort"),
        )


class BaseRAG(ABC):
    """Abstract base class for all RAG implementations.

    This class defines the interface that all RAG implementations must follow.
    It supports both the traditional query() method and the new separate
    retrieve()/generate() methods for more flexibility.

    The new interface enables:
    - Caching retrieval results and running generation experiments
    - Re-scoring and prompt experiments without re-retrieval
    - Better observability with standardized retrieval traces
    - Token usage tracking for cost calculation
    """

    def __init__(self, name: str, config: RAGConfig | None = None) -> None:
        """Initialize the RAG implementation.

        Args:
            name: Name of the RAG implementation
            config: Optional RAGConfig for configuration
        """
        self.name = name
        self.config = config or RAGConfig(name=name)
        self._total_token_usage = TokenUsage()
        self._token_local = threading.local()
        self._progress_callback: ProgressCallback | None = None
        self._metrics_lock = threading.Lock()

    @property
    def _token_usage(self) -> TokenUsage:
        """Get thread-local token usage.

        Returns:
            TokenUsage instance for the current thread.
        """
        if not hasattr(self._token_local, "usage"):
            # Create a new TokenUsage that also updates the total
            self._token_local.usage = TokenUsage()

            # We need to wrap the add methods to also update the global total
            # This is a bit hacky but avoids changing all RAG implementations
            original_prompt = self._token_local.usage.add_prompt_tokens
            original_completion = self._token_local.usage.add_completion_tokens
            original_embedding = self._token_local.usage.add_embedding_tokens

            def wrapped_prompt(count: int) -> None:
                original_prompt(count)
                self._total_token_usage.add_prompt_tokens(count)

            def wrapped_completion(count: int) -> None:
                original_completion(count)
                self._total_token_usage.add_completion_tokens(count)

            def wrapped_embedding(count: int) -> None:
                original_embedding(count)
                self._total_token_usage.add_embedding_tokens(count)

            self._token_local.usage.add_prompt_tokens = wrapped_prompt
            self._token_local.usage.add_completion_tokens = wrapped_completion
            self._token_local.usage.add_embedding_tokens = wrapped_embedding

        return self._token_local.usage  # type: ignore[no-any-return]

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Set callback for progress reporting during long operations.

        Args:
            callback: Function to call with (current, total) progress
        """
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set.

        Args:
            current: Current progress value
            total: Total expected value
        """
        if self._progress_callback:
            self._progress_callback(current, total)

    def close(self) -> None:
        """Close any resources held by the RAG implementation.

        Subclasses should override this to close database connections,
        file handles, or other resources.
        """
        pass

    def load_index(self) -> None:
        """Open existing prepared/indexed artifacts without mutating them.

        Build paths call :meth:`prepare_documents`. Query paths for ready
        indexes call this method so implementations can initialize runtime
        handles without rebuilding or rewriting stored artifacts.
        """
        return None

    @abstractmethod
    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents for retrieval.

        Args:
            documents_path: Path to the directory containing documents
        """
        pass

    def prepare_documents_resumable(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Prepare documents using a durable checkpoint store.

        Implementations that do not override this keep the previous behavior.
        """
        self.prepare_documents(documents_path)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieval only (no generation).

        Enables caching retrieval results and running generation experiments
        without re-indexing.

        The default implementation extracts retrieval from query() for
        backward compatibility. Subclasses should override for better
        performance and tracing.

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            RetrievedContext with chunks and trace information
        """
        # Default implementation for backward compatibility
        # Subclasses should override this for proper implementation
        import time

        from rag_evaluator.common.provider_interfaces import (
            RetrievalTrace,
            RetrievedChunk,
        )

        start_time = time.time()

        # Call query and extract context
        result = self._retrieve_only(question, top_k)

        retrieval_time = time.time() - start_time

        # Build chunk details from raw chunks
        chunks = result.get("context", [])
        metadata_list = result.get("metadata", {}).get("sources", [])

        chunk_details = []
        for i, chunk in enumerate(chunks):
            source = metadata_list[i] if i < len(metadata_list) else "unknown"
            chunk_details.append(
                RetrievedChunk(
                    content=chunk,
                    document_id=source,
                    chunk_id=f"chunk_{i}",
                    score=1.0 - (i * 0.1),  # Decreasing score by rank
                    rank=i,
                    source=source,
                )
            )

        # Build trace
        trace = RetrievalTrace(
            strategy=self._get_strategy_name(),
            total_duration_ms=retrieval_time * 1000,
        )
        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=chunks,
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=retrieval_time,
        )

    def _retrieve_only(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Internal method for retrieval-only operation.

        Subclasses can override this to provide retrieval without generation.
        Default calls query() which includes generation (less efficient).

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with context and metadata
        """
        # Default: call full query and just return context part
        result = self.query(question, top_k)
        return {
            "context": result.get("context", []),
            "metadata": result.get("metadata", {}),
        }

    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer:
        """Generation only.

        Enables re-scoring and prompt experiments without re-retrieval.

        The default implementation extracts generation logic from query().
        Subclasses should override for proper implementation.

        Args:
            question: The question to answer
            context: Previously retrieved context

        Returns:
            GeneratedAnswer with text and token usage
        """
        # Default implementation for backward compatibility
        # Subclasses should override this for proper implementation
        import time

        start_time = time.time()

        # Use the provided context to generate answer
        answer = self._generate_only(question, context.chunks)

        generation_time = time.time() - start_time

        return GeneratedAnswer(
            text=answer,
            generation_time=generation_time,
            prompt_tokens=self._token_usage.prompt_tokens,
            completion_tokens=self._token_usage.completion_tokens,
        )

    def _generate_only(self, question: str, context_chunks: list[str]) -> str:
        """Internal method for generation-only operation.

        Subclasses should override this to provide generation without retrieval.

        Args:
            question: The question to answer
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer text
        """
        # Default: not implemented, subclasses should override
        raise NotImplementedError("Subclass must implement _generate_only() or override generate()")

    def _get_strategy_name(self) -> str:
        """Get the retrieval strategy name for tracing.

        Returns:
            Strategy name (e.g., "vector", "hybrid", "graph", "agentic")
        """
        # Default based on class name
        name_lower = self.__class__.__name__.lower()
        if "hybrid" in name_lower:
            return "hybrid"
        elif "graph" in name_lower:
            return "graph"
        elif "filesystem" in name_lower or "agent" in name_lower:
            return "agentic"
        else:
            return "vector"

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG system.

        This is the traditional interface that combines retrieval and generation.
        For new implementations, consider using retrieve() + generate() separately.

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: Retrieved context documents
                - metadata: Additional metadata (retrieval time, token usage, etc.)
        """
        pass

    def query_with_trace(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query with full retrieval trace.

        Convenience method that uses retrieve() + generate() and includes
        the full retrieval trace in the response.

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing answer, context, metadata, and retrieval_trace
        """
        # Reset token usage for this query
        self._token_usage.reset()

        # Retrieve
        context = self.retrieve(question, top_k)

        # Generate
        answer = self.generate(question, context)

        return {
            "answer": answer.text,
            "context": context.chunks,
            "metadata": {
                "retrieval_time": context.retrieval_time,
                "generation_time": answer.generation_time,
                "token_usage": self._token_usage.to_dict(),
            },
            "retrieval_trace": context.trace.to_dict(),
        }

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics for this RAG implementation.

        Returns:
            Dictionary containing performance metrics
        """
        pass

    def get_token_usage(self) -> TokenUsage:
        """Return global token usage across all steps/threads.

        Returns:
            TokenUsage object with aggregate counts
        """
        return self._total_token_usage

    def reset_token_usage(self) -> None:
        """Reset thread-local token usage counters.
        Used at the start of individual queries to track per-query usage.
        """
        self._token_usage.reset()

    def reset_global_token_usage(self) -> None:
        """Reset global token usage counters."""
        self._total_token_usage.reset()
        self._token_usage.reset()

    def get_config(self) -> RAGConfig:
        """Get the RAG configuration.

        Returns:
            Current RAGConfig
        """
        return self.config
