"""Provider interfaces for RAG, LLM, and Embedding components.

This module defines the standardized interfaces and data structures
for RAG implementations, enabling separation of retrieval and generation,
consistent trace formats, and provider abstraction.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata.

    Represents a piece of content retrieved from a knowledge base,
    including relevance scoring and source provenance.
    """

    content: str
    document_id: str
    chunk_id: str
    score: float
    rank: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """Standardized retrieval trace for all RAG types.

    Provides a unified format for tracing retrieval operations
    across different RAG implementations (vector, hybrid, graph, agentic).
    This enables consistent debugging and explainability.
    """

    strategy: str  # "vector" | "hybrid" | "graph" | "agentic"
    steps: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    fusion_details: dict[str, Any] | None = None  # RRF k, per-list ranks, etc.
    total_duration_ms: float = 0.0

    def add_step(
        self,
        step_type: str,
        input_data: Any,
        output_refs: list[str] | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a step to the retrieval trace.

        Args:
            step_type: Type of step (e.g., "dense_search", "sparse_search", "fusion")
            input_data: Input to this step (query, parameters, etc.)
            output_refs: References to outputs (chunk IDs, etc.)
            duration_ms: Duration of this step in milliseconds
            metadata: Additional step-specific metadata
        """
        step = {
            "type": step_type,
            "input": input_data,
            "output_refs": output_refs or [],
            "duration_ms": duration_ms,
        }
        if metadata:
            step["metadata"] = metadata
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization.

        Returns:
            Dictionary representation of the trace.
        """
        return {
            "strategy": self.strategy,
            "steps": self.steps,
            "retrieved_chunks": [
                {
                    "content": c.content,
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "rank": c.rank,
                    "source": c.source,
                    "metadata": c.metadata,
                }
                for c in self.retrieved_chunks
            ],
            "fusion_details": self.fusion_details,
            "total_duration_ms": self.total_duration_ms,
        }


@dataclass
class RetrievedContext:
    """Result of retrieval operation.

    Contains the retrieved chunks along with detailed information
    for each chunk and the full retrieval trace.
    """

    chunks: list[str]  # Raw chunk content for backward compatibility
    chunk_details: list[RetrievedChunk]
    trace: RetrievalTrace
    retrieval_time: float  # Total retrieval time in seconds


@dataclass
class GeneratedAnswer:
    """Result of generation operation.

    Contains the generated text along with token usage
    and timing information.
    """

    text: str
    generation_time: float  # Generation time in seconds
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides a unified interface for text generation across
    different providers (OpenAI, Ollama, Anthropic, etc.).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier.

        Returns:
            Model name/identifier string
        """
        pass


class AsyncLLMProvider(ABC):
    """Async version of LLM provider for platform use."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """Generate text from a prompt asynchronously.

        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier.

        Returns:
            Model name/identifier string
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Provides a unified interface for generating embeddings
    from different providers.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model identifier.

        Returns:
            Model name/identifier string
        """
        pass


class AsyncEmbeddingProvider(ABC):
    """Async version of embedding provider for platform use."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query asynchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model identifier.

        Returns:
            Model name/identifier string
        """
        pass


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int], None]
