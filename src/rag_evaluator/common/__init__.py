"""Common utilities and interfaces for RAG implementations."""

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.provider_interfaces import (
    AsyncEmbeddingProvider,
    AsyncLLMProvider,
    EmbeddingProvider,
    GeneratedAnswer,
    LLMProvider,
    ProgressCallback,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
from rag_evaluator.common.token_tracker import TokenUsage

__all__ = [
    # Base classes
    "BaseRAG",
    "RAGConfig",
    # Provider interfaces
    "LLMProvider",
    "AsyncLLMProvider",
    "EmbeddingProvider",
    "AsyncEmbeddingProvider",
    # Data structures
    "RetrievedChunk",
    "RetrievalTrace",
    "RetrievedContext",
    "GeneratedAnswer",
    "TokenUsage",
    "ProgressCallback",
]
