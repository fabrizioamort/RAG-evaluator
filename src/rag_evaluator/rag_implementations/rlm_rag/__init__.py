"""RLM Filesystem RAG - Recursive Language Model approach for large corpora.

This subpackage implements an RLM-style RAG that treats the document corpus as
an external filesystem environment. The LLM writes Python code to explore,
filter, and analyze documents using recursive sub-calls.

Quick Start:
    >>> from rag_evaluator.rag_implementations.rlm_rag import RLMFilesystemRAG, RLMConfig
    >>> rag = RLMFilesystemRAG(rlm_config=RLMConfig())
    >>> rag.prepare_documents("./docs")
    >>> result = rag.query("What are the main topics?")
    >>> print(result["answer"])

Security Modes:
    - "lite" (default): Fast in-process execution for trusted environments
    - "full": Subprocess isolation for untrusted document content

Architecture:
    - RLMFilesystemRAG: Main entry point (inherits from BaseRAG)
    - RLMAgent: Orchestrates the exploration loop
    - LLMClient: Two-tier model with circuit breaker and caching
    - DocumentProcessor: Prepares document filesystem with indexes

See Also:
    - :class:`RLMConfig`: Configuration options
    - :class:`~rag_evaluator.common.base_rag.BaseRAG`: Base interface
    - https://arxiv.org/html/2512.24601v1: RLM paper
"""

from .agent import (
    BudgetManager,
    ExecutionResult,
    FilesystemTools,
    RLMAgent,
    RLMResponse,
    SimpleREPL,
)
from .llm_client import ChatResponse, CircuitBreaker, LLMClient
from .preparation import DocumentProcessor, ManifestManager, SimpleContextRAG
from .rlm_rag import RLMConfig, RLMFilesystemRAG, StreamEvent, rlm_config_from_rag_config
from .security import InjectionGuard, ProcessREPL, SecureFilesystemTools

__version__ = "0.1.0"

__all__ = [
    # Version (inherited from parent package)
    "__version__",
    # Main API
    "RLMFilesystemRAG",
    "RLMConfig",
    "rlm_config_from_rag_config",
    "StreamEvent",
    # Agent components
    "RLMAgent",
    "RLMResponse",
    "BudgetManager",
    "FilesystemTools",
    "SimpleREPL",
    "ExecutionResult",
    # LLM client
    "LLMClient",
    "CircuitBreaker",
    "ChatResponse",
    # Preparation
    "DocumentProcessor",
    "ManifestManager",
    "SimpleContextRAG",
    # Security (opt-in)
    "ProcessREPL",
    "InjectionGuard",
    "SecureFilesystemTools",
]
