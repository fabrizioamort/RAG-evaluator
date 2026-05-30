"""RLM Filesystem RAG - Main entry point and configuration."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig

if TYPE_CHECKING:
    pass  # Future type-only imports

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLMConfig:
    """Configuration for RLM Filesystem RAG.

    v6: Simplified with security_mode toggle.

    Security Modes:
        - "lite": In-process REPL, no injection defense (trusted environments)
        - "full": Subprocess isolation, injection guard (untrusted content)
    """

    # === Security Mode ===
    security_mode: Literal["lite", "full"] = "lite"

    # === Two-Tier Model Architecture ===
    orchestrator_model: str = "gpt-5-mini"      # Main reasoning
    worker_model: str = "gpt-5-nano"            # Chunk processing, summaries

    # === REPL Limits ===
    max_repl_steps: int = 15
    repl_timeout: float = 5.0                   # Seconds per step

    # === File Access ===
    max_file_reads: int = 12
    max_read_bytes: int = 50_000
    max_read_lines: int = 1000

    # === Sub-LLM Budget ===
    max_sub_calls: int = 8
    max_recursion_depth: int = 2
    max_tokens: int = 80_000

    # === Circuit Breaker ===
    circuit_failure_threshold: int = 3
    circuit_timeout: float = 60.0

    # === Retry ===
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # === Caching ===
    enable_cache: bool = True
    cache_max_entries: int = 100
    cache_ttl_seconds: float = 300.0

    # === Routing ===
    small_corpus_threshold: int = 10            # Use SimpleContextRAG below this

    # === Preparation ===
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_llm_summaries: bool = True
    use_llm_topics: bool = True
    max_topics_per_doc: int = 5

    # === Confidence ===
    min_sources_for_high_confidence: int = 2    # Rule-based, no LLM verify

    # === Observability ===
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        errors = []

        if self.max_repl_steps < 1:
            errors.append("max_repl_steps must be >= 1")
        if self.max_repl_steps > 50:
            errors.append("max_repl_steps > 50 is excessive")

        if self.repl_timeout < 0.1:
            errors.append("repl_timeout must be >= 0.1 seconds")
        if self.repl_timeout > 60:
            errors.append("repl_timeout > 60s is too long")

        if self.max_file_reads < 1:
            errors.append("max_file_reads must be >= 1")

        if self.max_read_bytes < 1000:
            errors.append("max_read_bytes must be >= 1000")

        if self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be < chunk_size")

        if self.small_corpus_threshold < 1:
            errors.append("small_corpus_threshold must be >= 1")

        if self.circuit_failure_threshold < 1:
            errors.append("circuit_failure_threshold must be >= 1")

        if self.min_sources_for_high_confidence < 1:
            errors.append("min_sources_for_high_confidence must be >= 1")

        if self.security_mode not in ("lite", "full"):
            errors.append("security_mode must be 'lite' or 'full'")

        if errors:
            raise ValueError(f"Invalid RLMConfig: {'; '.join(errors)}")

    @property
    def use_process_isolation(self) -> bool:
        """Whether to use subprocess isolation for REPL."""
        return self.security_mode == "full"

    @property
    def use_injection_defense(self) -> bool:
        """Whether to wrap documents with injection defense."""
        return self.security_mode == "full"

    @property
    def use_strict_paths(self) -> bool:
        """Whether to enforce strict path whitelist."""
        return self.security_mode == "full"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def rlm_config_from_rag_config(config: RAGConfig) -> RLMConfig:
    """Build RLMConfig from generic RAGConfig parameters.

    The platform stores RAG-specific parameters as JSON, while the core
    implementation expects a typed dataclass. Unknown parameters are ignored so
    platform-managed values like prepared_path can live in the same JSON blob.
    """
    params = dict(config.parameters or {})
    if not params.get("orchestrator_model") or params["orchestrator_model"] == "RAG config llm_model":
        params["orchestrator_model"] = config.llm_model

    field_names = {field.name for field in fields(RLMConfig)}
    rlm_params = {key: value for key, value in params.items() if key in field_names}
    return RLMConfig(**rlm_params)


# ============================================================================
# Stream Event
# ============================================================================

@dataclass
class StreamEvent:
    """Event emitted during streaming query execution."""
    event_type: str  # "step", "code", "output", "answer", "error"
    content: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Main RAG Class
# ============================================================================

class RLMFilesystemRAG(BaseRAG):
    """RLM-style RAG using code execution for corpus exploration.

    This RAG implementation treats the document corpus as a filesystem that
    an LLM agent explores by writing and executing Python code. The agent
    uses recursive sub-LLM calls to process document chunks.

    Features:
        - security_mode="lite" (default): Fast, in-process execution
        - security_mode="full": Subprocess isolation, injection defense
        - Two-tier model: orchestrator (reasoning) + worker (processing)
        - Circuit breaker for API reliability
        - Manifest-based cache invalidation
        - Automatic fallback to SimpleContextRAG for small corpora

    Example:
        >>> from rag_evaluator.rag_implementations.rlm_rag import RLMFilesystemRAG, RLMConfig
        >>> rag = RLMFilesystemRAG(rlm_config=RLMConfig(security_mode="lite"))
        >>> rag.prepare_documents("./my_documents")
        >>> result = rag.query("What are the main concepts discussed?")
        >>> print(result["answer"])
        >>> print(f"Confidence: {result['metadata']['confidence']}")

    See Also:
        - :class:`RLMConfig`: Configuration options
        - :class:`~rag_evaluator.common.base_rag.BaseRAG`: Base interface
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        rlm_config: RLMConfig | None = None,
        prepared_path: str | Path | None = None,
    ):
        """Initialize RLM Filesystem RAG.

        Args:
            config: Base RAG configuration (optional, for evaluator compatibility)
            rlm_config: RLM-specific configuration
            prepared_path: Output path for the prepared filesystem
        """
        super().__init__(name="RLM Filesystem RAG", config=config)
        self.rlm_config = rlm_config or rlm_config_from_rag_config(self.config)
        configured_prepared_path = (self.config.parameters or {}).get("prepared_path")
        self.prepared_path = str(
            Path(
                prepared_path
                or configured_prepared_path
                or Path(self.config.storage_path) / "rlm_rag"
            ).resolve()
        )

        # Set up logging
        logging.basicConfig(level=getattr(logging, self.rlm_config.log_level))

        # Components (initialized by prepare_documents)
        self._agent = None
        self._simple_rag = None
        self._prepared_path: Path | None = None
        self._manifest = None
        self._metrics: dict[str, Any] = {}
        self._use_simple_mode: bool = False

        logger.info(
            f"RLMFilesystemRAG initialized: "
            f"security_mode={self.rlm_config.security_mode}, "
            f"orchestrator={self.rlm_config.orchestrator_model}, "
            f"worker={self.rlm_config.worker_model}"
        )

    def prepare_documents(self, documents_path: str, force: bool = False) -> dict[str, Any]:
        """Prepare documents for querying.

        This method:
        1. Checks if documents have changed (via manifest)
        2. Processes documents into prepared filesystem
        3. Generates summaries and extracts topics
        4. Decides whether to use RLM agent or simple fallback

        Args:
            documents_path: Path to directory containing source documents
            force: If True, regenerate even if manifest indicates no changes

        Returns:
            Dict with preparation metrics

        Raises:
            FileNotFoundError: If documents_path doesn't exist
            ValueError: If documents_path isn't a directory
        """
        from .preparation import DocumentProcessor, ManifestManager

        input_dir = Path(documents_path).resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Documents path not found: {documents_path}")
        if not input_dir.is_dir():
            raise ValueError(f"Documents path must be a directory: {documents_path}")

        output_dir = Path(self.prepared_path).resolve()

        # Check manifest for changes
        self._manifest = ManifestManager(output_dir)

        if not force and self._manifest.is_valid(input_dir, self.rlm_config):
            logger.info("Documents unchanged (manifest valid), skipping preparation")
            self._prepared_path = output_dir
        else:
            logger.info(f"Preparing documents from {input_dir}")
            processor = DocumentProcessor(self.rlm_config)
            self._prepared_path, prep_metrics = processor.prepare(str(input_dir), output_dir)
            self._metrics["preparation"] = prep_metrics
            self._manifest.update(input_dir, self.rlm_config)

        # Load catalog and decide routing
        self._load_catalog_and_route()

        return {
            "prepared_path": str(self._prepared_path),
            "mode": "simple_context" if self._use_simple_mode else "rlm_agent",
            **self._metrics,
        }

    def _load_catalog_and_route(self) -> None:
        """Load catalog and decide between RLM agent or simple mode."""
        catalog_path = self._prepared_path / "_meta" / "catalog.json"

        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        documents = catalog.get("documents", [])
        doc_count = len(documents)

        self._metrics["total_documents"] = doc_count
        self._metrics["total_chunks"] = sum(
            d.get("chunk_count", 1) for d in documents
        )
        self._metrics["total_words"] = sum(
            d.get("word_count", 0) for d in documents
        )

        # Route based on corpus size
        if doc_count <= self.rlm_config.small_corpus_threshold:
            logger.info(
                f"Small corpus ({doc_count} docs <= {self.rlm_config.small_corpus_threshold}), "
                f"using SimpleContextRAG"
            )
            self._use_simple_mode = True
            self._init_simple_rag()
        else:
            logger.info(
                f"Large corpus ({doc_count} docs), using RLM agent"
            )
            self._use_simple_mode = False
            self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the RLM agent."""
        from .agent import RLMAgent

        self._agent = RLMAgent(
            prepared_path=self._prepared_path,
            config=self.rlm_config,
            token_usage=self._token_usage,
        )

    def _init_simple_rag(self) -> None:
        """Initialize simple fallback for small corpora."""
        from .preparation import SimpleContextRAG

        self._simple_rag = SimpleContextRAG(
            prepared_path=self._prepared_path,
            token_usage=self._token_usage,
            config=self.rlm_config,
        )

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Answer a question using the prepared corpus.

        Args:
            question: The question to answer
            top_k: Number of documents for simple mode (ignored in RLM mode)

        Returns:
            Dict with:
                - answer: The generated answer
                - context: List of relevant context snippets
                - metadata: Dict with timing, sources, confidence, etc.

        Raises:
            RuntimeError: If prepare_documents hasn't been called
        """
        if self._prepared_path is None:
            raise RuntimeError(
                "Documents not prepared. Call prepare_documents() first."
            )

        self.reset_token_usage()

        # Route to appropriate implementation
        if self._use_simple_mode and self._simple_rag:
            result = self._simple_rag.query(question, top_k)
            result["metadata"]["security_mode"] = self.rlm_config.security_mode
            return result

        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        response = self._agent.query(question)

        return {
            "answer": response.answer,
            "context": response.context,
            "metadata": {
                "retrieval_time": response.retrieval_time,
                "generation_time": response.generation_time,
                "sources": response.sources,
                "confidence": response.confidence,
                "token_usage": self._token_usage.to_dict(),
                "trace": response.trace,
                "mode": "rlm_agent",
                "security_mode": self.rlm_config.security_mode,
            },
        }

    def query_with_trace(self, question: str, top_k: int = 5) -> dict[str, Any]:
        result = self.query(question, top_k)
        return {
            "answer": result["answer"],
            "context": result["context"],
            "metadata": result["metadata"],
            "retrieval_trace": result["metadata"].get("trace", {}),
        }

    def query_stream(
        self, question: str, top_k: int = 5
    ) -> Generator[StreamEvent, None, dict[str, Any]]:
        """Answer a question with streaming events.

        Yields StreamEvent objects during exploration, allowing real-time
        UI updates. Returns final result when complete.

        Args:
            question: The question to answer
            top_k: Number of documents for simple mode

        Yields:
            StreamEvent objects with exploration progress

        Returns:
            Final result dict (same format as query())
        """
        if self._prepared_path is None:
            yield StreamEvent(
                event_type="error",
                content="Documents not prepared. Call prepare_documents() first.",
                step=0,
            )
            return {"answer": "", "context": [], "metadata": {"error": "not_prepared"}}

        self.reset_token_usage()

        # Simple mode doesn't support streaming (single call)
        if self._use_simple_mode and self._simple_rag:
            result = self._simple_rag.query(question, top_k)
            yield StreamEvent(
                event_type="answer",
                content=result["answer"],
                step=1,
                metadata={"mode": "simple_context"},
            )
            return result

        if self._agent is None:
            yield StreamEvent(
                event_type="error",
                content="Agent not initialized",
                step=0,
            )
            return {"answer": "", "context": [], "metadata": {"error": "no_agent"}}

        # Stream from agent
        return (yield from self._agent.query_stream(question))

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics about the RAG system.

        Returns:
            Dict with document counts, mode, configuration, etc.
        """
        metrics = {
            "name": self.name,
            "prepared_path": str(self._prepared_path) if self._prepared_path else None,
            "mode": "simple_context" if self._use_simple_mode else "rlm_agent",
            "security_mode": self.rlm_config.security_mode,
            "total_documents": self._metrics.get("total_documents", 0),
            "total_chunks": self._metrics.get("total_chunks", 0),
            "total_words": self._metrics.get("total_words", 0),
            "config": self.rlm_config.to_dict(),
        }

        if self._agent:
            metrics["agent_stats"] = self._agent.get_stats()

        if self._manifest:
            metrics["manifest"] = self._manifest.get_info()

        if "preparation" in self._metrics:
            metrics["preparation"] = self._metrics["preparation"]

        return metrics

    def reset_token_usage(self) -> None:
        """Reset token counters for a new query."""
        self._token_usage.reset()

    def close(self) -> None:
        """Clean up resources.

        Call this when done with the RAG instance.
        """
        if self._agent:
            self._agent.close()
            self._agent = None

        if self._simple_rag:
            self._simple_rag = None

        logger.info("RLMFilesystemRAG closed")

    def _get_strategy_name(self) -> str:
        """Return strategy name for retrieval tracing.

        Returns:
            "agentic" - this RAG uses an agent-based exploration strategy.
        """
        return "agentic"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
