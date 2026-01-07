"""Filesystem-based RAG implementation.

This module implements a RAG system that uses LLM-guided agent
navigation of a prepared filesystem structure instead of traditional
vector similarity search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.rag_implementations.filesystem_rag.agent.agent import (
    FilesystemRAGAgent,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.pipeline import (
    PreparationPipeline,
)


class FilesystemRAG(BaseRAG):
    """RAG implementation using LLM-guided filesystem navigation.

    Unlike traditional RAG approaches that use vector similarity search,
    Filesystem RAG employs an LLM-guided agent that navigates a prepared
    filesystem structure to find and retrieve relevant information.

    The system operates in two stages:
    1. Preparation: Documents are converted to markdown, analyzed, and
       indexed into a navigable filesystem structure
    2. Query: An agent uses tools to navigate the filesystem and
       synthesize answers

    Usage:
        rag = FilesystemRAG()
        rag.prepare_documents("data/raw")
        result = rag.query("What are the main challenges with RAG?")
        print(result["answer"])
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        prepared_path: str = "data/prepared/filesystem_rag",
        word_threshold: int = 1000,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        max_file_reads: int = 10,
    ) -> None:
        """Initialize Filesystem RAG.

        Args:
            llm_model: OpenAI model to use for agent and analysis
            prepared_path: Path for prepared filesystem output
            word_threshold: Word count threshold for LLM vs heuristic analysis
            max_iterations: Maximum ReAct loop iterations per query
            max_tool_calls: Maximum tool calls per query
            max_file_reads: Maximum file reads per query
        """
        super().__init__("Filesystem RAG")

        self.llm_model = llm_model
        self.prepared_path = prepared_path
        self.word_threshold = word_threshold
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_file_reads = max_file_reads

        # Components initialized during prepare_documents
        self._agent: FilesystemRAGAgent | None = None
        self._preparation_metrics: dict[str, Any] = {}

        # Query tracking
        self._query_metrics: list[dict[str, Any]] = []
        self._total_queries = 0

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare documents by converting to markdown and building indexes.

        This method:
        1. Loads raw documents from the input directory
        2. Converts them to markdown with structure detection
        3. Analyzes documents (hybrid: heuristic for simple, LLM for complex)
        4. Builds topic, entity, question, and timeline indexes
        5. Generates corpus overview and navigation guide
        6. Validates the output structure

        Args:
            documents_path: Path to directory containing raw documents
        """
        print(f"\nPreparing documents from: {documents_path}")

        # Run preparation pipeline
        pipeline = PreparationPipeline(
            input_path=documents_path,
            output_path=self.prepared_path,
            word_threshold=self.word_threshold,
            use_llm_synthesis=False,  # Use heuristic for cost savings
            preserve_originals=True,
        )

        result = pipeline.run()
        self._preparation_metrics = (
            result.get("metrics", {}).__dict__
            if hasattr(result.get("metrics", {}), "__dict__")
            else result.get("metrics", {})
        )

        # Initialize agent with prepared filesystem
        self._initialize_agent()

        print("Preparation complete. Agent initialized.")

    def _initialize_agent(self) -> None:
        """Initialize the agent with the prepared filesystem."""
        if not Path(self.prepared_path).exists():
            raise ValueError(
                f"Prepared path does not exist: {self.prepared_path}. "
                "Call prepare_documents() first."
            )

        self._agent = FilesystemRAGAgent(
            prepared_path=self.prepared_path,
            llm_model=self.llm_model,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            max_file_reads=self.max_file_reads,
        )

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using LLM-guided filesystem navigation.

        The agent:
        1. Routes the query to determine search strategy
        2. Navigates indexes and documents using tools
        3. Gathers relevant context
        4. Synthesizes an answer

        Args:
            question: The question to answer
            top_k: Ignored for filesystem RAG (included for interface compatibility)

        Returns:
            Dictionary containing:
            - answer: The generated answer
            - context: List of context chunks used
            - metadata: Query execution metadata
        """
        if self._agent is None:
            # Try to initialize if prepared path exists
            if Path(self.prepared_path).exists():
                self._initialize_agent()
            else:
                raise ValueError(
                    "Agent not initialized. Call prepare_documents() first, "
                    f"or ensure prepared path exists: {self.prepared_path}"
                )

        # Ensure agent is initialized (for type checker)
        if self._agent is None:
            raise RuntimeError("Agent initialization failed")

        # Execute query
        response = self._agent.query(question)

        # Track metrics
        query_metric = {
            "question": question,
            "query_time": response.metadata.get("query_time", 0.0),
            "tool_calls": response.metadata.get("tool_calls", 0),
            "files_read": len(response.metadata.get("files_read", [])),
            "search_mode": response.metadata.get("search_mode", "unknown"),
            "iterations": response.metadata.get("iterations", 0),
        }
        self._query_metrics.append(query_metric)
        self._total_queries += 1

        # Return in standard RAG format
        return {
            "answer": response.answer,
            "context": response.context,
            "metadata": {
                "retrieval_time": response.metadata.get("query_time", 0.0),
                "chunks_retrieved": len(response.context),
                "files_read": response.metadata.get("files_read", []),
                "tool_calls": response.metadata.get("tool_calls", 0),
                "search_mode": response.metadata.get("search_mode", "unknown"),
                "iterations": response.metadata.get("iterations", 0),
                "reasoning_trace": response.metadata.get("reasoning_trace", []),
            },
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary containing:
            - total_queries: Number of queries executed
            - avg_query_time: Average query execution time
            - avg_tool_calls: Average tool calls per query
            - avg_files_read: Average files read per query
            - search_mode_distribution: Count of queries by search mode
            - preparation_metrics: Metrics from document preparation
        """
        if not self._query_metrics:
            return {
                "total_queries": 0,
                "avg_query_time": 0.0,
                "avg_tool_calls": 0.0,
                "avg_files_read": 0.0,
                "search_mode_distribution": {},
                "preparation_metrics": self._preparation_metrics,
            }

        # Calculate averages
        total = len(self._query_metrics)
        avg_query_time = sum(m["query_time"] for m in self._query_metrics) / total
        avg_tool_calls = sum(m["tool_calls"] for m in self._query_metrics) / total
        avg_files_read = sum(m["files_read"] for m in self._query_metrics) / total
        avg_iterations = sum(m["iterations"] for m in self._query_metrics) / total

        # Count search modes
        search_mode_distribution: dict[str, int] = {}
        for m in self._query_metrics:
            mode = m["search_mode"]
            search_mode_distribution[mode] = search_mode_distribution.get(mode, 0) + 1

        return {
            "total_queries": total,
            "avg_retrieval_time": round(avg_query_time, 3),
            "avg_tool_calls": round(avg_tool_calls, 2),
            "avg_files_read": round(avg_files_read, 2),
            "avg_iterations": round(avg_iterations, 2),
            "search_mode_distribution": search_mode_distribution,
            "preparation_metrics": self._preparation_metrics,
        }

    def get_prepared_path(self) -> str:
        """Get the path to the prepared filesystem.

        Returns:
            Path to prepared filesystem
        """
        return self.prepared_path

    def is_prepared(self) -> bool:
        """Check if documents have been prepared.

        Returns:
            True if prepared filesystem exists
        """
        return Path(self.prepared_path).exists()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get agent cache statistics.

        Returns:
            Cache statistics dictionary
        """
        if self._agent is None:
            return {"error": "Agent not initialized"}
        return self._agent.get_cache_stats()

    def reload_cache(self) -> bool:
        """Reload the agent's session cache.

        Returns:
            True if reload successful
        """
        if self._agent is None:
            return False
        return self._agent.reload_cache()

    def get_corpus_overview(self) -> str | None:
        """Get the corpus overview content.

        Returns:
            Corpus overview markdown or None if not available
        """
        overview_path = Path(self.prepared_path) / "_meta" / "corpus_overview.md"
        if overview_path.exists():
            return overview_path.read_text(encoding="utf-8")
        return None

    def get_statistics(self) -> dict[str, Any] | None:
        """Get the prepared corpus statistics.

        Returns:
            Statistics dictionary or None if not available
        """
        import json

        stats_path = Path(self.prepared_path) / "_meta" / "statistics.json"
        if stats_path.exists():
            result: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
            return result
        return None
