"""Filesystem-based RAG implementation.

This module implements a RAG system that uses LLM-guided agent
navigation of a prepared filesystem structure instead of traditional
vector similarity search.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
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
        config: RAGConfig | None = None,
    ) -> None:
        """Initialize Filesystem RAG.

        Args:
            llm_model: OpenAI model to use for agent and analysis
            prepared_path: Path for prepared filesystem output
            word_threshold: Word count threshold for LLM vs heuristic analysis
            max_iterations: Maximum ReAct loop iterations per query
            max_tool_calls: Maximum tool calls per query
            max_file_reads: Maximum file reads per query
            config: Optional RAGConfig for configuration
        """
        super().__init__("Filesystem RAG", config=config)

        # Override llm_model from config if provided
        self.llm_model = self.config.llm_model if config else llm_model
        # Resolve to absolute path to ensure robustness against CWD changes
        self.prepared_path = str(Path(prepared_path).resolve())
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

    def close(self) -> None:
        """Close agent and clear metrics."""
        self._agent = None
        self._query_metrics = []

    def load_index(self) -> None:
        """Initialize the query agent from an existing prepared filesystem."""
        self._initialize_agent()

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
            reasoning_effort=self.config.llm_reasoning_effort if self.config else None,
        )

    def _ensure_agent(self) -> FilesystemRAGAgent:
        """Return an initialized agent, loading an existing prepared index if needed."""
        if self._agent is None:
            if Path(self.prepared_path).exists():
                self._initialize_agent()
            else:
                raise ValueError(
                    "Agent not initialized. Call prepare_documents() first, "
                    f"or ensure prepared path exists: {self.prepared_path}"
                )

        if self._agent is None:
            raise RuntimeError("Agent initialization failed")
        return self._agent

    def _context_from_agent_response(
        self,
        question: str,
        response: Any,
        retrieval_time: float,
    ) -> RetrievedContext:
        """Convert an agent response into standardized retrieved context and trace."""
        context_sources = response.metadata.get("context_sources") or response.metadata.get(
            "files_read", []
        )
        files_read = response.metadata.get("files_read", [])

        chunk_details = []
        for i, chunk in enumerate(response.context):
            source = context_sources[i] if i < len(context_sources) else f"agent_context_{i}"

            chunk_details.append(
                RetrievedChunk(
                    content=chunk,
                    document_id=source,
                    chunk_id=f"fs_chunk_{i}",
                    score=max(0.0, 1.0 - (i * 0.05)),
                    rank=i,
                    source=source,
                    metadata={
                        "search_mode": response.metadata.get("search_mode", "unknown"),
                    },
                )
            )

        trace = RetrievalTrace(
            strategy="agentic",
            total_duration_ms=retrieval_time * 1000,
        )

        trace.add_step(
            step_type="query_routing",
            input_data={"query": question},
            output_refs=[],
            duration_ms=0,
            metadata={
                "search_mode": response.metadata.get("search_mode", "unknown"),
                "routing_confidence": response.metadata.get("routing_confidence"),
            },
        )

        prefetch_candidates = response.metadata.get("prefetch_candidates", [])
        if prefetch_candidates:
            trace.add_step(
                step_type="lexical_prefetch",
                input_data={"terms": response.metadata.get("prefetch_terms", [])},
                output_refs=[candidate["doc_id"] for candidate in prefetch_candidates],
                duration_ms=0,
                metadata={"candidates": prefetch_candidates},
            )

        for i, step in enumerate(response.metadata.get("reasoning_trace", [])):
            trace.add_step(
                step_type="agent_step",
                input_data={"step_number": i + 1},
                output_refs=[],
                duration_ms=0,
                metadata={"reasoning": step},
            )

        for file_path in files_read:
            trace.add_step(
                step_type="file_read",
                input_data={"file": file_path},
                output_refs=[],
                duration_ms=0,
            )

        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=response.context,
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=retrieval_time,
        )

    def _track_agent_response(self, question: str, response: Any) -> None:
        """Track query metrics and estimated token use for an agent response."""
        query_metric = {
            "question": question,
            "query_time": response.metadata.get("query_time", 0.0),
            "tool_calls": response.metadata.get("tool_calls", 0),
            "files_read": len(response.metadata.get("files_read", [])),
            "search_mode": response.metadata.get("search_mode", "unknown"),
            "iterations": response.metadata.get("iterations", 0),
        }
        with self._metrics_lock:
            self._query_metrics.append(query_metric)
            self._total_queries += 1

        estimated_prompt = len(question) * response.metadata.get("iterations", 1) * 10
        estimated_completion = len(response.answer) // 4
        self._token_usage.add_prompt_tokens(estimated_prompt)
        self._token_usage.add_completion_tokens(estimated_completion)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve context using LLM-guided filesystem navigation.

        The agent navigates indexes and documents to gather relevant context.
        Note: top_k is ignored for filesystem RAG as the agent determines
        what to retrieve based on the query.

        Args:
            question: The question to retrieve context for
            top_k: Ignored for filesystem RAG

        Returns:
            RetrievedContext with chunks and trace information
        """
        agent = self._ensure_agent()

        start_time = time.time()

        # Execute query through agent
        response = agent.query(question)

        retrieval_time = time.time() - start_time

        return self._context_from_agent_response(question, response, retrieval_time)

    def _retrieve_only(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Perform retrieval without generation.

        For FilesystemRAG, retrieval and generation are interleaved by the agent,
        so this returns the full response but only extracts context.

        Args:
            question: The question to retrieve context for
            top_k: Ignored for filesystem RAG

        Returns:
            Dictionary with context and metadata
        """
        context = self.retrieve(question, top_k)
        return {
            "context": context.chunks,
            "metadata": {
                "sources": [c.source for c in context.chunk_details],
                "search_mode": context.trace.steps[0]
                .get("metadata", {})
                .get("search_mode", "unknown")
                if context.trace.steps
                else "unknown",
            },
        }

    def _generate_only(self, question: str, context_chunks: list[str]) -> str:
        """Generate answer from context.

        For FilesystemRAG, the agent typically generates the answer during retrieval.
        This method can be used for re-generation with different context.

        Args:
            question: The question to answer
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer text
        """
        # FilesystemRAG doesn't have a separate generation step
        # The agent generates during query. For re-generation, we'd need
        # to call the agent's LLM directly.
        if self._agent is None:
            raise ValueError("Agent not initialized")

        # Build context prompt and call agent's LLM
        context_text = "\n\n".join([f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""Based on the following context gathered from the filesystem, answer the question.

Context:
{context_text}

Question: {question}

Answer:"""

        # Use the agent to answer (simplified - ideally we'd call LLM directly)
        response = self._agent.query(f"Using this context: {context_text}\n\nAnswer: {question}")

        # Estimate token usage
        estimated_prompt_tokens = len(prompt) // 4
        estimated_completion_tokens = len(response.answer) // 4

        self._token_usage.add_prompt_tokens(estimated_prompt_tokens)
        self._token_usage.add_completion_tokens(estimated_completion_tokens)

        return response.answer

    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer:
        """Generate answer from retrieved context.

        Args:
            question: The question to answer
            context: Previously retrieved context

        Returns:
            GeneratedAnswer with text and token usage
        """
        start_time = time.time()

        answer = self._generate_only(question, context.chunks)

        generation_time = time.time() - start_time

        return GeneratedAnswer(
            text=answer,
            generation_time=generation_time,
            prompt_tokens=self._token_usage.prompt_tokens,
            completion_tokens=self._token_usage.completion_tokens,
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
        # Reset token usage for this query
        self.reset_token_usage()

        agent = self._ensure_agent()

        # Execute query
        response = agent.query(question)

        # Track metrics
        self._track_agent_response(question, response)

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
                "context_sources": response.metadata.get("context_sources", []),
                "prefetch_candidates": response.metadata.get("prefetch_candidates", []),
                "token_usage": self._token_usage.to_dict(),
            },
        }

    def query_with_trace(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query with a single agent pass and an aligned retrieval trace.

        Filesystem RAG interleaves retrieval and generation in the same agent
        loop. The base implementation would call retrieve() and then generate(),
        which invokes the agent twice and can produce an answer that does not
        match the stored trace.
        """
        self.reset_token_usage()
        agent = self._ensure_agent()

        start_time = time.time()
        response = agent.query(question)
        elapsed = time.time() - start_time

        self._track_agent_response(question, response)
        context = self._context_from_agent_response(question, response, elapsed)

        return {
            "answer": response.answer,
            "context": context.chunks,
            "metadata": {
                "retrieval_time": context.retrieval_time,
                "generation_time": 0.0,
                "token_usage": self._token_usage.to_dict(),
                "chunks_retrieved": len(context.chunks),
                "files_read": response.metadata.get("files_read", []),
                "tool_calls": response.metadata.get("tool_calls", 0),
                "search_mode": response.metadata.get("search_mode", "unknown"),
                "iterations": response.metadata.get("iterations", 0),
                "context_sources": response.metadata.get("context_sources", []),
                "prefetch_candidates": response.metadata.get("prefetch_candidates", []),
            },
            "retrieval_trace": context.trace.to_dict(),
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
                "token_usage": self._token_usage.to_dict(),
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
            "token_usage": self.get_token_usage().to_dict(),
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
        return (Path(self.prepared_path) / "_meta" / "statistics.json").exists()

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
