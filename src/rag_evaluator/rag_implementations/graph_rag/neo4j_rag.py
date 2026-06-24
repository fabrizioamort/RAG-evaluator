"""Neo4j-based Graph RAG implementation using neo4j-graphrag package."""

import time
from typing import Any, cast

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.indexing import CheckpointStore
from rag_evaluator.common.openai_client import (
    embedding_openai_kwargs,
    llm_openai_kwargs,
)
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.graph_rag.indexer import GraphIndexer
from rag_evaluator.rag_implementations.graph_rag.neo4j_connection import (
    Neo4jConnectionError,
    format_neo4j_connection_error,
    resolve_neo4j_connection_params,
)


class Neo4jGraphRAG(BaseRAG):
    """RAG implementation using Neo4j graph database with hybrid retrieval."""

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        vector_index_name: str = "chunk_embeddings",
        config: RAGConfig | None = None,
        label_prefix: str | None = None,
    ) -> None:
        """Initialize Neo4j Graph RAG.

        Args:
            neo4j_uri: Neo4j connection URI (defaults to settings)
            neo4j_username: Neo4j username (defaults to settings)
            neo4j_password: Neo4j password (defaults to settings)
            vector_index_name: Name of the vector index to use
            config: Optional RAGConfig for LLM and embedding configuration
            label_prefix: Optional prefix for Neo4j labels and index names for isolation
        """
        super().__init__("Neo4j Graph RAG", config=config)

        self.neo4j_uri, self.neo4j_username, self.neo4j_password = (
            resolve_neo4j_connection_params(
                neo4j_uri,
                neo4j_username,
                neo4j_password,
                default_uri=settings.neo4j_uri,
                default_username=settings.neo4j_username,
                default_password=settings.neo4j_password,
            )
        )
        self.label_prefix = label_prefix
        if label_prefix and not vector_index_name.startswith(f"{label_prefix}_"):
            self.vector_index_name = f"{label_prefix}_{vector_index_name}"
        else:
            self.vector_index_name = vector_index_name

        # Initialize Neo4j driver and fail early with a clear message if unreachable.
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password),
        )
        try:
            if hasattr(self.driver, "verify_connectivity"):
                self.driver.verify_connectivity()
            else:
                with self.driver.session() as session:
                    session.run("RETURN 1").consume()
        except Exception as exc:
            self.driver.close()
            raise Neo4jConnectionError(
                format_neo4j_connection_error(
                    exc, uri=self.neo4j_uri, username=self.neo4j_username
                )
            ) from exc

        # Get model from config or settings
        embedding_model = self.config.embedding_model or settings.embedding_model
        llm_model = self.config.llm_model or settings.openai_model

        # OpenAI-compatible client kwargs resolved from config (provider/base_url/
        # api_key); generation and embeddings can target different endpoints.
        llm_kwargs = llm_openai_kwargs(self.config)
        embed_kwargs = embedding_openai_kwargs(self.config)

        # Initialize embedder and LLM
        self.embedder = cast(Any, OpenAIEmbeddings)(model=embedding_model, **embed_kwargs)

        # LLM configuration for answer generation
        llm_params: dict[str, Any] = {}
        # Only add temperature for models that support it
        if "nano" not in llm_model.lower():
            llm_params["temperature"] = 0.2

        self.llm = cast(Any, OpenAILLM)(
            model_name=llm_model,
            model_params=llm_params,
            **llm_kwargs,
        )

        # Custom Cypher query for graph traversal retrieval
        # This retrieves chunks and expands to related entities via graph relationships
        self.retrieval_query = """
        // Get related entities and their connections
        OPTIONAL MATCH (node)-[:MENTIONS]->(entity)
        OPTIONAL MATCH (entity)-[rel:RELATED_TO|ASSOCIATED_WITH|PART_OF]-(related)

        // Return the chunk text with graph context
        RETURN
            node.text AS text,
            collect(DISTINCT entity.name) AS entities,
            collect(DISTINCT related.name) AS related_entities,
            collect(DISTINCT type(rel)) AS relationship_types
        """

        # Retriever and RAG pipeline will be initialized lazily when needed
        self._retriever: VectorCypherRetriever | None = None
        self._rag_pipeline: GraphRAG | None = None

        # Metrics tracking
        self._retrieval_times: list[float] = []
        self._total_queries = 0

    def close(self) -> None:
        """Close Neo4j driver."""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            self.driver = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """Close Neo4j driver on cleanup."""
        self.close()

    @property
    def retriever(self) -> VectorCypherRetriever:
        """Lazily initialize retriever when first accessed."""
        if self._retriever is None:
            self._retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name=self.vector_index_name,
                retrieval_query=self.retrieval_query,
                embedder=self.embedder,
            )
        return self._retriever

    @property
    def rag_pipeline(self) -> GraphRAG:
        """Lazily initialize RAG pipeline when first accessed."""
        if self._rag_pipeline is None:
            self._rag_pipeline = GraphRAG(retriever=self.retriever, llm=self.llm)
        return self._rag_pipeline

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents as knowledge graph in Neo4j.

        Args:
            documents_path: Path to the directory containing documents
        """
        llm_model = (
            self.config.parameters.get("extraction_model")
            or self.config.llm_model
            or settings.openai_model
        )
        embedding_model = self.config.embedding_model or settings.embedding_model

        indexer = GraphIndexer(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            llm_model=llm_model,
            embedding_model=embedding_model,
            vector_index_name=self.vector_index_name,
            label_prefix=self.label_prefix,
            llm_client_kwargs=llm_openai_kwargs(self.config),
            embedding_client_kwargs=embedding_openai_kwargs(self.config),
        )

        print(f"\nIndexing documents from: {documents_path}")
        stats = indexer.index_documents(documents_path)

        print("\n=== Graph Indexing Complete ===")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"Node label distribution: {stats['node_labels']}")
        print("================================\n")

    def prepare_documents_resumable(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Prepare and index documents as a resumable Neo4j knowledge graph."""
        llm_model = (
            self.config.parameters.get("extraction_model")
            or self.config.llm_model
            or settings.openai_model
        )
        embedding_model = self.config.embedding_model or settings.embedding_model

        indexer = GraphIndexer(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            llm_model=llm_model,
            embedding_model=embedding_model,
            vector_index_name=self.vector_index_name,
            label_prefix=self.label_prefix,
            llm_client_kwargs=llm_openai_kwargs(self.config),
            embedding_client_kwargs=embedding_openai_kwargs(self.config),
        )

        print(f"\nIndexing documents from: {documents_path}")
        stats = indexer.index_documents(documents_path, checkpoint_store=checkpoint_store)

        print("\n=== Graph Indexing Complete ===")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"Node label distribution: {stats['node_labels']}")
        print("================================\n")

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve context using graph traversal and vector similarity.

        Args:
            question: The question to retrieve context for
            top_k: Number of top chunks to retrieve

        Returns:
            RetrievedContext with chunks and trace information
        """
        start_time = time.time()

        try:
            # Use retriever directly for just retrieval
            retriever_result = self.retriever.search(
                query_text=question,
                top_k=top_k,
            )

            retrieval_time = time.time() - start_time
            with self._metrics_lock:
                self._retrieval_times.append(retrieval_time)

            # Extract context from retriever results
            context_chunks = []
            chunk_details = []

            if retriever_result and retriever_result.items:
                for i, item in enumerate(retriever_result.items):
                    # Include both chunk text and graph metadata
                    context_text = item.content
                    metadata = item.metadata or {}

                    # Build enriched context with entity information
                    entities = metadata.get("entities", [])
                    related = metadata.get("related_entities", [])

                    enriched_text = context_text
                    if entities:
                        enriched_text += f"\n[Entities: {', '.join(entities)}]"
                    if related:
                        enriched_text += f"\n[Related: {', '.join(related)}]"

                    context_chunks.append(enriched_text)

                    # Determine source from metadata
                    source = metadata.get("source", "graph_node")

                    chunk_details.append(
                        RetrievedChunk(
                            content=context_text,
                            document_id=source,
                            chunk_id=f"graph_chunk_{i}",
                            score=item.score if hasattr(item, "score") else 1.0 - (i * 0.1),
                            rank=i,
                            source=source,
                            metadata={
                                "entities": entities,
                                "related_entities": related,
                                "relationship_types": metadata.get("relationship_types", []),
                            },
                        )
                    )

            # Build trace
            trace = RetrievalTrace(
                strategy="graph",
                total_duration_ms=retrieval_time * 1000,
            )
            trace.add_step(
                step_type="vector_search",
                input_data={"query": question, "top_k": top_k},
                output_refs=[c.chunk_id for c in chunk_details],
                duration_ms=retrieval_time * 500,  # Approximate
                metadata={"index": self.vector_index_name},
            )
            trace.add_step(
                step_type="graph_expansion",
                input_data={"cypher_query": "entity_relationship_expansion"},
                output_refs=[c.chunk_id for c in chunk_details],
                duration_ms=retrieval_time * 500,  # Approximate
                metadata={
                    "patterns": ["MENTIONS", "RELATED_TO", "ASSOCIATED_WITH", "PART_OF"],
                },
            )
            trace.retrieved_chunks = chunk_details

            return RetrievedContext(
                chunks=context_chunks,
                chunk_details=chunk_details,
                trace=trace,
                retrieval_time=retrieval_time,
            )

        except Exception as e:
            retrieval_time = time.time() - start_time
            with self._metrics_lock:
                self._retrieval_times.append(retrieval_time)

            # Return empty context with error in trace
            trace = RetrievalTrace(
                strategy="graph",
                total_duration_ms=retrieval_time * 1000,
            )
            trace.add_step(
                step_type="error",
                input_data={"query": question},
                output_refs=[],
                duration_ms=retrieval_time * 1000,
                metadata={"error": str(e)},
            )

            return RetrievedContext(
                chunks=[],
                chunk_details=[],
                trace=trace,
                retrieval_time=retrieval_time,
            )

    def _retrieve_only(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Perform retrieval without generation.

        Args:
            question: The question to retrieve context for
            top_k: Number of top chunks to retrieve

        Returns:
            Dictionary with context and metadata
        """
        context = self.retrieve(question, top_k)
        return {
            "context": context.chunks,
            "metadata": {
                "sources": [c.source for c in context.chunk_details],
                "graph_enhanced": True,
            },
        }

    def _generate_only(self, question: str, context_chunks: list[str]) -> str:
        """Generate answer from context without retrieval.

        Args:
            question: The question to answer
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer text
        """
        # Build context for LLM
        context_text = "\n\n".join([f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""Answer the following question based only on the provided context which includes graph-derived entity relationships. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Use the LLM directly
        try:
            response = self.llm.invoke(prompt)

            # Note: neo4j-graphrag LLM doesn't provide token counts directly
            # We estimate based on character count
            estimated_prompt_tokens = len(prompt) // 4
            estimated_completion_tokens = len(response.content) // 4 if response.content else 0

            self._token_usage.add_prompt_tokens(estimated_prompt_tokens)
            self._token_usage.add_completion_tokens(estimated_completion_tokens)

            return response.content or "No answer generated"

        except Exception as e:
            return f"Error generating answer: {str(e)}"

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
        """Query using graph traversal and vector similarity.

        Args:
            question: The question to answer
            top_k: Number of top chunks to retrieve

        Returns:
            Dictionary containing answer, context, and metadata
        """
        # Reset token usage for this query
        self.reset_token_usage()

        start_time = time.time()
        with self._metrics_lock:
            self._total_queries += 1

        try:
            # Use GraphRAG pipeline to retrieve and generate
            response = self.rag_pipeline.search(
                query_text=question,
                retriever_config={"top_k": top_k},
                return_context=True,
            )

            retrieval_time = time.time() - start_time
            with self._metrics_lock:
                self._retrieval_times.append(retrieval_time)

            # Extract context from retriever results
            context_chunks = []
            if response.retriever_result and response.retriever_result.items:
                for item in response.retriever_result.items:
                    # Include both chunk text and graph metadata
                    context_text = item.content
                    metadata = item.metadata

                    # Add entity information if available
                    if metadata and "entities" in metadata and metadata["entities"]:
                        entities_str = ", ".join(metadata["entities"])
                        context_text += f"\n[Entities: {entities_str}]"

                    if metadata and "related_entities" in metadata and metadata["related_entities"]:
                        related_str = ", ".join(metadata["related_entities"])
                        context_text += f"\n[Related: {related_str}]"

                    context_chunks.append(context_text)

            # Estimate token usage
            prompt_estimate = len(question) // 4
            completion_estimate = len(response.answer) // 4 if response.answer else 0
            self._token_usage.add_prompt_tokens(prompt_estimate)
            self._token_usage.add_completion_tokens(completion_estimate)

            return {
                "answer": response.answer,
                "context": context_chunks,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "chunks_retrieved": len(context_chunks),
                    "graph_enhanced": True,
                    "token_usage": self._token_usage.to_dict(),
                },
            }

        except Exception as e:
            # Fallback response on error
            retrieval_time = time.time() - start_time
            with self._metrics_lock:
                self._retrieval_times.append(retrieval_time)

            return {
                "answer": f"Error querying graph RAG: {str(e)}",
                "context": [],
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "chunks_retrieved": 0,
                    "error": str(e),
                    "token_usage": self._token_usage.to_dict(),
                },
            }

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        avg_retrieval_time = (
            sum(self._retrieval_times) / len(self._retrieval_times)
            if self._retrieval_times
            else 0.0
        )

        # Get graph statistics
        graph_stats = self._get_graph_statistics()

        return {
            "avg_retrieval_time": avg_retrieval_time,
            "total_queries": self._total_queries,
            "token_usage": self.get_token_usage().to_dict(),
            **graph_stats,
        }

    def _get_graph_statistics(self) -> dict[str, Any]:
        """Get current graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        try:
            with self.driver.session() as session:
                if self.label_prefix:
                    params = {"label_prefix": self.label_prefix}
                    node_result = session.run(
                        "MATCH (n) "
                        "WHERE any(label IN labels(n) WHERE label STARTS WITH $label_prefix) "
                        "RETURN count(n) as count",
                        params,
                    )
                    rel_result = session.run(
                        "MATCH (n)-[r]->(m) "
                        "WHERE any(label IN labels(n) WHERE label STARTS WITH $label_prefix) "
                        "AND any(label IN labels(m) WHERE label STARTS WITH $label_prefix) "
                        "RETURN count(r) as count",
                        params,
                    )
                else:
                    # Count nodes
                    node_result = session.run("MATCH (n) RETURN count(n) as count")
                    # Count relationships
                    rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")

                node_count = node_result.single()["count"]  # type: ignore[index]
                rel_count = rel_result.single()["count"]  # type: ignore[index]

                return {
                    "total_nodes": node_count,
                    "total_relationships": rel_count,
                }
        except Exception:
            return {
                "total_nodes": 0,
                "total_relationships": 0,
            }
