"""Neo4j-based Graph RAG implementation using neo4j-graphrag package."""

import time
from typing import Any

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.graph_rag.indexer import GraphIndexer


class Neo4jGraphRAG(BaseRAG):
    """RAG implementation using Neo4j graph database with hybrid retrieval."""

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        vector_index_name: str = "chunk_embeddings",
    ) -> None:
        """Initialize Neo4j Graph RAG.

        Args:
            neo4j_uri: Neo4j connection URI (defaults to settings)
            neo4j_username: Neo4j username (defaults to settings)
            neo4j_password: Neo4j password (defaults to settings)
            vector_index_name: Name of the vector index to use
        """
        super().__init__("Neo4j Graph RAG")

        # Use settings as defaults
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.neo4j_username = neo4j_username or settings.neo4j_username
        self.neo4j_password = neo4j_password or settings.neo4j_password
        self.vector_index_name = vector_index_name

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)
        )

        # Initialize embedder and LLM
        self.embedder = OpenAIEmbeddings(model=settings.embedding_model)

        # LLM configuration for answer generation
        llm_params: dict[str, Any] = {}
        # Only add temperature for models that support it
        if "nano" not in settings.openai_model.lower():
            llm_params["temperature"] = 0.2

        self.llm = OpenAILLM(
            model_name=settings.openai_model,
            model_params=llm_params,
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

    def __del__(self) -> None:
        """Close Neo4j driver on cleanup."""
        if hasattr(self, "driver"):
            self.driver.close()

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
        indexer = GraphIndexer(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            llm_model=settings.openai_model,
            embedding_model=settings.embedding_model,
        )

        print(f"\nIndexing documents from: {documents_path}")
        stats = indexer.index_documents(documents_path)

        print("\n=== Graph Indexing Complete ===")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"Node label distribution: {stats['node_labels']}")
        print("================================\n")

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using graph traversal and vector similarity.

        Args:
            question: The question to answer
            top_k: Number of top chunks to retrieve

        Returns:
            Dictionary containing answer, context, and metadata
        """
        start_time = time.time()
        self._total_queries += 1

        try:
            # Use GraphRAG pipeline to retrieve and generate
            response = self.rag_pipeline.search(
                query_text=question,
                retriever_config={"top_k": top_k},
                return_context=True,
            )

            retrieval_time = time.time() - start_time
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

            return {
                "answer": response.answer,
                "context": context_chunks,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "chunks_retrieved": len(context_chunks),
                    "graph_enhanced": True,
                },
            }

        except Exception as e:
            # Fallback response on error
            retrieval_time = time.time() - start_time
            self._retrieval_times.append(retrieval_time)

            return {
                "answer": f"Error querying graph RAG: {str(e)}",
                "context": [],
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "chunks_retrieved": 0,
                    "error": str(e),
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
            **graph_stats,
        }

    def _get_graph_statistics(self) -> dict[str, Any]:
        """Get current graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]  # type: ignore[index]

                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
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
