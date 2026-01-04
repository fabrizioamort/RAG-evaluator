"""Graph indexer for building knowledge graphs from documents using neo4j-graphrag."""

import asyncio
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm.openai_llm import OpenAILLM

from rag_evaluator.common.document_loaders import create_loader


class GraphIndexer:
    """Indexes documents into Neo4j knowledge graph with dynamic schema inference."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm_model: str,
        embedding_model: str,
    ) -> None:
        """Initialize the graph indexer.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            llm_model: LLM model name for entity extraction
            embedding_model: Embedding model name
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    def __del__(self) -> None:
        """Close Neo4j driver on cleanup."""
        if hasattr(self, "driver"):
            self.driver.close()

    def _create_vector_indexes(self) -> None:
        """Create vector indexes for semantic search on Chunk nodes."""
        try:
            # Determine embedding dimensions based on model
            dimensions = 1536 if "3-small" in self.embedding_model else 3072

            create_vector_index(
                self.driver,
                name="chunk_embeddings",
                label="Chunk",
                embedding_property="embedding",
                dimensions=dimensions,
                similarity_fn="cosine",
                fail_if_exists=False,
            )
            print(f"Created vector index 'chunk_embeddings' with {dimensions} dimensions")
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")

    def index_documents(self, documents_path: str) -> dict[str, Any]:
        """Index documents into Neo4j knowledge graph.

        Args:
            documents_path: Path to directory containing documents

        Returns:
            Dictionary with indexing statistics
        """
        docs_path = Path(documents_path)

        if not docs_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")

        # Validate extensions
        valid_extensions = {".txt", ".pdf", ".docx"}
        all_text_content = []
        doc_sources = []

        # Load documents
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in valid_extensions and file_path.is_file():
                try:
                    loader = create_loader(str(file_path))
                    doc = loader.load(str(file_path))
                    all_text_content.append(doc.content)
                    doc_sources.append(doc.source)
                    print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")

        if not all_text_content:
            raise ValueError(f"No documents found in {documents_path}")

        # Combine all documents into single text for processing
        combined_text = "\n\n".join(all_text_content)
        print(f"\nLoaded {len(all_text_content)} documents")

        # Initialize LLM and embedder
        llm = OpenAILLM(
            model_name=self.llm_model,
            model_params={"temperature": 0, "response_format": {"type": "json_object"}},
        )

        embedder = OpenAIEmbeddings(model=self.embedding_model)

        # Define flexible schema for dynamic entity extraction
        # The LLM will infer appropriate node types and relationships
        schema = {
            "node_types": ["Entity", "Concept", "Person", "Organization", "Location", "Event"],
            "relationship_types": [
                "RELATED_TO",
                "MENTIONS",
                "PART_OF",
                "ASSOCIATED_WITH",
                "OCCURS_IN",
            ],
            "patterns": [
                ("Entity", "RELATED_TO", "Entity"),
                ("Concept", "RELATED_TO", "Concept"),
                ("Person", "ASSOCIATED_WITH", "Organization"),
                ("Event", "OCCURS_IN", "Location"),
            ],
        }

        # Create knowledge graph pipeline
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=self.driver,
            embedder=embedder,
            schema=schema,  # type: ignore[arg-type]
            from_pdf=False,
            on_error="IGNORE",
            perform_entity_resolution=True,
        )

        try:
            # Run async pipeline
            print("\nBuilding knowledge graph (this may take a few minutes)...")
            result = asyncio.run(kg_builder.run_async(text=combined_text))
            print(f"Knowledge graph construction completed: {result}")

            # Create vector indexes after graph is built
            self._create_vector_indexes()

            # Get graph statistics
            stats = self._get_graph_statistics()

            return {
                "documents_processed": len(all_text_content),
                "sources": doc_sources,
                **stats,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to build knowledge graph: {e}") from e

    def _get_graph_statistics(self) -> dict[str, Any]:
        """Get statistics about the created graph.

        Returns:
            Dictionary with graph statistics
        """
        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]  # type: ignore[index]

            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]  # type: ignore[index]

            # Get node labels distribution
            label_result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
                """
            )
            label_dist = {record["label"]: record["count"] for record in label_result}

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "node_labels": label_dist,
            }

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Graph database cleared")
