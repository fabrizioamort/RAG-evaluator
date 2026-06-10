"""Graph indexer for building knowledge graphs from documents using neo4j-graphrag."""

import asyncio
from pathlib import Path
from typing import Any

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import (
    DEFAULT_CHUNK_NODE_LABEL,
    DEFAULT_DOCUMENT_NODE_LABEL,
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm.openai_llm import OpenAILLM

from rag_evaluator.common.document_loaders import create_loader
from rag_evaluator.common.indexing import CheckpointStore, SourceDocument, discover_source_documents
from rag_evaluator.rag_implementations.graph_rag.neo4j_connection import (
    create_verified_neo4j_driver,
)


class GraphIndexer:
    """Indexes documents into Neo4j knowledge graph with dynamic schema inference."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm_model: str,
        embedding_model: str,
        vector_index_name: str = "chunk_embeddings",
        label_prefix: str | None = None,
        llm_client_kwargs: dict[str, str] | None = None,
        embedding_client_kwargs: dict[str, str] | None = None,
    ) -> None:
        """Initialize the graph indexer.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            llm_model: LLM model name for entity extraction
            embedding_model: Embedding model name
            vector_index_name: Name of the Neo4j vector index
            label_prefix: Optional prefix for Neo4j labels for isolation
            llm_client_kwargs: OpenAI client kwargs (api_key/base_url) for the LLM
            embedding_client_kwargs: OpenAI client kwargs for the embedder
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.vector_index_name = vector_index_name
        self.label_prefix = label_prefix
        self.llm_client_kwargs = llm_client_kwargs or {}
        self.embedding_client_kwargs = embedding_client_kwargs or {}
        self.chunk_label = self._prefix_label(DEFAULT_CHUNK_NODE_LABEL)
        self.document_label = self._prefix_label(DEFAULT_DOCUMENT_NODE_LABEL)

        # Initialize Neo4j driver and validate connectivity/auth immediately.
        self.driver = create_verified_neo4j_driver(neo4j_uri, neo4j_username, neo4j_password)

    def __del__(self) -> None:
        """Close Neo4j driver on cleanup."""
        if hasattr(self, "driver"):
            self.driver.close()

    def _prefix_label(self, label: str) -> str:
        if not self.label_prefix:
            return label
        if label.startswith(f"{self.label_prefix}_"):
            return label
        return f"{self.label_prefix}_{label}"

    def _build_schema(self) -> dict[str, Any]:
        base_node_types = [
            "Entity",
            "Concept",
            "Person",
            "Organization",
            "Location",
            "Event",
        ]
        node_types = [self._prefix_label(node_type) for node_type in base_node_types]
        relationship_types = [
            "RELATED_TO",
            "MENTIONS",
            "PART_OF",
            "ASSOCIATED_WITH",
            "OCCURS_IN",
        ]
        patterns = [
            (
                self._prefix_label("Entity"),
                "RELATED_TO",
                self._prefix_label("Entity"),
            ),
            (
                self._prefix_label("Concept"),
                "RELATED_TO",
                self._prefix_label("Concept"),
            ),
            (
                self._prefix_label("Person"),
                "ASSOCIATED_WITH",
                self._prefix_label("Organization"),
            ),
            (
                self._prefix_label("Event"),
                "OCCURS_IN",
                self._prefix_label("Location"),
            ),
        ]
        return {
            "node_types": node_types,
            "relationship_types": relationship_types,
            "patterns": patterns,
        }

    def _build_lexical_graph_config(self) -> LexicalGraphConfig | None:
        if not self.label_prefix:
            return None
        return LexicalGraphConfig(
            document_node_label=self.document_label,
            chunk_node_label=self.chunk_label,
        )

    def _create_vector_indexes(self) -> None:
        """Create vector indexes for semantic search on Chunk nodes."""
        try:
            # Determine embedding dimensions based on model
            dimensions = 1536 if "3-small" in self.embedding_model else 3072

            create_vector_index(
                self.driver,
                name=self.vector_index_name,
                label=self.chunk_label,
                embedding_property="embedding",
                dimensions=dimensions,
                similarity_fn="cosine",
                fail_if_exists=False,
            )
            print(
                f"Created vector index '{self.vector_index_name}' "
                f"for label '{self.chunk_label}' with {dimensions} dimensions"
            )
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")

    async def _process_document_with_retry(
        self,
        kg_builder: SimpleKGPipeline,
        text: str,
        document_metadata: dict[str, Any] | None = None,
        max_retries: int = 3,
        delay: float = 2.0,
    ) -> Any:
        """Process a single document text with retry logic for rate limits.

        Args:
            kg_builder: The initialized SimpleKGPipeline
            text: Text content to process
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds

        Returns:
            Result from kg_builder
        """
        for attempt in range(max_retries + 1):
            try:
                if document_metadata is not None:
                    try:
                        return await kg_builder.run_async(
                            text=text,
                            document_metadata=document_metadata,
                        )
                    except TypeError:
                        return await kg_builder.run_async(text=text)
                return await kg_builder.run_async(text=text)
            except Exception as e:
                # Check for rate limit error messages in the exception string
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        wait_time = delay * (2**attempt)  # Exponential backoff
                        print(
                            f"Rate limit hit. Retrying in {wait_time:.1f}s (Attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                # If not rate limit or max retries reached, raise
                raise e

    def index_documents(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore | None = None,
    ) -> dict[str, Any]:
        """Index documents into Neo4j knowledge graph.

        Args:
            documents_path: Path to directory containing documents

        Returns:
            Dictionary with indexing statistics
        """
        sources = discover_source_documents(documents_path)
        documents_to_process: list[tuple[SourceDocument, str]] = []
        doc_sources = []

        for source in sources:
            checkpoint = checkpoint_store.ensure_document(source) if checkpoint_store else None
            if (
                checkpoint
                and checkpoint.status == "completed"
                and self._document_marker_exists(source.doc_key, source.checksum)
            ):
                doc_sources.append(source.source_path)
                continue

            if checkpoint and checkpoint.status in {"building", "failed"}:
                self._delete_document_artifacts(source.doc_key)

            try:
                loader = create_loader(source.source_path)
                doc = loader.load(source.source_path)
                documents_to_process.append((source, doc.content))
                doc_sources.append(doc.source)
                print(f"Loaded: {Path(source.source_path).name}")
            except Exception as e:
                print(f"Warning: Failed to load {Path(source.source_path).name}: {e}")

        if not documents_to_process and not doc_sources:
            raise ValueError(f"No documents found in {documents_path}")

        print(f"\nLoaded {len(doc_sources)} documents")

        # Initialize LLM and embedder
        llm_params: dict[str, Any] = {"response_format": {"type": "json_object"}}

        if "nano" not in self.llm_model.lower() and "o1" not in self.llm_model.lower():
            llm_params["temperature"] = 0

        llm = OpenAILLM(
            model_name=self.llm_model,
            model_params=llm_params,
            **self.llm_client_kwargs,
        )

        embedder = OpenAIEmbeddings(model=self.embedding_model, **self.embedding_client_kwargs)

        schema = self._build_schema()
        lexical_graph_config = self._build_lexical_graph_config()

        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=self.driver,
            embedder=embedder,
            schema=schema,  # type: ignore[arg-type]
            lexical_graph_config=lexical_graph_config,
            from_pdf=False,
            on_error="IGNORE",
            perform_entity_resolution=True,
        )

        # Run async pipeline wrapper
        async def process_all_documents() -> None:
            print("\nBuilding knowledge graph iteratively...")
            total = len(documents_to_process)

            for i, (source, text) in enumerate(documents_to_process):
                print(f"Processing document {i + 1}/{total} ({len(text)} chars)...")
                try:
                    if checkpoint_store:
                        checkpoint_store.start_document(source.doc_key)
                    metadata = {
                        "index_id": self.label_prefix or self.vector_index_name,
                        "doc_key": source.doc_key,
                        "checksum": source.checksum,
                        "source_path": source.source_path,
                    }
                    await self._process_document_with_retry(kg_builder, text, metadata)
                    self._write_document_marker(source)
                    if checkpoint_store:
                        checkpoint_store.complete_document(source.doc_key, 1)
                        checkpoint_store.update_progress(
                            i + 1,
                            total,
                            {"document": source.relative_path},
                        )
                    # Add small delay between successful calls to be polite to the API
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"Error processing document {i + 1}: {e}")
                    if checkpoint_store:
                        checkpoint_store.fail_document(source.doc_key, str(e))
                    # Continue with next document instead of crashing entire pipeline
                    continue

        try:
            asyncio.run(process_all_documents())
            print("Knowledge graph construction completed.")

            # Create vector indexes after graph is built
            self._create_vector_indexes()

            # Get graph statistics
            stats = self._get_graph_statistics()

            return {
                "documents_processed": len(doc_sources),
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
                label_result = session.run(
                    """
                    MATCH (n)
                    WHERE any(label IN labels(n) WHERE label STARTS WITH $label_prefix)
                    RETURN labels(n)[0] as label, count(*) as count
                    ORDER BY count DESC
                    """,
                    params,
                )
            else:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                # Get node labels distribution
                label_result = session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(*) as count
                    ORDER BY count DESC
                    """
                )

            node_count = node_result.single()["count"]  # type: ignore[index]
            rel_count = rel_result.single()["count"]  # type: ignore[index]
            label_dist = {record["label"]: record["count"] for record in label_result}

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "node_labels": label_dist,
            }

    def _label(self, label: str) -> str:
        return f"`{label.replace('`', '')}`"

    def _document_marker_exists(self, doc_key: str, checksum: str) -> bool:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (d:{self._label(self.document_label)})
                WHERE d.doc_key = $doc_key AND d.checksum = $checksum
                RETURN count(d) AS count
                """,
                {"doc_key": doc_key, "checksum": checksum},
            )
            return bool(result.single()["count"])  # type: ignore[index]

    def _write_document_marker(self, source: SourceDocument) -> None:
        with self.driver.session() as session:
            session.run(
                f"""
                MERGE (d:{self._label(self.document_label)} {{doc_key: $doc_key}})
                SET d.checksum = $checksum,
                    d.source_path = $source_path,
                    d.index_id = $index_id,
                    d.index_status = 'completed'
                """,
                {
                    "doc_key": source.doc_key,
                    "checksum": source.checksum,
                    "source_path": source.source_path,
                    "index_id": self.label_prefix or self.vector_index_name,
                },
            )

    def _delete_document_artifacts(self, doc_key: str) -> None:
        with self.driver.session() as session:
            if self.label_prefix:
                session.run(
                    """
                    MATCH (n)
                    WHERE n.doc_key = $doc_key
                    AND any(label IN labels(n) WHERE label STARTS WITH $label_prefix)
                    DETACH DELETE n
                    """,
                    {"doc_key": doc_key, "label_prefix": self.label_prefix},
                )
            else:
                session.run(
                    "MATCH (n) WHERE n.doc_key = $doc_key DETACH DELETE n",
                    {"doc_key": doc_key},
                )

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph database."""
        with self.driver.session() as session:
            if self.label_prefix:
                session.run(
                    "MATCH (n) "
                    "WHERE any(label IN labels(n) WHERE label STARTS WITH $label_prefix) "
                    "DETACH DELETE n",
                    {"label_prefix": self.label_prefix},
                )
                print(f"Graph database cleared for prefix: {self.label_prefix}")
            else:
                session.run("MATCH (n) DETACH DELETE n")
                print("Graph database cleared")
