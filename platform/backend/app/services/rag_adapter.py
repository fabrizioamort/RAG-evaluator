"""RAG adapter service for instantiating RAG implementations from config models.

This service provides a unified interface for creating and using RAG implementations
based on database configuration models, enabling the platform to work with all
supported RAG types.
"""

import sys
from pathlib import Path
from typing import Any

from app.config import settings
from app.models.rag_config import RAGConfig as RAGConfigModel
from app.utils.logging_config import get_logger

# Add src to path for importing rag_evaluator
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig  # noqa: E402
from rag_evaluator.common.provider_interfaces import (  # noqa: E402
    GeneratedAnswer,
    RetrievedContext,
)

logger = get_logger(__name__)


# Registry of RAG types to their implementation classes
RAG_TYPE_REGISTRY: dict[str, str] = {
    "vector_semantic": "rag_evaluator.rag_implementations.vector_semantic.chroma_rag.ChromaSemanticRAG",
    "vector_hybrid": "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.HybridSearchRAG",
    "graph_rag": "rag_evaluator.rag_implementations.graph_rag.neo4j_rag.Neo4jGraphRAG",
    "filesystem_rag": "rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag.FilesystemRAG",
}


# Parameter schemas for each RAG type
RAG_TYPE_PARAMETERS: dict[str, dict[str, Any]] = {
    "vector_semantic": {
        "properties": {
            "chunk_size": {
                "type": "integer",
                "default": 1000,
                "description": "Size of text chunks for indexing",
            },
            "chunk_overlap": {
                "type": "integer",
                "default": 200,
                "description": "Overlap between chunks",
            },
            "collection_name": {
                "type": "string",
                "default": "rag_documents",
                "description": "ChromaDB collection name",
            },
        },
    },
    "vector_hybrid": {
        "properties": {
            "chunk_size": {
                "type": "integer",
                "default": 500,
                "description": "Size of text chunks for indexing",
            },
            "chunk_overlap": {
                "type": "integer",
                "default": 50,
                "description": "Overlap between chunks",
            },
            "collection_name": {
                "type": "string",
                "default": "hybrid_rag",
                "description": "Qdrant collection name",
            },
        },
    },
    "graph_rag": {
        "properties": {
            "vector_index_name": {
                "type": "string",
                "default": "chunk_embeddings",
                "description": "Name of the Neo4j vector index",
            },
        },
    },
    "filesystem_rag": {
        "properties": {
            "word_threshold": {
                "type": "integer",
                "default": 1000,
                "description": "Word count threshold for LLM vs heuristic analysis",
            },
            "max_iterations": {
                "type": "integer",
                "default": 10,
                "description": "Maximum ReAct loop iterations per query",
            },
            "max_tool_calls": {
                "type": "integer",
                "default": 20,
                "description": "Maximum tool calls per query",
            },
            "max_file_reads": {
                "type": "integer",
                "default": 10,
                "description": "Maximum file reads per query",
            },
        },
    },
}


class RAGAdapterService:
    """Service for adapting database RAG configs to RAG implementations.

    This service provides methods to:
    - Create RAG instances from database config models
    - List available RAG types and their parameter schemas
    - Execute queries through RAG implementations
    - Manage index paths for knowledge bases
    """

    def __init__(self) -> None:
        """Initialize the RAG adapter service."""
        self._rag_instances: dict[str, BaseRAG] = {}

    def get_available_rag_types(self) -> list[dict[str, Any]]:
        """Get list of available RAG types with descriptions.

        Returns:
            List of RAG type info dictionaries.
        """
        return [
            {
                "type": "vector_semantic",
                "name": "Vector Semantic Search",
                "description": "ChromaDB-based semantic vector search using embeddings",
            },
            {
                "type": "vector_hybrid",
                "name": "Hybrid Search",
                "description": "Qdrant-based hybrid search combining dense and sparse vectors with RRF fusion",
            },
            {
                "type": "graph_rag",
                "name": "Graph RAG",
                "description": "Neo4j-based graph RAG with entity relationships and vector search",
            },
            {
                "type": "filesystem_rag",
                "name": "Filesystem RAG",
                "description": "LLM-guided agent that navigates a prepared filesystem structure",
            },
        ]

    def get_parameter_schema(self, rag_type: str) -> dict[str, Any]:
        """Get the parameter schema for a RAG type.

        Args:
            rag_type: The RAG type identifier.

        Returns:
            Parameter schema dictionary.

        Raises:
            ValueError: If the RAG type is not supported.
        """
        if rag_type not in RAG_TYPE_PARAMETERS:
            raise ValueError(f"Unknown RAG type: {rag_type}")
        return RAG_TYPE_PARAMETERS[rag_type]

    def _get_rag_class(self, rag_type: str) -> type[BaseRAG]:
        """Dynamically import and return the RAG class for a type.

        Args:
            rag_type: The RAG type identifier.

        Returns:
            The RAG implementation class.

        Raises:
            ValueError: If the RAG type is not supported.
            ImportError: If the RAG class cannot be imported.
        """
        if rag_type not in RAG_TYPE_REGISTRY:
            raise ValueError(f"Unknown RAG type: {rag_type}")

        module_path = RAG_TYPE_REGISTRY[rag_type]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            import importlib

            module = importlib.import_module(module_name)
            rag_class: type[BaseRAG] = getattr(module, class_name)
            return rag_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import RAG class: {module_path}", error=str(e))
            raise ImportError(f"Failed to import RAG class: {module_path}") from e

    def create_rag_from_config(
        self,
        config_model: RAGConfigModel,
        index_path: str | None = None,
    ) -> BaseRAG:
        """Create a RAG instance from a database config model.

        Args:
            config_model: The RAGConfig database model.
            index_path: Optional custom index path for the knowledge base.

        Returns:
            Configured RAG instance.

        Raises:
            ValueError: If the RAG type is not supported.
        """
        # Build RAGConfig from model
        rag_config = RAGConfig(
            name=config_model.name,
            parameters=config_model.parameters or {},
            storage_path=index_path or str(Path(settings.STORAGE_PATH) / "indexes"),
            llm_provider=config_model.llm_provider,
            llm_model=config_model.llm_model,
            llm_base_url=config_model.llm_base_url,
        )

        # Get the RAG class
        rag_class = self._get_rag_class(config_model.rag_type)

        # Build constructor kwargs based on RAG type
        kwargs: dict[str, Any] = {"config": rag_config}

        if config_model.rag_type == "vector_semantic":
            kwargs["collection_name"] = config_model.parameters.get(
                "collection_name", f"kb_{config_model.project_id}"
            )
            kwargs["persist_directory"] = index_path

        elif config_model.rag_type == "vector_hybrid":
            kwargs["collection_name"] = config_model.parameters.get(
                "collection_name", f"hybrid_{config_model.project_id}"
            )
            # Qdrant URL from settings or parameters
            kwargs["qdrant_url"] = config_model.parameters.get("qdrant_url")

        elif config_model.rag_type == "graph_rag":
            kwargs["vector_index_name"] = config_model.parameters.get(
                "vector_index_name", "chunk_embeddings"
            )
            # Neo4j connection from settings or parameters
            kwargs["neo4j_uri"] = config_model.parameters.get("neo4j_uri")
            kwargs["neo4j_username"] = config_model.parameters.get("neo4j_username")
            kwargs["neo4j_password"] = config_model.parameters.get("neo4j_password")

        elif config_model.rag_type == "filesystem_rag":
            kwargs["llm_model"] = config_model.llm_model
            kwargs["prepared_path"] = index_path or str(
                Path(settings.STORAGE_PATH) / "indexes" / f"fs_{config_model.project_id}"
            )
            kwargs["word_threshold"] = config_model.parameters.get("word_threshold", 1000)
            kwargs["max_iterations"] = config_model.parameters.get("max_iterations", 10)
            kwargs["max_tool_calls"] = config_model.parameters.get("max_tool_calls", 20)
            kwargs["max_file_reads"] = config_model.parameters.get("max_file_reads", 10)

        # Create and return the RAG instance
        logger.info(
            "Creating RAG instance",
            rag_type=config_model.rag_type,
            config_name=config_model.name,
        )

        return rag_class(**kwargs)

    def get_or_create_rag(
        self,
        config_model: RAGConfigModel,
        index_path: str | None = None,
        force_new: bool = False,
    ) -> BaseRAG:
        """Get an existing RAG instance or create a new one.

        Caches RAG instances by config ID for reuse.

        Args:
            config_model: The RAGConfig database model.
            index_path: Optional custom index path for the knowledge base.
            force_new: If True, create a new instance even if one exists.

        Returns:
            RAG instance.
        """
        cache_key = str(config_model.id)

        if not force_new and cache_key in self._rag_instances:
            logger.debug("Using cached RAG instance", config_id=cache_key)
            return self._rag_instances[cache_key]

        # Create new instance
        rag = self.create_rag_from_config(config_model, index_path)
        self._rag_instances[cache_key] = rag

        return rag

    def clear_cache(self, config_id: str | None = None) -> None:
        """Clear cached RAG instances.

        Args:
            config_id: If provided, only clear that specific instance.
        """
        if config_id:
            self._rag_instances.pop(config_id, None)
        else:
            self._rag_instances.clear()

    async def prepare_documents(
        self,
        rag: BaseRAG,
        documents_path: str,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Prepare documents for a RAG instance.

        Args:
            rag: The RAG instance.
            documents_path: Path to the documents directory.
            progress_callback: Optional callback for progress updates.

        Returns:
            Preparation metrics.
        """
        if progress_callback:
            rag.set_progress_callback(progress_callback)

        try:
            # Run in thread pool to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, rag.prepare_documents, documents_path)
            # Get metrics returns a dict[str, Any]
            metrics: dict[str, Any] = rag.get_metrics()
            return metrics
        except Exception as e:
            logger.error("Document preparation failed", error=str(e))
            raise

    async def query(
        self,
        rag: BaseRAG,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Execute a query on a RAG instance.

        Args:
            rag: The RAG instance.
            question: The question to answer.
            top_k: Number of documents to retrieve.

        Returns:
            Query result with answer, context, and metadata.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        result: dict[str, Any] = await loop.run_in_executor(None, rag.query, question, top_k)
        return result

    async def query_with_trace(
        self,
        rag: BaseRAG,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Execute a query with full retrieval trace.

        Args:
            rag: The RAG instance.
            question: The question to answer.
            top_k: Number of documents to retrieve.

        Returns:
            Query result with answer, context, metadata, and retrieval_trace.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        result: dict[str, Any] = await loop.run_in_executor(
            None, rag.query_with_trace, question, top_k
        )
        return result

    async def retrieve(
        self,
        rag: BaseRAG,
        question: str,
        top_k: int = 5,
    ) -> RetrievedContext:
        """Retrieve context without generation.

        Args:
            rag: The RAG instance.
            question: The question to retrieve context for.
            top_k: Number of documents to retrieve.

        Returns:
            RetrievedContext with chunks and trace.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, rag.retrieve, question, top_k)

    async def generate(
        self,
        rag: BaseRAG,
        question: str,
        context: RetrievedContext,
    ) -> GeneratedAnswer:
        """Generate answer from context.

        Args:
            rag: The RAG instance.
            question: The question to answer.
            context: Previously retrieved context.

        Returns:
            GeneratedAnswer with text and token usage.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, rag.generate, question, context)


# Singleton instance
_rag_adapter_service: RAGAdapterService | None = None


def get_rag_adapter_service() -> RAGAdapterService:
    """Get the RAG adapter service singleton.

    Returns:
        RAGAdapterService instance.
    """
    global _rag_adapter_service
    if _rag_adapter_service is None:
        _rag_adapter_service = RAGAdapterService()
    return _rag_adapter_service
