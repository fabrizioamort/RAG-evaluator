"""RAG adapter service for instantiating RAG implementations from config models.

This service provides a unified interface for creating and using RAG implementations
based on database configuration models, enabling the platform to work with all
supported RAG types.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievedContext,
)
from rag_evaluator.config import settings as core_settings
from rag_evaluator.rag_implementations.graph_rag.neo4j_connection import (
    resolve_neo4j_connection_params,
)
from rag_evaluator.rag_implementations.registry import (
    RAG_TYPES,
    get_parameter_schema,
    get_rag_class,
    split_parameters,
    validate_query_overrides,
)
from rag_evaluator.rag_implementations.rlm_rag.rlm_rag import rlm_config_from_rag_config

from app.config import settings
from app.models.rag_config import RAGConfig as RAGConfigModel
from app.utils.logging_config import get_logger

if TYPE_CHECKING:
    from app.models.knowledge_base_index import KnowledgeBaseIndex

logger = get_logger(__name__)


@dataclass(frozen=True)
class EffectiveRAGConfig:
    """Resolved configuration used to query an existing index."""

    build_config_snapshot: dict[str, Any]
    query_overrides: dict[str, Any]
    effective_config_snapshot: dict[str, Any]
    top_k: int
    generation_model: str


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
            {"type": k, "name": v["name"], "description": v["description"]}
            for k, v in RAG_TYPES.items()
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
        return get_parameter_schema(rag_type)

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
        return get_rag_class(rag_type)

    def _resolve_graph_neo4j_connection(
        self, params: dict[str, Any] | None
    ) -> tuple[str, str, str]:
        """Resolve graph_rag Neo4j params with backend settings fallback."""
        parameters = params or {}
        return resolve_neo4j_connection_params(
            parameters.get("neo4j_uri"),
            parameters.get("neo4j_username"),
            parameters.get("neo4j_password"),
            default_uri=settings.NEO4J_URI,
            default_username=settings.NEO4J_USERNAME,
            default_password=settings.NEO4J_PASSWORD,
        )

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
            embedding_model=getattr(config_model, "embedding_model", "text-embedding-3-small"),
            llm_reasoning_effort=getattr(config_model, "llm_reasoning_effort", None),
        )

        # Get the RAG class
        rag_class = self._get_rag_class(config_model.rag_type)
        parameters = config_model.parameters or {}

        # Build constructor kwargs based on RAG type
        kwargs: dict[str, Any] = {"config": rag_config}

        if config_model.rag_type == "vector_semantic":
            kwargs["collection_name"] = parameters.get(
                "collection_name", f"kb_{config_model.project_id}"
            )
            kwargs["persist_directory"] = str(Path(index_path) / "chroma") if index_path else None

        elif config_model.rag_type == "vector_hybrid":
            kwargs["collection_name"] = parameters.get(
                "collection_name", f"hybrid_{config_model.project_id}"
            )
            # Qdrant URL from settings or parameters
            kwargs["qdrant_url"] = parameters.get("qdrant_url")

        elif config_model.rag_type == "graph_rag":
            kwargs["vector_index_name"] = parameters.get("vector_index_name", "chunk_embeddings")
            neo4j_uri, neo4j_username, neo4j_password = self._resolve_graph_neo4j_connection(
                parameters
            )
            kwargs["neo4j_uri"] = neo4j_uri
            kwargs["neo4j_username"] = neo4j_username
            kwargs["neo4j_password"] = neo4j_password

        elif config_model.rag_type == "filesystem_rag":
            kwargs["llm_model"] = config_model.llm_model
            kwargs["prepared_path"] = (
                str(Path(index_path) / "filesystem_rag")
                if index_path
                else str(
                    Path(settings.STORAGE_PATH)
                    / "indexes"
                    / f"{config_model.id}"
                    / "filesystem_rag"
                )
            )
            kwargs["word_threshold"] = parameters.get("word_threshold", 1000)
            kwargs["max_iterations"] = parameters.get("max_iterations", 10)
            kwargs["max_tool_calls"] = parameters.get("max_tool_calls", 20)
            kwargs["max_file_reads"] = parameters.get("max_file_reads", 10)

        elif config_model.rag_type == "rlm_rag":
            kwargs["rlm_config"] = rlm_config_from_rag_config(rag_config)
            kwargs["prepared_path"] = (
                str(Path(index_path) / "rlm_rag")
                if index_path
                else str(Path(settings.STORAGE_PATH) / "indexes" / f"{config_model.id}" / "rlm_rag")
            )

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
            instance = self._rag_instances.pop(config_id, None)
            if instance:
                instance.close()
        else:
            for instance in self._rag_instances.values():
                instance.close()
            self._rag_instances.clear()

    def create_rag_for_index_build(self, index: "KnowledgeBaseIndex") -> BaseRAG:
        """Create a RAG instance for building a KnowledgeBaseIndex."""
        return self._create_rag_for_index_snapshot(index, index.config_snapshot)

    def create_rag_for_index(self, index: "KnowledgeBaseIndex") -> BaseRAG:
        """Create a RAG instance from an index snapshot.

        Uses the index's physical_id for storage isolation and the
        config_snapshot for reproducibility.

        Args:
            index: The KnowledgeBaseIndex to create a RAG for.

        Returns:
            Configured RAG instance using the frozen config from the index.

        Raises:
            ValueError: If the RAG type is not supported.

        This compatibility method does not call ``load_index``. New build code
        should call ``create_rag_for_index_build`` and query code should call
        ``load_rag_for_index_query``.
        """
        return self._create_rag_for_index_snapshot(index, index.config_snapshot)

    def _create_rag_for_index_snapshot(
        self,
        index: "KnowledgeBaseIndex",
        config_snapshot: dict[str, Any],
    ) -> BaseRAG:
        """Create a RAG instance configured with a provided index config snapshot."""
        storage_path = self._get_index_storage_path(index)

        rag_config = RAGConfig(
            name=index.name,
            parameters=config_snapshot.get("parameters", {}),
            storage_path=str(storage_path),
            llm_provider=config_snapshot.get("llm_provider", "openai"),
            llm_model=config_snapshot.get("llm_model", "gpt-4o-mini"),
            llm_base_url=config_snapshot.get("llm_base_url"),
            embedding_model=config_snapshot.get("embedding_model", "text-embedding-3-small"),
            llm_reasoning_effort=config_snapshot.get("llm_reasoning_effort"),
        )

        rag_type = config_snapshot.get("rag_type", "")
        if rag_type not in RAG_TYPES:
            raise ValueError(f"Unknown RAG type: {rag_type}")

        # Get the RAG class
        rag_class = self._get_rag_class(rag_type)

        # Build constructor kwargs based on RAG type
        # Use index.physical_id for collection/storage isolation
        kwargs: dict[str, Any] = {"config": rag_config}

        if rag_type == "vector_semantic":
            kwargs["collection_name"] = index.physical_id  # Isolation key
            kwargs["persist_directory"] = str(storage_path / "chroma")

        elif rag_type == "vector_hybrid":
            kwargs["collection_name"] = index.physical_id  # Isolation key
            # Qdrant URL from parameters
            kwargs["qdrant_url"] = config_snapshot.get("parameters", {}).get("qdrant_url")

        elif rag_type == "graph_rag":
            kwargs["vector_index_name"] = config_snapshot.get("parameters", {}).get(
                "vector_index_name", "chunk_embeddings"
            )
            # Add label prefix for node isolation
            kwargs["label_prefix"] = index.physical_id
            params = config_snapshot.get("parameters", {})
            neo4j_uri, neo4j_username, neo4j_password = self._resolve_graph_neo4j_connection(
                params
            )
            kwargs["neo4j_uri"] = neo4j_uri
            kwargs["neo4j_username"] = neo4j_username
            kwargs["neo4j_password"] = neo4j_password

        elif rag_type == "filesystem_rag":
            kwargs["llm_model"] = rag_config.llm_model
            kwargs["prepared_path"] = str(storage_path / "filesystem_rag")
            params = config_snapshot.get("parameters", {})
            kwargs["word_threshold"] = params.get("word_threshold", 1000)
            kwargs["max_iterations"] = params.get("max_iterations", 10)
            kwargs["max_tool_calls"] = params.get("max_tool_calls", 20)
            kwargs["max_file_reads"] = params.get("max_file_reads", 10)

        elif rag_type == "rlm_rag":
            kwargs["rlm_config"] = rlm_config_from_rag_config(rag_config)
            kwargs["prepared_path"] = str(storage_path / "rlm_rag")

        logger.info(
            "Creating RAG instance for index",
            rag_type=rag_type,
            index_id=str(index.id),
            physical_id=index.physical_id,
        )

        return rag_class(**kwargs)

    def build_effective_config(
        self,
        index: "KnowledgeBaseIndex",
        query_overrides: dict[str, Any] | None = None,
    ) -> EffectiveRAGConfig:
        """Build and validate the effective query config for an existing index."""
        build_snapshot = deepcopy(index.config_snapshot or {})
        rag_type = build_snapshot.get("rag_type")
        if not rag_type:
            raise ValueError("Index config snapshot is missing `rag_type`")

        build_snapshot["parameters"] = dict(build_snapshot.get("parameters") or {})
        build_snapshot.setdefault("llm_provider", "openai")
        build_snapshot.setdefault("llm_model", "gpt-4o-mini")
        build_snapshot.setdefault("llm_base_url", None)
        build_snapshot.setdefault("llm_reasoning_effort", None)
        build_snapshot.setdefault(
            "embedding_model",
            getattr(index, "embedding_model", None) or "text-embedding-3-small",
        )
        if rag_type == "graph_rag":
            build_snapshot["parameters"].setdefault(
                "extraction_model", build_snapshot["llm_model"]
            )
        if rag_type == "vector_hybrid":
            build_snapshot["parameters"].setdefault(
                "sparse_model_name", core_settings.sparse_model_name
            )

        normalized_overrides = validate_query_overrides(rag_type, query_overrides)
        effective_snapshot = deepcopy(build_snapshot)

        build_parameters, query_default_parameters = split_parameters(
            rag_type, build_snapshot.get("parameters", {})
        )
        effective_parameters = {
            **build_parameters,
            **query_default_parameters,
            **normalized_overrides.get("parameters", {}),
        }
        effective_snapshot["parameters"] = effective_parameters
        effective_snapshot["build_parameters"] = build_parameters
        effective_snapshot["query_default_parameters"] = query_default_parameters

        if normalized_overrides.get("llm_model"):
            effective_snapshot["llm_model"] = normalized_overrides["llm_model"]
            if (
                rag_type == "rlm_rag"
                and "orchestrator_model" not in normalized_overrides.get("parameters", {})
            ):
                effective_parameters["orchestrator_model"] = normalized_overrides["llm_model"]

        top_k = int(normalized_overrides.get("top_k", 5))
        generation_model = effective_snapshot.get("llm_model", "gpt-4o-mini")
        effective_snapshot["query_execution"] = {"top_k": top_k}

        return EffectiveRAGConfig(
            build_config_snapshot=build_snapshot,
            query_overrides=normalized_overrides,
            effective_config_snapshot=effective_snapshot,
            top_k=top_k,
            generation_model=generation_model,
        )

    def load_rag_for_index_query(
        self,
        index: "KnowledgeBaseIndex",
        query_overrides: dict[str, Any] | None = None,
    ) -> tuple[BaseRAG, EffectiveRAGConfig]:
        """Load a ready index for querying without calling preparation methods."""
        effective = self.build_effective_config(index, query_overrides)
        rag = self._create_rag_for_index_snapshot(index, effective.effective_config_snapshot)
        rag.load_index()
        return rag, effective

    def query_instance_cache_key(
        self, index: "KnowledgeBaseIndex", effective: EffectiveRAGConfig
    ) -> str:
        """Return a stable cache key for an index/effective config pair."""
        payload = json.dumps(effective.effective_config_snapshot, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{index.physical_id}:{digest}"

    def _get_index_storage_path(self, index: "KnowledgeBaseIndex") -> Path:
        """Get the storage path for a KnowledgeBaseIndex.

        Args:
            index: The index to get storage path for.

        Returns:
            Path to the index's isolated storage directory.
        """
        return Path(settings.STORAGE_PATH) / "indexes" / index.physical_id

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
