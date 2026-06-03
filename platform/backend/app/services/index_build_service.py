"""Service for building Knowledge Base Indexes.

This service manages the lifecycle of index building, from creation
through execution and cleanup.
"""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine
from uuid import UUID, uuid4

from rag_evaluator.config import settings as core_settings
from rag_evaluator.rag_implementations.graph_rag.neo4j_connection import (
    resolve_neo4j_connection_params,
)
from rag_evaluator.rag_implementations.registry import split_parameters
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.rag_config import RAGConfig
from app.services.job_event_log import JobEventLog, get_job_event_log
from app.services.rag_adapter import get_rag_adapter_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# Storage type mapping from RAG type
RAG_TYPE_TO_STORAGE: dict[str, str] = {
    "vector_semantic": "chroma",
    "vector_hybrid": "qdrant",
    "graph_rag": "neo4j",
    "filesystem_rag": "filesystem",
    "rlm_rag": "filesystem",
}


def generate_physical_id() -> str:
    """Generate a unique physical ID for an index.

    Returns:
        A unique identifier like "idx_a1b2c3d4e5f6g7h8i9j0k1l2".
    """
    return f"idx_{uuid4().hex[:24]}"


class IndexBuildService:
    """Service for managing Knowledge Base Index lifecycle.

    This service handles:
    - Creating new index records
    - Executing the index build process
    - Cleaning up storage on deletion
    - Archiving indexes
    """

    def __init__(self, db: AsyncSession, event_log: JobEventLog | None = None) -> None:
        """Initialize the index build service.

        Args:
            db: Async database session.
            event_log: Optional job event log for streaming progress.
        """
        self.db = db
        self.event_log = event_log or get_job_event_log()
        self.rag_adapter = get_rag_adapter_service()

    async def create_index(
        self,
        kb_id: UUID,
        rag_config_id: UUID,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBaseIndex:
        """Create a new index record and prepare for building.

        Args:
            kb_id: Knowledge base ID to index.
            rag_config_id: RAG configuration to use.
            name: Optional custom name (auto-generated if not provided).
            description: Optional description.

        Returns:
            The created KnowledgeBaseIndex with status="pending".

        Raises:
            ValueError: If KB or config not found, KB is archived, or KB has no documents.
        """
        # Load KB with documents
        kb_query = (
            select(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id)
            .options(
                selectinload(KnowledgeBase.documents),
                selectinload(KnowledgeBase.versions),
            )
        )
        result = await self.db.execute(kb_query)
        kb = result.scalar_one_or_none()

        if not kb:
            raise ValueError(f"Knowledge base {kb_id} not found")

        if kb.is_archived:
            raise ValueError("Cannot create index for archived knowledge base")

        if not kb.documents:
            raise ValueError("Cannot index empty knowledge base")

        # Load RAG config
        config_query = select(RAGConfig).where(RAGConfig.id == rag_config_id)
        config_result = await self.db.execute(config_query)
        rag_config = config_result.scalar_one_or_none()

        if not rag_config:
            raise ValueError(f"RAG config {rag_config_id} not found")

        # Generate unique physical ID for storage isolation
        physical_id = generate_physical_id()

        # Determine storage type from RAG type
        storage_type = RAG_TYPE_TO_STORAGE.get(rag_config.rag_type, "unknown")

        # Auto-generate name if not provided
        if not name:
            name = f"{kb.name} - {rag_config.name}"

        # Create frozen config snapshot for reproducibility
        config_snapshot = self._build_config_snapshot(rag_config)

        # Get the latest version ID if available
        kb_version_id = None
        if kb.versions:
            # Get the latest version
            sorted_versions = sorted(kb.versions, key=lambda v: v.created_at, reverse=True)
            kb_version_id = sorted_versions[0].id

        # Create index record
        index = KnowledgeBaseIndex(
            knowledge_base_id=kb_id,
            kb_version_id=kb_version_id,
            rag_config_id=rag_config_id,
            name=name,
            description=description,
            status="pending",
            physical_id=physical_id,
            storage_type=storage_type,
            config_snapshot=config_snapshot,
            document_count=len(kb.documents),
        )

        self.db.add(index)
        await self.db.commit()
        await self.db.refresh(index)

        logger.info(
            "Created index record",
            index_id=str(index.id),
            physical_id=physical_id,
            kb_id=str(kb_id),
            rag_type=rag_config.rag_type,
        )

        return index

    def get_storage_path(self, index: KnowledgeBaseIndex) -> Path:
        """Get the storage path for an index.

        Args:
            index: The index to get storage path for.

        Returns:
            Path to the index storage directory.
        """
        return Path(settings.STORAGE_PATH) / "indexes" / index.physical_id

    def _build_config_snapshot(self, rag_config: RAGConfig) -> dict[str, Any]:
        """Build the immutable config snapshot stored on a KnowledgeBaseIndex."""
        parameters = dict(rag_config.parameters or {})
        if rag_config.rag_type == "graph_rag":
            parameters.setdefault("extraction_model", rag_config.llm_model)
        if rag_config.rag_type == "vector_hybrid":
            parameters.setdefault("sparse_model_name", core_settings.sparse_model_name)

        build_parameters, query_default_parameters = split_parameters(
            rag_config.rag_type, parameters
        )

        return {
            "rag_type": rag_config.rag_type,
            "parameters": parameters,
            "build_parameters": build_parameters,
            "query_default_parameters": query_default_parameters,
            "llm_provider": rag_config.llm_provider,
            "llm_model": rag_config.llm_model,
            "llm_base_url": rag_config.llm_base_url,
            "embedding_model": rag_config.embedding_model,
            "embedding_provider": rag_config.embedding_provider,
            "embedding_base_url": rag_config.embedding_base_url,
        }

    async def build_index(
        self,
        index_id: UUID,
        progress_callback: Callable[[int, int, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Execute the index build process.

        Args:
            index_id: ID of the index to build.
            progress_callback: Optional async callback for progress updates.
                Signature: (current: int, total: int, doc_name: str) -> None
        """
        # Load index with relationships
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.id == index_id)
            .options(
                selectinload(KnowledgeBaseIndex.knowledge_base).selectinload(
                    KnowledgeBase.documents
                ),
                selectinload(KnowledgeBaseIndex.rag_config),
            )
        )
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            logger.error("Index not found", index_id=str(index_id))
            return

        if index.status not in ("pending", "failed"):
            logger.warning(
                "Index not in buildable state",
                index_id=str(index_id),
                status=index.status,
            )
            return

        # Update status to building
        index.status = "building"
        index.build_started_at = datetime.now(timezone.utc)
        index.error_message = None  # Clear any previous error
        await self.db.commit()

        # Emit start event
        await self.event_log.log_event(
            index.id,
            "building",
            {
                "message": "Starting index build",
                "total": index.document_count,
                "physical_id": index.physical_id,
            },
        )

        try:
            # Create RAG instance configured for this index
            rag = self.rag_adapter.create_rag_for_index_build(index)

            # Define internal progress callback that broadcasts to SSE
            async def internal_progress(current: int, total: int, doc_name: str = "") -> None:
                await self.event_log.log_event(
                    index.id,
                    "progress",
                    {"current": current, "total": total, "document": doc_name},
                )
                if progress_callback:
                    await progress_callback(current, total, doc_name)

            # Create sync wrapper for the RAG's progress callback
            def sync_progress_wrapper(current: int, total: int, doc_name: str = "") -> None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(internal_progress(current, total, doc_name))
                except RuntimeError:
                    # No running event loop - skip progress reporting
                    pass

            # Set progress callback on RAG if supported
            if hasattr(rag, "set_progress_callback"):
                rag.set_progress_callback(sync_progress_wrapper)

            # Execute indexing - run in thread pool to avoid blocking
            documents_path = index.knowledge_base.storage_path
            if not documents_path:
                raise ValueError("Knowledge base has no storage path")

            metrics = await self.rag_adapter.prepare_documents(rag, documents_path)

            # Update index with results
            index.status = "ready"
            index.chunk_count = metrics.get("chunk_count", metrics.get("total_chunks", 0))
            index.embedding_model = metrics.get(
                "embedding_model", index.config_snapshot.get("embedding_model")
            )
            index.build_completed_at = datetime.now(timezone.utc)

            if index.build_started_at:
                index.build_duration_seconds = (
                    index.build_completed_at - index.build_started_at
                ).total_seconds()

            await self.db.commit()

            await self.event_log.log_event(
                index.id,
                "complete",
                {
                    "chunk_count": index.chunk_count,
                    "duration_seconds": index.build_duration_seconds,
                },
            )

            logger.info(
                "Index build complete",
                index_id=str(index.id),
                physical_id=index.physical_id,
                chunks=index.chunk_count,
                duration=index.build_duration_seconds,
            )

        except Exception as e:
            logger.exception("Index build failed", index_id=str(index_id))

            index.status = "failed"
            index.error_message = str(e)
            index.build_completed_at = datetime.now(timezone.utc)
            await self.db.commit()

            await self.event_log.log_event(
                index.id,
                "failed",
                {"error": str(e)},
            )

    async def retry_build(self, index_id: UUID) -> KnowledgeBaseIndex:
        """Retry a failed index build.

        Args:
            index_id: ID of the failed index.

        Returns:
            The index with status reset to pending.

        Raises:
            ValueError: If index not found or not in failed state.
        """
        query = select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == index_id)
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            raise ValueError(f"Index {index_id} not found")

        if index.status != "failed":
            raise ValueError(f"Can only retry failed indexes, current status: {index.status}")

        # Reset status
        index.status = "pending"
        index.error_message = None
        index.build_started_at = None
        index.build_completed_at = None
        index.build_duration_seconds = None
        index.chunk_count = 0

        await self.db.commit()
        await self.db.refresh(index)

        logger.info("Index reset for retry", index_id=str(index_id))

        return index

    async def archive_index(self, index_id: UUID) -> KnowledgeBaseIndex:
        """Archive an index (soft delete that preserves evaluations).

        Args:
            index_id: ID of the index to archive.

        Returns:
            The archived index.

        Raises:
            ValueError: If index not found.
        """
        query = select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == index_id)
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            raise ValueError(f"Index {index_id} not found")

        index.status = "archived"
        await self.db.commit()
        await self.db.refresh(index)

        logger.info("Index archived", index_id=str(index_id))

        return index

    async def delete_index(self, index_id: UUID, force: bool = False) -> None:
        """Delete an index and its physical storage.

        Args:
            index_id: ID of the index to delete.
            force: If True, delete even if evaluations reference this index.

        Raises:
            ValueError: If index not found or has evaluations (unless force=True).
        """
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.id == index_id)
            .options(selectinload(KnowledgeBaseIndex.evaluations))
        )
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            raise ValueError(f"Index {index_id} not found")

        # Check for evaluations
        if index.evaluations and not force:
            raise ValueError(
                f"Cannot delete index with {len(index.evaluations)} evaluations. "
                "Delete evaluations first, archive the index, or use force=True."
            )

        # Clean up physical storage
        await self._cleanup_storage(index)

        # Delete from database
        await self.db.delete(index)
        await self.db.commit()

        logger.info(
            "Index deleted",
            index_id=str(index_id),
            physical_id=index.physical_id,
        )

    async def _cleanup_storage(self, index: KnowledgeBaseIndex) -> None:
        """Clean up physical storage for an index.

        Args:
            index: The index whose storage should be cleaned up.
        """
        storage_path = self.get_storage_path(index)

        if index.storage_type == "chroma":
            # Delete Chroma collection directory
            if storage_path.exists():
                try:
                    shutil.rmtree(storage_path)
                    logger.info("Deleted Chroma storage", path=str(storage_path))
                except Exception as e:
                    logger.warning("Failed to delete Chroma storage", error=str(e))

        elif index.storage_type == "qdrant":
            # Delete Qdrant collection via API
            try:
                from qdrant_client import QdrantClient

                # Try to get Qdrant URL from config snapshot or settings
                qdrant_url = index.config_snapshot.get("parameters", {}).get("qdrant_url")
                if qdrant_url:
                    client = QdrantClient(url=qdrant_url)
                    client.delete_collection(index.physical_id)
                    logger.info("Deleted Qdrant collection", collection=index.physical_id)
            except ImportError:
                logger.warning("qdrant_client not installed, cannot cleanup Qdrant collection")
            except Exception as e:
                logger.warning("Failed to delete Qdrant collection", error=str(e))

        elif index.storage_type == "neo4j":
            # Delete Neo4j nodes with this label prefix
            # This requires running Cypher to delete all nodes with the prefix
            try:
                neo4j_params = index.config_snapshot.get("parameters", {})
                neo4j_uri, neo4j_username, neo4j_password = resolve_neo4j_connection_params(
                    neo4j_params.get("neo4j_uri"),
                    neo4j_params.get("neo4j_username"),
                    neo4j_params.get("neo4j_password"),
                    default_uri=settings.NEO4J_URI,
                    default_username=settings.NEO4J_USERNAME,
                    default_password=settings.NEO4J_PASSWORD,
                )

                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
                with driver.session() as session:
                    session.run(
                        "MATCH (n) "
                        "WHERE any(label IN labels(n) WHERE label STARTS WITH $label_prefix) "
                        "DETACH DELETE n",
                        {"label_prefix": index.physical_id},
                    )
                driver.close()
                logger.info("Deleted Neo4j nodes", prefix=index.physical_id)
            except ImportError:
                logger.warning("neo4j driver not installed, cannot cleanup Neo4j nodes")
            except Exception as e:
                logger.warning("Failed to delete Neo4j nodes", error=str(e))

        elif index.storage_type == "filesystem":
            # Delete filesystem RAG directory
            if storage_path.exists():
                try:
                    shutil.rmtree(storage_path)
                    logger.info("Deleted filesystem storage", path=str(storage_path))
                except Exception as e:
                    logger.warning("Failed to delete filesystem storage", error=str(e))

    async def get_index(self, index_id: UUID) -> KnowledgeBaseIndex | None:
        """Get an index by ID with relationships loaded.

        Args:
            index_id: ID of the index.

        Returns:
            The index or None if not found.
        """
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.id == index_id)
            .options(
                selectinload(KnowledgeBaseIndex.knowledge_base),
                selectinload(KnowledgeBaseIndex.rag_config),
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_indexes(
        self,
        knowledge_base_id: UUID | None = None,
        project_id: UUID | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[KnowledgeBaseIndex], int]:
        """List indexes with optional filtering.

        Args:
            knowledge_base_id: Filter by knowledge base.
            project_id: Filter by project.
            status: Filter by status.
            offset: Pagination offset.
            limit: Pagination limit.

        Returns:
            Tuple of (list of indexes, total count).
        """
        from sqlalchemy import func

        # Build base query
        query = select(KnowledgeBaseIndex).options(
            selectinload(KnowledgeBaseIndex.knowledge_base),
            selectinload(KnowledgeBaseIndex.rag_config),
        )

        # Apply filters
        if knowledge_base_id:
            query = query.where(KnowledgeBaseIndex.knowledge_base_id == knowledge_base_id)

        if project_id:
            query = query.join(KnowledgeBase).where(KnowledgeBase.project_id == project_id)

        if status:
            query = query.where(KnowledgeBaseIndex.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()

        # Apply pagination and ordering
        query = query.order_by(KnowledgeBaseIndex.created_at.desc())
        query = query.offset(offset).limit(limit)

        result = await self.db.execute(query)
        indexes = list(result.scalars().all())

        return indexes, total


# Factory function for getting the service
def get_index_build_service(db: AsyncSession) -> IndexBuildService:
    """Get an IndexBuildService instance.

    Args:
        db: Async database session.

    Returns:
        IndexBuildService instance.
    """
    return IndexBuildService(db)
