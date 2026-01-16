# FIXING PLAN v2: Knowledge Base Index as First-Class Entity

## Executive Summary

This document provides a comprehensive plan to introduce `KnowledgeBaseIndex` as a first-class entity in the RAG Evaluator platform. This fixes the fundamental architectural issue where indexing was treated as a state of the Knowledge Base rather than an independent artifact.

**Key Changes:**
1. New `KnowledgeBaseIndex` model that captures KB + RAG Config = Index artifact
2. Separate top-level "Indexes" section in UI (per user preference)
3. Evaluation now references an Index, not KB + RAG Config separately
4. Soft delete for KBs with historical data preservation
5. Full storage isolation per index (no overwrites)

---

## 1. The Problem Restated

### Current Behavior
```
KB.status = "indexing" → "indexed"
KB.index_path = "/storage/indexes/{kb_id}"
```

When you index the same KB with two different RAG configs:
- **Hybrid_1 indexing** → writes to `/storage/indexes/{kb_id}/`
- **Hybrid_2 indexing** → OVERWRITES the same location!

The second indexing destroys the first. There's no way to:
- Compare two different indexing strategies on the same KB
- Know which index was used for which evaluation
- Re-run an evaluation with the exact same index

### The Solution
Treat the **output of indexing** as a first-class database object (`KnowledgeBaseIndex`) with:
- Its own unique physical storage location
- Immutable config snapshot (what was used to build it)
- Independent lifecycle (create, use, delete)

---

## 2. Database Schema Changes

### 2.1 New Table: `knowledge_base_indexes`

```sql
CREATE TABLE knowledge_base_indexes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Relationships
    knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    kb_version_id UUID REFERENCES knowledge_base_versions(id) ON DELETE SET NULL,
    rag_config_id UUID NOT NULL REFERENCES rag_configs(id) ON DELETE RESTRICT,

    -- User-facing identity
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Index status
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    -- Values: pending, building, ready, failed, archived

    -- Physical storage (unique per index - enables isolation)
    physical_id VARCHAR(64) UNIQUE NOT NULL,  -- UUID-based, e.g., "idx_abc123..."
    storage_type VARCHAR(50) NOT NULL,  -- chroma, qdrant, neo4j, filesystem

    -- Immutable snapshot of config at build time (for reproducibility)
    config_snapshot JSONB NOT NULL,  -- Full copy of RAGConfig.parameters + llm settings

    -- Build metadata
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    embedding_model VARCHAR(100),

    -- Timing
    build_started_at TIMESTAMP WITH TIME ZONE,
    build_completed_at TIMESTAMP WITH TIME ZONE,
    build_duration_seconds FLOAT,

    -- Error handling
    error_message TEXT,

    -- Standard timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_kbi_kb ON knowledge_base_indexes(knowledge_base_id);
CREATE INDEX idx_kbi_status ON knowledge_base_indexes(status);
CREATE INDEX idx_kbi_physical ON knowledge_base_indexes(physical_id);
CREATE INDEX idx_kbi_rag_config ON knowledge_base_indexes(rag_config_id);
```

### 2.2 Changes to `knowledge_bases` Table

```sql
-- Add soft-delete support (per user preference)
ALTER TABLE knowledge_bases
    ADD COLUMN archived_at TIMESTAMP WITH TIME ZONE;

-- REMOVE these columns (no longer needed)
-- index_path is now on KnowledgeBaseIndex
-- The old approach of one index per KB is obsolete
ALTER TABLE knowledge_bases
    DROP COLUMN IF EXISTS index_path;

-- Update status enum to remove "indexing" state
-- KB status is now only about document management: pending, ready, archived
-- Indexing status is on KnowledgeBaseIndex
```

### 2.3 Changes to `evaluations` Table

```sql
-- Add reference to specific index
ALTER TABLE evaluations
    ADD COLUMN knowledge_base_index_id UUID
    REFERENCES knowledge_base_indexes(id) ON DELETE RESTRICT;

-- Keep knowledge_base_id for convenience (denormalized for queries)
-- It's derivable from the index, but useful for filtering/display

-- REMOVE rag_config_id - it's now implicit via the index
-- The index captures the exact config used
ALTER TABLE evaluations
    DROP COLUMN IF EXISTS rag_config_id;

-- Note: For migration, we need to handle existing evaluations
-- See Migration section below
```

### 2.4 SQLAlchemy Models

#### New Model: `KnowledgeBaseIndex`

```python
# platform/backend/app/models/knowledge_base_index.py
"""Knowledge Base Index model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate, JSONType

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.knowledge_base import KnowledgeBase
    from app.models.knowledge_base_version import KnowledgeBaseVersion
    from app.models.rag_config import RAGConfig


class KnowledgeBaseIndex(BaseModelNoUpdate):
    """An indexed version of a Knowledge Base using a specific RAG configuration.

    This represents the artifact produced by indexing a KB with a RAG config.
    It is immutable once created (build parameters are frozen in config_snapshot).
    """

    __tablename__ = "knowledge_base_indexes"

    # Relationships
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    kb_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    rag_config_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("rag_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # User-facing identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Index status: pending, building, ready, failed, archived
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)

    # Physical storage (unique per index)
    physical_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    storage_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # storage_type values: "chroma", "qdrant", "neo4j", "filesystem"

    # Immutable snapshot of config at build time
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False
    )

    # Build metadata
    document_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Timing
    build_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    build_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    build_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="indexes"
    )
    kb_version: Mapped["KnowledgeBaseVersion | None"] = relationship(
        "KnowledgeBaseVersion"
    )
    rag_config: Mapped["RAGConfig"] = relationship(
        "RAGConfig", back_populates="indexes"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation", back_populates="index"
    )
```

#### Updated Model: `KnowledgeBase`

```python
# Changes to knowledge_base.py

class KnowledgeBase(BaseModelNoUpdate):
    # ... existing fields ...

    # REMOVE: index_path (no longer needed)
    # ADD: archived_at for soft delete
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ADD relationship to indexes
    indexes: Mapped[list["KnowledgeBaseIndex"]] = relationship(
        "KnowledgeBaseIndex",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )

    @property
    def is_archived(self) -> bool:
        """Check if KB is archived (soft deleted)."""
        return self.archived_at is not None
```

#### Updated Model: `Evaluation`

```python
# Changes to evaluation.py

class Evaluation(BaseModelNoUpdate):
    # ... existing fields ...

    # ADD: Reference to index
    knowledge_base_index_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_indexes.id", ondelete="RESTRICT"),
        nullable=True,  # Nullable for migration, should be NOT NULL for new evals
    )

    # REMOVE: rag_config_id (implicit via index)
    # rag_config_id: Mapped[uuid.UUID | None] = ...  # DELETE THIS

    # Keep knowledge_base_id for convenience (denormalized)
    # It can be derived from index.knowledge_base_id but useful for queries

    # ADD relationship
    index: Mapped["KnowledgeBaseIndex | None"] = relationship(
        "KnowledgeBaseIndex", back_populates="evaluations"
    )
```

#### Updated Model: `RAGConfig`

```python
# Changes to rag_config.py

class RAGConfig(BaseModelNoUpdate):
    # ... existing fields ...

    # ADD relationship to indexes
    indexes: Mapped[list["KnowledgeBaseIndex"]] = relationship(
        "KnowledgeBaseIndex",
        back_populates="rag_config",
    )

    # REMOVE or deprecate: evaluations relationship
    # Evaluations now link through indexes
```

---

## 3. API Changes

### 3.1 New Endpoints: `/api/v1/indexes`

```
# List all indexes (with optional filters)
GET /api/v1/indexes
    Query params:
    - kb_id: UUID (filter by knowledge base)
    - project_id: UUID (filter by project)
    - status: string (filter by status)
    - offset, limit: pagination

# Create new index (start build process)
POST /api/v1/knowledge-bases/{kb_id}/indexes
    Body: {
        "rag_config_id": UUID,
        "name": string (optional, auto-generated if not provided),
        "description": string (optional)
    }
    Returns: KnowledgeBaseIndex with status="pending"

# Get index details
GET /api/v1/indexes/{id}

# Get build progress (SSE stream)
GET /api/v1/indexes/{id}/stream
    Returns: SSE events for build progress

# Delete index (only if no evaluations reference it)
DELETE /api/v1/indexes/{id}
    Returns: 204 No Content
    Errors: 409 Conflict if evaluations exist

# Retry failed build
POST /api/v1/indexes/{id}/retry

# Archive index (soft delete, keeps evaluations working)
POST /api/v1/indexes/{id}/archive
```

### 3.2 Changes to Existing Endpoints

#### Knowledge Bases

```python
# REMOVE: POST /api/v1/knowledge-bases/{kb_id}/index
# This endpoint is replaced by POST /api/v1/knowledge-bases/{kb_id}/indexes

# MODIFY: DELETE /api/v1/knowledge-bases/{kb_id}
# Now performs soft-delete (sets archived_at) if indexes/evaluations exist
# Hard delete only if no indexes exist

# ADD: POST /api/v1/knowledge-bases/{kb_id}/archive
# Explicitly archive a KB

# ADD: POST /api/v1/knowledge-bases/{kb_id}/restore
# Restore an archived KB
```

#### Evaluations

```python
# MODIFY: POST /api/v1/evaluations
# Old body:
{
    "test_set_id": UUID,
    "knowledge_base_id": UUID,
    "rag_config_id": UUID
}
# New body:
{
    "test_set_id": UUID,
    "knowledge_base_index_id": UUID
}

# The knowledge_base_id and rag_config_id are derived from the index
# This simplifies the API and ensures consistency
```

### 3.3 Pydantic Schemas

```python
# platform/backend/app/schemas/knowledge_base_index.py

from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


class KnowledgeBaseIndexCreate(BaseModel):
    """Request to create a new index."""
    rag_config_id: UUID
    name: str | None = None  # Auto-generated if not provided
    description: str | None = None


class KnowledgeBaseIndexResponse(BaseModel):
    """Response for index details."""
    id: UUID
    knowledge_base_id: UUID
    kb_version_id: UUID | None
    rag_config_id: UUID
    name: str
    description: str | None
    status: str
    physical_id: str
    storage_type: str
    config_snapshot: dict[str, Any]
    document_count: int
    chunk_count: int
    embedding_model: str | None
    build_started_at: datetime | None
    build_completed_at: datetime | None
    build_duration_seconds: float | None
    error_message: str | None
    created_at: datetime

    # Denormalized for display convenience
    knowledge_base_name: str | None = None
    rag_config_name: str | None = None
    project_id: UUID | None = None


class KnowledgeBaseIndexList(BaseModel):
    """Paginated list of indexes."""
    items: list[KnowledgeBaseIndexResponse]
    total: int
    offset: int
    limit: int


class IndexBuildProgress(BaseModel):
    """Progress event for index building."""
    status: str  # building, processing_doc, embedding, storing, complete, failed
    current: int
    total: int
    current_document: str | None = None
    message: str | None = None
```

---

## 4. Storage Isolation Strategy

### 4.1 Physical ID Generation

Each index gets a unique `physical_id` that determines its storage location:

```python
import uuid

def generate_physical_id() -> str:
    """Generate a unique physical ID for an index."""
    return f"idx_{uuid.uuid4().hex[:24]}"

# Examples:
# idx_a1b2c3d4e5f6g7h8i9j0k1l2
# idx_m3n4o5p6q7r8s9t0u1v2w3x4
```

### 4.2 Storage Paths by RAG Type

| RAG Type | Storage Location | Details |
|----------|-----------------|---------|
| **Vector Semantic (Chroma)** | `storage/indexes/{physical_id}/chroma/` | Collection name = physical_id |
| **Hybrid Search (Qdrant)** | Qdrant collection: `{physical_id}` | Collection name = physical_id |
| **Graph RAG (Neo4j)** | Labels prefixed: `:Chunk_{physical_id}`, `:Entity_{physical_id}` | Label prefix isolation |
| **Filesystem RAG** | `storage/indexes/{physical_id}/filesystem/` | Directory per index |

### 4.3 RAG Implementation Changes

Each RAG implementation must be updated to accept explicit collection/storage identifiers:

```python
# Example: HybridSearchRAG changes

class HybridSearchRAG(BaseRAG):
    def __init__(
        self,
        collection_name: str,  # NOW REQUIRED (was optional)
        qdrant_url: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        # collection_name is now the physical_id from KnowledgeBaseIndex
        self.collection_name = collection_name  # e.g., "idx_abc123..."
        # ...
```

```python
# Example: Neo4jGraphRAG changes

class Neo4jGraphRAG(BaseRAG):
    def __init__(
        self,
        label_prefix: str,  # NEW PARAMETER
        # ...
    ) -> None:
        self.label_prefix = label_prefix  # e.g., "idx_abc123"
        # All Cypher queries now use :Chunk_{label_prefix}, :Entity_{label_prefix}
```

### 4.4 RAGAdapterService Changes

```python
# platform/backend/app/services/rag_adapter.py

def create_rag_for_index(
    self,
    index: KnowledgeBaseIndex,
) -> BaseRAG:
    """Create a RAG instance configured for a specific index.

    Uses the index's physical_id for storage isolation.
    Uses the index's config_snapshot for reproducibility.
    """
    # Build RAGConfig from the frozen snapshot
    rag_config = RAGConfig(
        name=index.name,
        parameters=index.config_snapshot.get("parameters", {}),
        storage_path=self._get_storage_path(index),
        llm_provider=index.config_snapshot.get("llm_provider", "openai"),
        llm_model=index.config_snapshot.get("llm_model", "gpt-4o-mini"),
        llm_base_url=index.config_snapshot.get("llm_base_url"),
    )

    # Determine storage type and create appropriate RAG
    if index.storage_type == "chroma":
        return ChromaSemanticRAG(
            config=rag_config,
            collection_name=index.physical_id,  # Isolation key
            persist_directory=self._get_storage_path(index) / "chroma",
        )
    elif index.storage_type == "qdrant":
        return HybridSearchRAG(
            config=rag_config,
            collection_name=index.physical_id,  # Isolation key
        )
    elif index.storage_type == "neo4j":
        return Neo4jGraphRAG(
            config=rag_config,
            label_prefix=index.physical_id,  # Isolation key
        )
    elif index.storage_type == "filesystem":
        return FilesystemRAG(
            config=rag_config,
            prepared_path=self._get_storage_path(index) / "filesystem",
        )
```

---

## 5. Service Layer Changes

### 5.1 New Service: `IndexBuildService`

```python
# platform/backend/app/services/index_build_service.py
"""Service for building Knowledge Base Indexes."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.rag_config import RAGConfig
from app.services.rag_adapter import RAGAdapterService
from app.services.job_event_log import JobEventLog
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class IndexBuildService:
    """Manages the lifecycle of index building."""

    def __init__(self, db: AsyncSession, event_log: JobEventLog):
        self.db = db
        self.event_log = event_log
        self.rag_adapter = RAGAdapterService()

    async def create_index(
        self,
        kb_id: UUID,
        rag_config_id: UUID,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBaseIndex:
        """Create a new index record and start the build process."""
        # Load KB with documents
        kb_query = (
            select(KnowledgeBase)
            .where(KnowledgeBase.id == kb_id)
            .options(selectinload(KnowledgeBase.documents))
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

        # Generate unique physical ID
        physical_id = f"idx_{uuid4().hex[:24]}"

        # Determine storage type from RAG type
        storage_type = self._get_storage_type(rag_config.rag_type)

        # Auto-generate name if not provided
        if not name:
            name = f"{kb.name} - {rag_config.name}"

        # Create frozen config snapshot
        config_snapshot = {
            "rag_type": rag_config.rag_type,
            "parameters": rag_config.parameters,
            "llm_provider": rag_config.llm_provider,
            "llm_model": rag_config.llm_model,
            "llm_base_url": rag_config.llm_base_url,
        }

        # Create index record
        index = KnowledgeBaseIndex(
            knowledge_base_id=kb_id,
            kb_version_id=kb.versions[-1].id if kb.versions else None,
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
        )

        return index

    def _get_storage_type(self, rag_type: str) -> str:
        """Map RAG type to storage type."""
        mapping = {
            "vector_semantic": "chroma",
            "vector_hybrid": "qdrant",
            "graph_rag": "neo4j",
            "filesystem_rag": "filesystem",
        }
        return mapping.get(rag_type, "unknown")

    async def build_index(self, index_id: UUID) -> None:
        """Execute the index build process."""
        # Load index with relationships
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.id == index_id)
            .options(
                selectinload(KnowledgeBaseIndex.knowledge_base)
                .selectinload(KnowledgeBase.documents)
            )
        )
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            logger.error("Index not found", index_id=str(index_id))
            return

        # Update status
        index.status = "building"
        index.build_started_at = datetime.now(timezone.utc)
        await self.db.commit()

        # Emit start event
        await self.event_log.append_event(
            str(index.id),
            "building",
            {"message": "Starting index build", "total": index.document_count}
        )

        try:
            # Create RAG instance for this index
            rag = self.rag_adapter.create_rag_for_index(index)

            # Define progress callback
            async def progress_callback(current: int, total: int, doc_name: str = ""):
                await self.event_log.append_event(
                    str(index.id),
                    "progress",
                    {"current": current, "total": total, "document": doc_name}
                )

            # Run document preparation
            rag.set_progress_callback(
                lambda c, t: asyncio.create_task(progress_callback(c, t))
            )

            # Execute indexing
            metrics = await self.rag_adapter.prepare_documents(
                rag,
                index.knowledge_base.storage_path
            )

            # Update index with results
            index.status = "ready"
            index.chunk_count = metrics.get("chunk_count", 0)
            index.embedding_model = metrics.get("embedding_model")
            index.build_completed_at = datetime.now(timezone.utc)
            index.build_duration_seconds = (
                index.build_completed_at - index.build_started_at
            ).total_seconds()

            await self.db.commit()

            await self.event_log.append_event(
                str(index.id),
                "complete",
                {"chunk_count": index.chunk_count}
            )

            logger.info(
                "Index build complete",
                index_id=str(index.id),
                chunks=index.chunk_count,
            )

        except Exception as e:
            logger.exception("Index build failed", index_id=str(index_id))

            index.status = "failed"
            index.error_message = str(e)
            index.build_completed_at = datetime.now(timezone.utc)
            await self.db.commit()

            await self.event_log.append_event(
                str(index.id),
                "failed",
                {"error": str(e)}
            )

    async def delete_index(self, index_id: UUID) -> None:
        """Delete an index and its storage."""
        query = select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == index_id)
        result = await self.db.execute(query)
        index = result.scalar_one_or_none()

        if not index:
            raise ValueError(f"Index {index_id} not found")

        # Check for evaluations
        if index.evaluations:
            raise ValueError(
                f"Cannot delete index with {len(index.evaluations)} evaluations. "
                "Delete evaluations first or archive the index."
            )

        # Clean up physical storage
        await self._cleanup_storage(index)

        # Delete from database
        await self.db.delete(index)
        await self.db.commit()

    async def _cleanup_storage(self, index: KnowledgeBaseIndex) -> None:
        """Clean up physical storage for an index."""
        if index.storage_type == "chroma":
            # Delete Chroma collection
            storage_path = Path(settings.STORAGE_PATH) / "indexes" / index.physical_id
            if storage_path.exists():
                import shutil
                shutil.rmtree(storage_path)

        elif index.storage_type == "qdrant":
            # Delete Qdrant collection
            from qdrant_client import QdrantClient
            client = QdrantClient(url=settings.qdrant_url)
            try:
                client.delete_collection(index.physical_id)
            except Exception as e:
                logger.warning(f"Failed to delete Qdrant collection: {e}")

        elif index.storage_type == "neo4j":
            # Delete Neo4j nodes with this prefix
            # This requires running Cypher to delete :Chunk_{physical_id} etc.
            pass  # Implementation depends on Neo4j driver

        elif index.storage_type == "filesystem":
            storage_path = Path(settings.STORAGE_PATH) / "indexes" / index.physical_id
            if storage_path.exists():
                import shutil
                shutil.rmtree(storage_path)
```

### 5.2 Updated `EvaluationRunner`

```python
# Changes to platform/backend/app/services/evaluation_runner.py

class EvaluationRunner:
    async def run_evaluation(self, evaluation: Evaluation) -> None:
        """Run an evaluation using the index's RAG configuration."""

        # Load the index (which contains the frozen config)
        index = evaluation.index
        if not index:
            raise ValueError("Evaluation has no associated index")

        if index.status != "ready":
            raise ValueError(f"Index is not ready: {index.status}")

        # Create RAG instance from the index
        rag = self.rag_adapter.create_rag_for_index(index)

        # Run evaluation against this RAG instance
        # ... rest of evaluation logic ...
```

---

## 6. Frontend Changes

### 6.1 New Pages

#### Indexes List Page (`/indexes`)

```
+------------------------------------------------------------------+
|  Indexes                                               [+ New Index]
+------------------------------------------------------------------+
|  Filter: [All KBs ▼] [All Status ▼] [All Projects ▼]  [Search...] |
+------------------------------------------------------------------+
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | idx_abc123 - Finance Docs (Hybrid Search)                    |  |
|  | KB: Finance Docs v3 | RAG: Hybrid Search - High Precision    |  |
|  | Status: ✓ Ready | Chunks: 1,234 | Built: 2h ago              |  |
|  | [View] [Use in Evaluation] [Delete]                          |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | idx_def456 - Finance Docs (Vector Semantic)                  |  |
|  | KB: Finance Docs v3 | RAG: Vector - Small Chunks             |  |
|  | Status: ✓ Ready | Chunks: 2,567 | Built: 1d ago              |  |
|  | [View] [Use in Evaluation] [Delete]                          |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | idx_ghi789 - HR Policies (Graph RAG)           [In Progress] |  |
|  | KB: HR Policies v1 | RAG: Graph RAG                          |  |
|  | Status: ⏳ Building (45%) | Progress: 45/100 docs            |  |
|  | [View Progress] [Cancel]                                      |  |
|  +--------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

#### Index Detail Page (`/indexes/{id}`)

```
+------------------------------------------------------------------+
|  Index: Finance Docs (Hybrid Search)                              |
|  Physical ID: idx_abc123def456...                                 |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------------+  +-----------------------------------+  |
|  | Source               |  | Configuration (Frozen)            |  |
|  | KB: Finance Docs     |  | RAG Type: Hybrid Search          |  |
|  | Version: v3          |  | Chunk Size: 500                  |  |
|  | Documents: 45        |  | Chunk Overlap: 50                |  |
|  +----------------------+  | LLM: gpt-4o-mini                 |  |
|                            +-----------------------------------+  |
|                                                                    |
|  +-----------------------------------------------------------+    |
|  | Build Statistics                                          |    |
|  | Chunks Created: 1,234                                     |    |
|  | Build Time: 3m 45s                                        |    |
|  | Built At: Jan 16, 2026 10:30 AM                          |    |
|  | Embedding Model: text-embedding-3-small                   |    |
|  +-----------------------------------------------------------+    |
|                                                                    |
|  +-----------------------------------------------------------+    |
|  | Evaluations Using This Index (3)                          |    |
|  | - Eval #12: 85% pass rate (Jan 15)                       |    |
|  | - Eval #8: 82% pass rate (Jan 14)                        |    |
|  | - Eval #5: 78% pass rate (Jan 13)                        |    |
|  +-----------------------------------------------------------+    |
|                                                                    |
|  [Run New Evaluation]  [Delete Index]                             |
+------------------------------------------------------------------+
```

### 6.2 Updated Evaluation Creation Flow

```
Step 1: Select Test Set
+------------------------------------------------------------------+
|  New Evaluation - Step 1 of 2                                     |
+------------------------------------------------------------------+
|  Select Test Set:                                                 |
|  +--------------------------------------------------------------+|
|  | ○ General Q&A (50 questions)                                 ||
|  | ● Finance FAQ (25 questions) ← Selected                      ||
|  | ○ HR Policies Test (30 questions)                            ||
|  +--------------------------------------------------------------+|
|                                                    [Next →]       |
+------------------------------------------------------------------+

Step 2: Select Index (KB + Config combined)
+------------------------------------------------------------------+
|  New Evaluation - Step 2 of 2                                     |
+------------------------------------------------------------------+
|  Select Knowledge Base:                                           |
|  [Finance Docs ▼]                                                 |
|                                                                    |
|  Select Index:                                                    |
|  +--------------------------------------------------------------+|
|  | ○ Hybrid Search - High Precision (1,234 chunks)              ||
|  |   Built Jan 16, used in 3 evaluations                        ||
|  | ● Vector Semantic - Large Chunks (567 chunks) ← Selected     ||
|  |   Built Jan 15, used in 1 evaluation                         ||
|  | ○ Graph RAG (2,100 chunks)                                   ||
|  |   Built Jan 14, never used                                   ||
|  +--------------------------------------------------------------+|
|                                                                    |
|  [← Back]                                      [Start Evaluation] |
+------------------------------------------------------------------+
```

### 6.3 Updated Navigation

```typescript
// Layout sidebar
<nav>
  <NavItem to="/projects">Projects</NavItem>
  <NavItem to="/knowledge-bases">Knowledge Bases</NavItem>
  <NavItem to="/indexes">Indexes</NavItem>  {/* NEW */}
  <NavItem to="/test-sets">Test Sets</NavItem>
  <NavItem to="/evaluations">Evaluations</NavItem>
</nav>
```

### 6.4 Component Changes

```typescript
// New: IndexCard.tsx
interface IndexCardProps {
  index: KnowledgeBaseIndex;
  onDelete?: () => void;
  onRunEvaluation?: () => void;
}

// New: IndexList.tsx
// Lists all indexes with filtering

// New: CreateIndexDialog.tsx
interface CreateIndexDialogProps {
  knowledgeBaseId: string;
  onCreated: (index: KnowledgeBaseIndex) => void;
}

// New: IndexBuildProgress.tsx
// Shows SSE-based build progress

// Modified: StartEvaluationWizard.tsx
// Now selects Index instead of KB + Config
```

---

## 7. Migration Strategy

### 7.1 Database Migration

```python
# alembic/versions/xxx_add_knowledge_base_indexes.py

def upgrade():
    # 1. Create new table
    op.create_table(
        'knowledge_base_indexes',
        # ... columns as defined above
    )

    # 2. Add archived_at to knowledge_bases
    op.add_column(
        'knowledge_bases',
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True)
    )

    # 3. Add knowledge_base_index_id to evaluations
    op.add_column(
        'evaluations',
        sa.Column(
            'knowledge_base_index_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('knowledge_base_indexes.id', ondelete='RESTRICT'),
            nullable=True
        )
    )

    # 4. Create indexes
    op.create_index('idx_kbi_kb', 'knowledge_base_indexes', ['knowledge_base_id'])
    op.create_index('idx_kbi_status', 'knowledge_base_indexes', ['status'])


def downgrade():
    op.drop_column('evaluations', 'knowledge_base_index_id')
    op.drop_column('knowledge_bases', 'archived_at')
    op.drop_table('knowledge_base_indexes')
```

### 7.2 Data Migration for Existing Evaluations

```python
# scripts/migrate_existing_evaluations.py
"""Migrate existing evaluations to use indexes."""

async def migrate():
    """Create indexes for existing evaluations and link them."""

    # Find evaluations without index
    old_evals = await db.execute(
        select(Evaluation)
        .where(Evaluation.knowledge_base_index_id.is_(None))
        .where(Evaluation.knowledge_base_id.isnot(None))
        .where(Evaluation.rag_config_id.isnot(None))
    )

    for eval in old_evals.scalars():
        # Check if a matching index already exists
        existing_index = await db.execute(
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.knowledge_base_id == eval.knowledge_base_id)
            .where(KnowledgeBaseIndex.rag_config_id == eval.rag_config_id)
            .where(KnowledgeBaseIndex.status == "ready")
        )

        index = existing_index.scalar_one_or_none()

        if not index:
            # Create a "legacy" index
            # Mark it so users know it was migrated
            index = await create_legacy_index(eval)

        # Link evaluation to index
        eval.knowledge_base_index_id = index.id

    await db.commit()
```

---

## 8. Implementation Phases

### Phase 1: Database & Models (Week 1)
- [ ] Create `KnowledgeBaseIndex` model
- [ ] Add `archived_at` to `KnowledgeBase`
- [ ] Add `knowledge_base_index_id` to `Evaluation`
- [ ] Write Alembic migration
- [ ] Update model relationships
- [ ] Add Pydantic schemas

### Phase 2: Services (Week 1-2)
- [ ] Create `IndexBuildService`
- [ ] Update `RAGAdapterService.create_rag_for_index()`
- [ ] Update each RAG implementation to use explicit collection naming
- [ ] Add storage cleanup utilities
- [ ] Update `EvaluationRunner` to use indexes

### Phase 3: API Endpoints (Week 2)
- [ ] Create `/api/v1/indexes` CRUD endpoints
- [ ] Create `/api/v1/indexes/{id}/stream` SSE endpoint
- [ ] Update `/api/v1/evaluations` to accept `knowledge_base_index_id`
- [ ] Update `/api/v1/knowledge-bases/{id}` for soft delete
- [ ] Write API tests

### Phase 4: Frontend (Week 2-3)
- [ ] Create `IndexList` page
- [ ] Create `IndexDetail` page
- [ ] Create `CreateIndexDialog` component
- [ ] Create `IndexBuildProgress` component
- [ ] Update `StartEvaluationWizard` to select Index
- [ ] Update navigation sidebar
- [ ] Add filtering/search for indexes

### Phase 5: Migration & Testing (Week 3)
- [ ] Write data migration script for existing evaluations
- [ ] Test migration on sample data
- [ ] End-to-end testing of new flow
- [ ] Performance testing with multiple indexes
- [ ] Documentation updates

---

## 9. Testing Plan

### 9.1 Unit Tests

```python
# tests/test_services/test_index_build_service.py

async def test_create_index_generates_unique_physical_id():
    """Each index should have a unique physical_id."""
    ...

async def test_create_index_captures_config_snapshot():
    """Config snapshot should be immutable copy of RAG config."""
    ...

async def test_cannot_delete_index_with_evaluations():
    """Should raise error when deleting index with linked evaluations."""
    ...

async def test_storage_isolation_chroma():
    """Two indexes should not interfere with each other."""
    ...
```

### 9.2 Integration Tests

```python
# tests/test_api/test_indexes.py

async def test_create_and_build_index():
    """Full flow: create index, wait for build, verify ready."""
    ...

async def test_create_evaluation_with_index():
    """Create evaluation using an index."""
    ...

async def test_soft_delete_kb_with_indexes():
    """KB with indexes should be archived, not deleted."""
    ...
```

### 9.3 Manual Test Scenarios

1. **Multiple Indexes Same KB**
   - Create KB with 10 documents
   - Create 3 indexes with different RAG configs
   - Verify each has different physical_id
   - Run evaluation on each, compare results

2. **Index Deletion**
   - Create index, run evaluation
   - Try to delete index (should fail)
   - Delete evaluation, then delete index (should succeed)

3. **Build Progress**
   - Create index with 50 documents
   - Watch SSE progress in UI
   - Verify progress updates correctly

4. **Archive Flow**
   - Create KB, create indexes, run evaluations
   - Archive KB
   - Verify evaluations still work (read-only)
   - Cannot create new indexes on archived KB

---

## 10. API Response Examples

### Create Index

```http
POST /api/v1/knowledge-bases/kb123/indexes
Content-Type: application/json

{
  "rag_config_id": "cfg456",
  "name": "High Precision Hybrid",
  "description": "Tuned for accuracy over speed"
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": "idx789",
  "knowledge_base_id": "kb123",
  "rag_config_id": "cfg456",
  "name": "High Precision Hybrid",
  "description": "Tuned for accuracy over speed",
  "status": "pending",
  "physical_id": "idx_a1b2c3d4e5f6g7h8i9j0k1l2",
  "storage_type": "qdrant",
  "config_snapshot": {
    "rag_type": "vector_hybrid",
    "parameters": {"chunk_size": 500, "chunk_overlap": 50},
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini"
  },
  "document_count": 45,
  "chunk_count": 0,
  "created_at": "2026-01-16T10:30:00Z"
}
```

### Create Evaluation (New Format)

```http
POST /api/v1/evaluations
Content-Type: application/json

{
  "project_id": "proj123",
  "test_set_id": "ts456",
  "knowledge_base_index_id": "idx789"
}
```

---

## 11. Summary of Changes

| Area | Before | After |
|------|--------|-------|
| **KB Model** | Has `index_path` | Has `archived_at`, no index_path |
| **Indexing** | Overwrites same path | Creates unique `KnowledgeBaseIndex` |
| **Evaluation** | References KB + RAG Config | References `KnowledgeBaseIndex` |
| **Storage** | `storage/indexes/{kb_id}/` | `storage/indexes/{physical_id}/` |
| **UI Flow** | Index KB → Evaluate | Create Index → Use Index in Eval |
| **Deletion** | Hard delete KB | Soft delete if has indexes |

---

## 12. Open Questions / Decisions Made

| Question | Decision |
|----------|----------|
| Where to show Indexes in UI? | **Separate top-level menu** (per user preference) |
| How to handle KB deletion? | **Soft delete** - archive instead of cascade |
| Test generation from Index or KB? | **KB only** - test sets independent of indexing |
| Evaluation selection flow? | **Pick KB first, then Index** |

---

**Document Version:** 2.0
**Last Updated:** 2026-01-16
**Status:** Ready for Implementation
