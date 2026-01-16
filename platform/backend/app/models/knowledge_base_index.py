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

    # Relationships to source data
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

    # Physical storage (unique per index - enables isolation)
    physical_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # storage_type values: "chroma", "qdrant", "neo4j", "filesystem"
    storage_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Immutable snapshot of config at build time (for reproducibility)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)

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
    kb_version: Mapped["KnowledgeBaseVersion | None"] = relationship("KnowledgeBaseVersion")
    rag_config: Mapped["RAGConfig"] = relationship("RAGConfig", back_populates="indexes")
    evaluations: Mapped[list["Evaluation"]] = relationship("Evaluation", back_populates="index")
