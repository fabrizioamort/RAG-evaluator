"""Knowledge base model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate, JSONType

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.evaluation import Evaluation
    from app.models.knowledge_base_index import KnowledgeBaseIndex
    from app.models.knowledge_base_version import KnowledgeBaseVersion
    from app.models.project import Project


class KnowledgeBase(BaseModelNoUpdate):
    """Knowledge base containing documents for RAG evaluation."""

    __tablename__ = "knowledge_bases"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    index_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONType, default=dict, nullable=False
    )

    # Soft delete support
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="knowledge_bases")
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )
    versions: Mapped[list["KnowledgeBaseVersion"]] = relationship(
        "KnowledgeBaseVersion",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        back_populates="knowledge_base",
    )
    indexes: Mapped[list["KnowledgeBaseIndex"]] = relationship(
        "KnowledgeBaseIndex",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
    )

    @property
    def is_archived(self) -> bool:
        """Check if KB is archived (soft deleted)."""
        return self.archived_at is not None
