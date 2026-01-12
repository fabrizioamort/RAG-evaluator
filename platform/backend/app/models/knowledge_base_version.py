"""Knowledge base version model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.knowledge_base import KnowledgeBase


class KnowledgeBaseVersion(BaseModelNoUpdate):
    """Version snapshot of a knowledge base."""

    __tablename__ = "knowledge_base_versions"
    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "version_number", name="uq_kb_version"),
    )

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    change_type: Mapped[str] = mapped_column(String(50), nullable=False)
    document_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, nullable=False)
    change_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="versions"
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        back_populates="kb_version",
    )
