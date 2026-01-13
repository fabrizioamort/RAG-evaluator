"""Test generation job model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate, JSONType

if TYPE_CHECKING:
    from app.models.knowledge_base import KnowledgeBase
    from app.models.test_set import TestSet


class TestGenerationJob(BaseModelNoUpdate):
    """Job for generating test cases from a knowledge base."""

    __tablename__ = "test_generation_jobs"

    test_set_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("test_sets.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    questions_generated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    questions_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    questions_rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    test_set: Mapped["TestSet"] = relationship("TestSet", back_populates="generation_jobs")
    knowledge_base: Mapped["KnowledgeBase | None"] = relationship("KnowledgeBase")
