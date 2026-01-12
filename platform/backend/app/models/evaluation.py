"""Evaluation model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.evaluation_job import EvaluationJob
    from app.models.evaluation_result import EvaluationResult
    from app.models.knowledge_base import KnowledgeBase
    from app.models.knowledge_base_version import KnowledgeBaseVersion
    from app.models.project import Project
    from app.models.rag_config import RAGConfig
    from app.models.run_manifest import RunManifest
    from app.models.test_set import TestSet


class Evaluation(BaseModelNoUpdate):
    """Evaluation run of a RAG configuration against a test set."""

    __tablename__ = "evaluations"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
    )
    kb_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    test_set_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("test_sets.id", ondelete="SET NULL"),
        nullable=True,
    )
    rag_config_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("rag_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    run_manifest_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("run_manifests.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    cost_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    performance_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    baseline_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="evaluations")
    knowledge_base: Mapped["KnowledgeBase | None"] = relationship(
        "KnowledgeBase", back_populates="evaluations"
    )
    kb_version: Mapped["KnowledgeBaseVersion | None"] = relationship(
        "KnowledgeBaseVersion", back_populates="evaluations"
    )
    test_set: Mapped["TestSet | None"] = relationship("TestSet", back_populates="evaluations")
    rag_config: Mapped["RAGConfig | None"] = relationship("RAGConfig", back_populates="evaluations")
    run_manifest: Mapped["RunManifest | None"] = relationship("RunManifest")
    results: Mapped[list["EvaluationResult"]] = relationship(
        "EvaluationResult",
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )
    job: Mapped["EvaluationJob | None"] = relationship(
        "EvaluationJob",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        uselist=False,
    )
