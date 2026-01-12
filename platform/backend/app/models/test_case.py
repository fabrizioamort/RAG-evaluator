"""Test case model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.evaluation_result import EvaluationResult
    from app.models.test_set import TestSet
    from app.models.test_template import TestTemplate


class TestCase(BaseModelNoUpdate):
    """Individual test case for RAG evaluation."""

    __tablename__ = "test_cases"

    test_set_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("test_sets.id", ondelete="CASCADE"),
        nullable=False,
    )
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("test_templates.id", ondelete="SET NULL"),
        nullable=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    difficulty: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    question_type: Mapped[str] = mapped_column(String(50), default="factual", nullable=False)
    is_generated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    provenance_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    test_set: Mapped["TestSet"] = relationship("TestSet", back_populates="test_cases")
    template: Mapped["TestTemplate | None"] = relationship("TestTemplate")
    provenance_artifact: Mapped["Artifact | None"] = relationship("Artifact")
    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
        "EvaluationResult",
        back_populates="test_case",
    )
