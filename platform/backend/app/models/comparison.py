"""Comparison model for comparing multiple evaluations."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate, JSONType

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.project import Project


class Comparison(BaseModelNoUpdate):
    """Comparison between two or more evaluations."""

    __tablename__ = "comparisons"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Name and description for the comparison
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The baseline evaluation to compare against
    baseline_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # List of evaluation IDs being compared (stored as JSON array)
    compared_evaluation_ids: Mapped[list[str]] = mapped_column(
        JSONType, default=list, nullable=False
    )

    # Aggregate comparison results
    aggregate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)

    # Per-question comparison results (for detailed analysis)
    per_question_deltas: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONType, nullable=True
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="comparisons")
    baseline_evaluation: Mapped["Evaluation"] = relationship(
        "Evaluation", foreign_keys=[baseline_evaluation_id]
    )
