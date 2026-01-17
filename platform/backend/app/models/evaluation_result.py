"""Evaluation result model."""

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.evaluation import Evaluation
    from app.models.test_case import TestCase


class EvaluationResult(BaseModelNoUpdate):
    """Result of evaluating a single test case."""

    __tablename__ = "evaluation_results"

    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    test_case_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("test_cases.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Generated answer
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Artifact references
    retrieved_context_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    retrieval_trace_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    raw_metrics_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Metric scores with explanations
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevancy_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    recall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance metrics
    latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)

    # Relationships
    evaluation: Mapped["Evaluation"] = relationship("Evaluation", back_populates="results")
    test_case: Mapped["TestCase | None"] = relationship(
        "TestCase", back_populates="evaluation_results"
    )
    retrieved_context_artifact: Mapped["Artifact | None"] = relationship(
        "Artifact", foreign_keys=[retrieved_context_artifact_id]
    )
    retrieval_trace_artifact: Mapped["Artifact | None"] = relationship(
        "Artifact", foreign_keys=[retrieval_trace_artifact_id]
    )
    raw_metrics_artifact: Mapped["Artifact | None"] = relationship(
        "Artifact", foreign_keys=[raw_metrics_artifact_id]
    )
