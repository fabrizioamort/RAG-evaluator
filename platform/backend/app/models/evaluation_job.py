"""Evaluation job model for tracking progress and checkpoints."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation


class EvaluationJob(BaseModelNoUpdate):
    """Job state for resumable evaluation execution."""

    __tablename__ = "evaluation_jobs"

    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    state: Mapped[str] = mapped_column(String(50), default="created", nullable=False)
    progress_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_checkpoint: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    checkpoint_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    evaluation: Mapped["Evaluation"] = relationship("Evaluation", back_populates="job")
