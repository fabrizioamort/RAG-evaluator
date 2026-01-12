"""Test set model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.project import Project
    from app.models.test_case import TestCase
    from app.models.test_generation_job import TestGenerationJob


class TestSet(BaseModelNoUpdate):
    """Collection of test cases for evaluation."""

    __tablename__ = "test_sets"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[dict[str, Any]] = mapped_column(JSONB, default=list, nullable=False)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="test_sets")
    test_cases: Mapped[list["TestCase"]] = relationship(
        "TestCase",
        back_populates="test_set",
        cascade="all, delete-orphan",
    )
    generation_jobs: Mapped[list["TestGenerationJob"]] = relationship(
        "TestGenerationJob",
        back_populates="test_set",
        cascade="all, delete-orphan",
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        back_populates="test_set",
    )

    @property
    def test_case_count(self) -> int:
        """Get the number of test cases in this set."""
        return len(self.test_cases) if self.test_cases else 0
