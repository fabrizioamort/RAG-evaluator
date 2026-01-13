"""Test template model."""

from typing import Any

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModelNoUpdate, JSONType


class TestTemplate(BaseModelNoUpdate):
    """Template for generating test cases."""

    __tablename__ = "test_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    question_template: Mapped[str] = mapped_column(Text, nullable=False)
    answer_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_types: Mapped[dict[str, Any]] = mapped_column(JSONType, default=list, nullable=False)
    complexity_level: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
