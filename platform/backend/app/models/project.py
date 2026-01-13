"""Project model."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel, JSONType

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.knowledge_base import KnowledgeBase
    from app.models.rag_config import RAGConfig
    from app.models.test_set import TestSet
    from app.models.webhook import Webhook


class Project(BaseModel):
    """Project model for organizing RAG evaluations."""

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    tags: Mapped[dict[str, Any]] = mapped_column(JSONType, default=list, nullable=False)

    # Relationships
    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        "KnowledgeBase",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    test_sets: Mapped[list["TestSet"]] = relationship(
        "TestSet",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    rag_configs: Mapped[list["RAGConfig"]] = relationship(
        "RAGConfig",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    webhooks: Mapped[list["Webhook"]] = relationship(
        "Webhook",
        back_populates="project",
        cascade="all, delete-orphan",
    )
