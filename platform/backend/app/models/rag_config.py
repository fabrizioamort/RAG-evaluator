"""RAG configuration model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModelNoUpdate, JSONType

if TYPE_CHECKING:
    from app.models.evaluation import Evaluation
    from app.models.project import Project


class RAGConfig(BaseModelNoUpdate):
    """Configuration for a RAG implementation."""

    __tablename__ = "rag_configs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rag_type: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(50), default="openai", nullable=False)
    llm_model: Mapped[str] = mapped_column(String(100), default="gpt-4o-mini", nullable=False)
    llm_base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="rag_configs")
    evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        back_populates="rag_config",
    )
