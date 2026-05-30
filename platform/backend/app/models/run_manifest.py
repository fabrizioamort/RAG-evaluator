"""Run manifest model for reproducibility."""

from typing import Any

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModelNoUpdate, JSONType


class RunManifest(BaseModelNoUpdate):
    """Immutable snapshot of configuration at evaluation time."""

    __tablename__ = "run_manifests"

    rag_config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    build_config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONType, default=dict, nullable=False
    )
    query_overrides: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    effective_config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONType, default=dict, nullable=False
    )
    kb_version_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    generation_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    eval_judge_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_templates: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    rag_evaluator_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    platform_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
