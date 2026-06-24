"""Playground Query model for storing query history."""

import uuid
from typing import Any

from sqlalchemy import Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModelNoUpdate, JSONType


class PlaygroundQuery(BaseModelNoUpdate):
    """A playground query execution with results.

    Stores query history for the RAG playground feature, enabling users
    to review past queries and their results.
    """

    __tablename__ = "playground_queries"

    # Query details
    question: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)

    # Indexes queried (stored as list of UUIDs)
    index_ids: Mapped[list[uuid.UUID]] = mapped_column(JSONType, nullable=False)

    # Results (stored as JSON blob for flexibility)
    # Contains list of PlaygroundQueryResult dictionaries
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, nullable=False)

    # Summary metrics for quick display in history
    index_count: Mapped[int] = mapped_column(Integer, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Optional extra data (e.g., user preferences, notes)
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
