"""Artifact model for content-addressed storage."""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModelNoUpdate


class Artifact(BaseModelNoUpdate):
    """Content-addressed artifact for storing large blobs."""

    __tablename__ = "artifacts"

    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(100), default="application/json", nullable=False
    )
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
