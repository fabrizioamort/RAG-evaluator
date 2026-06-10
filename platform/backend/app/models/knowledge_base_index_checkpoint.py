"""Checkpoint models for resumable knowledge base index builds."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.knowledge_base_index import KnowledgeBaseIndex


class KnowledgeBaseIndexDocument(BaseModel):
    """Checkpoint row for one source document in an index build."""

    __tablename__ = "knowledge_base_index_documents"
    __table_args__ = (
        UniqueConstraint("index_id", "doc_key", name="uq_kbi_doc_index_doc_key"),
        Index("idx_kbi_doc_index_status", "index_id", "status"),
    )

    index_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_indexes.id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_key: Mapped[str] = mapped_column(String(128), nullable=False)
    source_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    index: Mapped["KnowledgeBaseIndex"] = relationship(
        "KnowledgeBaseIndex",
        back_populates="checkpoint_documents",
    )
    chunks: Mapped[list["KnowledgeBaseIndexChunk"]] = relationship(
        "KnowledgeBaseIndexChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class KnowledgeBaseIndexChunk(BaseModel):
    """Checkpoint row for one chunk or point in an index build."""

    __tablename__ = "knowledge_base_index_chunks"
    __table_args__ = (
        UniqueConstraint("index_id", "storage_id", name="uq_kbi_chunk_index_storage_id"),
        Index("idx_kbi_chunk_index_status", "index_id", "status"),
        Index("idx_kbi_chunk_doc_key", "index_id", "doc_key"),
    )

    index_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_indexes.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_base_index_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_key: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_usage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped[KnowledgeBaseIndexDocument] = relationship(
        "KnowledgeBaseIndexDocument",
        back_populates="chunks",
    )
    index: Mapped["KnowledgeBaseIndex"] = relationship(
        "KnowledgeBaseIndex",
        back_populates="checkpoint_chunks",
    )
