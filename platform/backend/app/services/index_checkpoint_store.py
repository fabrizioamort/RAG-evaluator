"""Database-backed checkpoint store for resumable index builds."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Coroutine, TypeVar
from uuid import UUID

from rag_evaluator.common.indexing import (
    CheckpointChunk,
    CheckpointDocument,
    CheckpointStore,
    SourceDocument,
)
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.knowledge_base_index_checkpoint import (
    KnowledgeBaseIndexChunk,
    KnowledgeBaseIndexDocument,
)

T = TypeVar("T")


class DatabaseCheckpointStore(CheckpointStore):
    """Sync checkpoint API backed by the platform database.

    RAG implementations run in an executor thread. This store schedules all
    database work onto the owning event loop so the AsyncSession is not used
    directly from that worker thread.
    """

    def __init__(self, db: AsyncSession, index_id: UUID, loop: asyncio.AbstractEventLoop) -> None:
        self.db = db
        self.index_id = index_id
        self.loop = loop

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    @staticmethod
    def _document(row: KnowledgeBaseIndexDocument) -> CheckpointDocument:
        return CheckpointDocument(
            doc_key=row.doc_key,
            source_path=row.source_path,
            checksum=row.checksum,
            status=row.status,
            attempts=row.attempts,
            error_message=row.error_message,
            chunk_count=row.chunk_count,
            completed_chunks=row.completed_chunks,
        )

    @staticmethod
    def _chunk(row: KnowledgeBaseIndexChunk) -> CheckpointChunk:
        return CheckpointChunk(
            doc_key=row.doc_key,
            chunk_hash=row.chunk_hash,
            storage_id=row.storage_id,
            chunk_index=row.chunk_index,
            status=row.status,
            attempts=row.attempts,
            token_usage=row.token_usage,
            error_message=row.error_message,
        )

    def ensure_document(self, document: SourceDocument) -> CheckpointDocument:
        return self._run(self._ensure_document(document))

    async def _ensure_document(self, document: SourceDocument) -> CheckpointDocument:
        existing_for_path = await self.db.execute(
            select(KnowledgeBaseIndexDocument).where(
                KnowledgeBaseIndexDocument.index_id == self.index_id,
                KnowledgeBaseIndexDocument.source_path == document.source_path,
            )
        )
        path_row = existing_for_path.scalar_one_or_none()
        if path_row and path_row.checksum != document.checksum:
            raise ValueError(
                f"Source document changed since checkpoint was created: {document.source_path}. "
                "Force rebuild this index to continue."
            )

        result = await self.db.execute(
            select(KnowledgeBaseIndexDocument).where(
                KnowledgeBaseIndexDocument.index_id == self.index_id,
                KnowledgeBaseIndexDocument.doc_key == document.doc_key,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = KnowledgeBaseIndexDocument(
                index_id=self.index_id,
                doc_key=document.doc_key,
                source_path=document.source_path,
                checksum=document.checksum,
                status="pending",
            )
            self.db.add(row)
            await self.db.commit()
            await self.db.refresh(row)
        return self._document(row)

    def start_document(self, doc_key: str) -> None:
        self._run(self._start_document(doc_key))

    async def _start_document(self, doc_key: str) -> None:
        row = await self._get_document_row(doc_key)
        row.status = "building"
        row.attempts += 1
        row.error_message = None
        row.started_at = datetime.now(timezone.utc)
        row.completed_at = None
        await self.db.commit()

    def complete_document(self, doc_key: str, chunk_count: int) -> None:
        self._run(self._complete_document(doc_key, chunk_count))

    async def _complete_document(self, doc_key: str, chunk_count: int) -> None:
        row = await self._get_document_row(doc_key)
        row.status = "completed"
        row.chunk_count = chunk_count
        row.completed_chunks = chunk_count
        row.error_message = None
        row.completed_at = datetime.now(timezone.utc)
        await self.db.commit()

    def fail_document(self, doc_key: str, error: str) -> None:
        self._run(self._fail_document(doc_key, error))

    async def _fail_document(self, doc_key: str, error: str) -> None:
        row = await self._get_document_row(doc_key)
        row.status = "failed"
        row.error_message = error
        await self.db.commit()

    def completed_chunks(self, doc_key: str) -> dict[str, CheckpointChunk]:
        return self._run(self._completed_chunks(doc_key))

    async def _completed_chunks(self, doc_key: str) -> dict[str, CheckpointChunk]:
        result = await self.db.execute(
            select(KnowledgeBaseIndexChunk).where(
                KnowledgeBaseIndexChunk.index_id == self.index_id,
                KnowledgeBaseIndexChunk.doc_key == doc_key,
                KnowledgeBaseIndexChunk.status == "completed",
            )
        )
        rows = result.scalars().all()
        return {row.storage_id: self._chunk(row) for row in rows}

    def ensure_chunk(
        self,
        doc_key: str,
        chunk_hash: str,
        storage_id: str,
        chunk_index: int,
    ) -> CheckpointChunk:
        return self._run(self._ensure_chunk(doc_key, chunk_hash, storage_id, chunk_index))

    async def _ensure_chunk(
        self,
        doc_key: str,
        chunk_hash: str,
        storage_id: str,
        chunk_index: int,
    ) -> CheckpointChunk:
        result = await self.db.execute(
            select(KnowledgeBaseIndexChunk).where(
                KnowledgeBaseIndexChunk.index_id == self.index_id,
                KnowledgeBaseIndexChunk.storage_id == storage_id,
            )
        )
        row = result.scalar_one_or_none()
        if row and row.chunk_hash != chunk_hash:
            raise ValueError(
                f"Chunk content changed for storage ID {storage_id}. "
                "Force rebuild this index to continue."
            )
        if row is None:
            document = await self._get_document_row(doc_key)
            row = KnowledgeBaseIndexChunk(
                index_id=self.index_id,
                document_id=document.id,
                doc_key=doc_key,
                chunk_hash=chunk_hash,
                storage_id=storage_id,
                chunk_index=chunk_index,
                status="pending",
            )
            self.db.add(row)
            await self.db.commit()
            await self.db.refresh(row)
        return self._chunk(row)

    def start_chunk(self, storage_id: str) -> None:
        self._run(self._start_chunk(storage_id))

    async def _start_chunk(self, storage_id: str) -> None:
        row = await self._get_chunk_row(storage_id)
        row.status = "building"
        row.attempts += 1
        row.error_message = None
        row.started_at = datetime.now(timezone.utc)
        row.completed_at = None
        await self.db.commit()

    def complete_chunk(self, storage_id: str, token_usage: int = 0) -> None:
        self._run(self._complete_chunk(storage_id, token_usage))

    async def _complete_chunk(self, storage_id: str, token_usage: int = 0) -> None:
        row = await self._get_chunk_row(storage_id)
        row.status = "completed"
        row.token_usage = token_usage
        row.error_message = None
        row.completed_at = datetime.now(timezone.utc)

        completed_count = await self.db.scalar(
            select(func.count()).select_from(KnowledgeBaseIndexChunk).where(
                KnowledgeBaseIndexChunk.index_id == self.index_id,
                KnowledgeBaseIndexChunk.doc_key == row.doc_key,
                KnowledgeBaseIndexChunk.status == "completed",
            )
        )
        document = await self._get_document_row(row.doc_key)
        document.completed_chunks = int(completed_count or 0) + (
            0 if row.status == "completed" else 1
        )
        await self.db.commit()

    def fail_chunk(self, storage_id: str, error: str) -> None:
        self._run(self._fail_chunk(storage_id, error))

    async def _fail_chunk(self, storage_id: str, error: str) -> None:
        row = await self._get_chunk_row(storage_id)
        row.status = "failed"
        row.error_message = error
        await self.db.commit()

    def mark_chunk_pending(self, storage_id: str, error: str | None = None) -> None:
        self._run(self._mark_chunk_pending(storage_id, error))

    async def _mark_chunk_pending(self, storage_id: str, error: str | None = None) -> None:
        row = await self._get_chunk_row(storage_id)
        row.status = "pending"
        row.error_message = error
        row.completed_at = None
        await self.db.commit()

    def update_progress(
        self,
        current: int,
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run(self._update_progress(current, total, metadata))

    async def _update_progress(
        self,
        current: int,
        total: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        result = await self.db.execute(
            select(KnowledgeBaseIndex).where(KnowledgeBaseIndex.id == self.index_id)
        )
        index = result.scalar_one()
        index.progress_current = current
        index.progress_total = total
        index.last_heartbeat_at = datetime.now(timezone.utc)
        index.resume_metadata = metadata or {}
        await self.db.commit()

    def clear(self) -> None:
        self._run(self._clear())

    async def _clear(self) -> None:
        await self.db.execute(
            delete(KnowledgeBaseIndexChunk).where(
                KnowledgeBaseIndexChunk.index_id == self.index_id
            )
        )
        await self.db.execute(
            delete(KnowledgeBaseIndexDocument).where(
                KnowledgeBaseIndexDocument.index_id == self.index_id
            )
        )
        await self.db.commit()

    async def _get_document_row(self, doc_key: str) -> KnowledgeBaseIndexDocument:
        result = await self.db.execute(
            select(KnowledgeBaseIndexDocument).where(
                KnowledgeBaseIndexDocument.index_id == self.index_id,
                KnowledgeBaseIndexDocument.doc_key == doc_key,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise KeyError(f"Checkpoint document not found: {doc_key}")
        return row

    async def _get_chunk_row(self, storage_id: str) -> KnowledgeBaseIndexChunk:
        result = await self.db.execute(
            select(KnowledgeBaseIndexChunk).where(
                KnowledgeBaseIndexChunk.index_id == self.index_id,
                KnowledgeBaseIndexChunk.storage_id == storage_id,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise KeyError(f"Checkpoint chunk not found: {storage_id}")
        return row
