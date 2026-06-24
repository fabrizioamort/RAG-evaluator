"""Shared helpers for resumable index preparation."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

VALID_DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".docx"}


@dataclass(frozen=True)
class SourceDocument:
    """Stable identity for a source file being indexed."""

    doc_key: str
    source_path: str
    relative_path: str
    checksum: str


@dataclass
class CheckpointDocument:
    """Persisted checkpoint state for a source document."""

    doc_key: str
    source_path: str
    checksum: str
    status: str
    attempts: int = 0
    error_message: str | None = None
    chunk_count: int = 0
    completed_chunks: int = 0


@dataclass
class CheckpointChunk:
    """Persisted checkpoint state for one stored retrieval unit."""

    doc_key: str
    chunk_hash: str
    storage_id: str
    chunk_index: int
    status: str
    attempts: int = 0
    token_usage: int = 0
    error_message: str | None = None


class CheckpointStore(Protocol):
    """Synchronous checkpoint API used by RAG builders.

    Implementations may persist to JSON, a database, or another durable store.
    Methods are synchronous because RAG builders currently run in a worker thread.
    """

    def ensure_document(self, document: SourceDocument) -> CheckpointDocument:
        """Create or validate a document checkpoint row."""

    def start_document(self, doc_key: str) -> None:
        """Mark a document as in progress and increment its attempt counter."""

    def complete_document(self, doc_key: str, chunk_count: int) -> None:
        """Mark a document as complete after all of its chunks are durable."""

    def fail_document(self, doc_key: str, error: str) -> None:
        """Mark a document as failed."""

    def completed_chunks(self, doc_key: str) -> dict[str, CheckpointChunk]:
        """Return completed chunks for a document, keyed by storage ID."""

    def ensure_chunk(
        self,
        doc_key: str,
        chunk_hash: str,
        storage_id: str,
        chunk_index: int,
    ) -> CheckpointChunk:
        """Create or validate a chunk checkpoint row."""

    def start_chunk(self, storage_id: str) -> None:
        """Mark a chunk as in progress and increment its attempt counter."""

    def complete_chunk(self, storage_id: str, token_usage: int = 0) -> None:
        """Mark a chunk as complete after it is stored."""

    def fail_chunk(self, storage_id: str, error: str) -> None:
        """Mark a chunk as failed."""

    def mark_chunk_pending(self, storage_id: str, error: str | None = None) -> None:
        """Return a previously completed chunk to pending state."""

    def update_progress(
        self,
        current: int,
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist lightweight progress counters."""

    def clear(self) -> None:
        """Remove all persisted checkpoint state."""


def file_checksum(path: Path) -> str:
    """Return a SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: str, length: int | None = None) -> str:
    """Return a deterministic SHA256 hex digest for text parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    value = digest.hexdigest()
    return value[:length] if length else value


def storage_id(prefix: str, *parts: str, length: int = 48) -> str:
    """Return a deterministic storage ID safe for vector stores."""
    return f"{prefix}_{stable_hash(*parts, length=length)}"


def discover_source_documents(documents_path: str) -> list[SourceDocument]:
    """Find supported source documents and assign stable document keys."""
    docs_path = Path(documents_path)
    if not docs_path.exists():
        raise ValueError(f"Documents path does not exist: {documents_path}")

    documents: list[SourceDocument] = []
    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_DOCUMENT_EXTENSIONS:
            continue
        relative_path = file_path.relative_to(docs_path).as_posix()
        checksum = file_checksum(file_path)
        documents.append(
            SourceDocument(
                doc_key=storage_id("doc", relative_path, checksum),
                source_path=str(file_path),
                relative_path=relative_path,
                checksum=checksum,
            )
        )

    if not documents:
        raise ValueError(f"No documents found in {documents_path}")

    return documents


class JsonCheckpointStore:
    """Durable JSON checkpoint store for local CLI builds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"documents": {}, "chunks": {}, "progress": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("documents", {})
        data.setdefault("chunks", {})
        data.setdefault("progress", {})
        return cast(dict[str, Any], data)

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=True)
        os.replace(temp_path, self.path)

    def ensure_document(self, document: SourceDocument) -> CheckpointDocument:
        with self._lock:
            docs = self._state["documents"]
            for saved in docs.values():
                if (
                    saved.get("source_path") == document.source_path
                    and saved.get("checksum") != document.checksum
                ):
                    raise ValueError(
                        "Source document changed since checkpoint was created; "
                        "run a force rebuild to continue"
                    )
            existing = docs.get(document.doc_key)
            if existing and existing["checksum"] != document.checksum:
                raise ValueError(
                    "Source document changed since checkpoint was created; "
                    "run a force rebuild to continue"
                )
            if not existing:
                existing = {
                    "doc_key": document.doc_key,
                    "source_path": document.source_path,
                    "relative_path": document.relative_path,
                    "checksum": document.checksum,
                    "status": "pending",
                    "attempts": 0,
                    "error_message": None,
                    "chunk_count": 0,
                    "completed_chunks": 0,
                }
                docs[document.doc_key] = existing
                self._save_locked()
            document_data: dict[str, Any] = {
                k: existing[k] for k in CheckpointDocument.__dataclass_fields__
            }
            return CheckpointDocument(**document_data)

    def start_document(self, doc_key: str) -> None:
        with self._lock:
            doc = self._state["documents"][doc_key]
            doc["status"] = "building"
            doc["attempts"] = int(doc.get("attempts", 0)) + 1
            doc["error_message"] = None
            self._save_locked()

    def complete_document(self, doc_key: str, chunk_count: int) -> None:
        with self._lock:
            doc = self._state["documents"][doc_key]
            doc["status"] = "completed"
            doc["chunk_count"] = chunk_count
            doc["completed_chunks"] = chunk_count
            doc["error_message"] = None
            self._save_locked()

    def fail_document(self, doc_key: str, error: str) -> None:
        with self._lock:
            doc = self._state["documents"][doc_key]
            doc["status"] = "failed"
            doc["error_message"] = error
            self._save_locked()

    def completed_chunks(self, doc_key: str) -> dict[str, CheckpointChunk]:
        with self._lock:
            return {
                storage_id_: CheckpointChunk(**chunk)
                for storage_id_, chunk in self._state["chunks"].items()
                if chunk["doc_key"] == doc_key and chunk["status"] == "completed"
            }

    def ensure_chunk(
        self,
        doc_key: str,
        chunk_hash: str,
        storage_id: str,
        chunk_index: int,
    ) -> CheckpointChunk:
        with self._lock:
            chunks = self._state["chunks"]
            existing = chunks.get(storage_id)
            if existing and existing["chunk_hash"] != chunk_hash:
                raise ValueError(
                    "Chunk content changed for an existing storage ID; run a force rebuild"
                )
            if not existing:
                existing = {
                    "doc_key": doc_key,
                    "chunk_hash": chunk_hash,
                    "storage_id": storage_id,
                    "chunk_index": chunk_index,
                    "status": "pending",
                    "attempts": 0,
                    "token_usage": 0,
                    "error_message": None,
                }
                chunks[storage_id] = existing
                self._save_locked()
            return CheckpointChunk(**existing)

    def start_chunk(self, storage_id: str) -> None:
        with self._lock:
            chunk = self._state["chunks"][storage_id]
            chunk["status"] = "building"
            chunk["attempts"] = int(chunk.get("attempts", 0)) + 1
            chunk["error_message"] = None
            self._save_locked()

    def complete_chunk(self, storage_id: str, token_usage: int = 0) -> None:
        with self._lock:
            chunk = self._state["chunks"][storage_id]
            chunk["status"] = "completed"
            chunk["token_usage"] = token_usage
            chunk["error_message"] = None
            self._save_locked()

    def fail_chunk(self, storage_id: str, error: str) -> None:
        with self._lock:
            chunk = self._state["chunks"][storage_id]
            chunk["status"] = "failed"
            chunk["error_message"] = error
            self._save_locked()

    def mark_chunk_pending(self, storage_id: str, error: str | None = None) -> None:
        with self._lock:
            chunk = self._state["chunks"][storage_id]
            chunk["status"] = "pending"
            chunk["error_message"] = error
            self._save_locked()

    def update_progress(
        self,
        current: int,
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._state["progress"] = {
                "current": current,
                "total": total,
                "metadata": metadata or {},
            }
            self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self._state = {"documents": {}, "chunks": {}, "progress": {}}
            self._save_locked()
