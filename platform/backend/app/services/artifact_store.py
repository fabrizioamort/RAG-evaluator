"""Content-addressed artifact storage service."""

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.artifact import Artifact
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ArtifactStore:
    """Content-addressed storage for evaluation artifacts.

    Heavy blobs (retrieved_context, raw_metrics, retrieval traces) are stored
    on the filesystem with SHA256 keys, keeping the database fast and enabling
    deduplication.

    Storage structure:
        {base_path}/artifacts/{key[:2]}/{key}

    The first two characters of the key are used as a subdirectory to avoid
    having too many files in a single directory.
    """

    # Artifact kinds
    KIND_RETRIEVED_CONTEXT = "retrieved_context"
    KIND_RAW_METRICS = "raw_metrics"
    KIND_RETRIEVAL_TRACE = "retrieval_trace"
    KIND_PROVENANCE = "provenance"

    def __init__(self, base_path: str | None = None) -> None:
        """Initialize artifact store.

        Args:
            base_path: Base storage path. Defaults to settings.STORAGE_PATH.
        """
        self.base_path = Path(base_path or settings.STORAGE_PATH) / "artifacts"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, content: bytes) -> str:
        """Compute SHA256 content-addressed key.

        Args:
            content: Raw bytes content.

        Returns:
            Hex-encoded SHA256 hash (64 characters).
        """
        return hashlib.sha256(content).hexdigest()

    def _get_file_path(self, storage_key: str) -> Path:
        """Get filesystem path for a storage key.

        Args:
            storage_key: SHA256 content-addressed key.

        Returns:
            Path to the artifact file.
        """
        # Use first 2 chars as subdirectory for better filesystem distribution
        return self.base_path / storage_key[:2] / storage_key

    async def store(
        self,
        db: AsyncSession,
        data: Any,
        kind: str,
        content_type: str = "application/json",
    ) -> Artifact:
        """Store data and return artifact reference.

        If the content already exists (same SHA256), returns the existing artifact
        (deduplication).

        Args:
            db: Database session.
            data: Data to store. Will be JSON-encoded if content_type is application/json.
            kind: Artifact kind (retrieved_context, raw_metrics, retrieval_trace, provenance).
            content_type: MIME content type.

        Returns:
            Artifact model instance (existing or newly created).
        """
        # Serialize content
        if content_type == "application/json":
            content = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        else:
            content = str(data).encode("utf-8")

        # Compute content-addressed key
        storage_key = self._compute_key(content)

        # Check if artifact already exists (deduplication)
        query = select(Artifact).where(Artifact.storage_key == storage_key)
        result = await db.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            logger.debug(
                "Artifact already exists (deduplicated)",
                storage_key=storage_key[:16] + "...",
                kind=kind,
            )
            return existing

        # Store on filesystem
        file_path = self._get_file_path(storage_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Create database record
        artifact = Artifact(
            kind=kind,
            storage_key=storage_key,
            content_type=content_type,
            size_bytes=len(content),
        )
        db.add(artifact)
        await db.flush()
        await db.refresh(artifact)

        logger.info(
            "Stored artifact",
            artifact_id=str(artifact.id),
            storage_key=storage_key[:16] + "...",
            kind=kind,
            size_bytes=len(content),
        )

        return artifact

    async def store_json(
        self,
        db: AsyncSession,
        data: Any,
        kind: str,
    ) -> Artifact:
        """Convenience method to store JSON data.

        Args:
            db: Database session.
            data: Data to store (will be JSON-encoded).
            kind: Artifact kind.

        Returns:
            Artifact model instance.
        """
        return await self.store(db, data, kind, content_type="application/json")

    async def retrieve(self, storage_key: str) -> bytes | None:
        """Retrieve artifact content by storage key.

        Args:
            storage_key: SHA256 content-addressed key.

        Returns:
            Raw bytes content, or None if not found.
        """
        file_path = self._get_file_path(storage_key)

        if not file_path.exists():
            logger.warning("Artifact not found", storage_key=storage_key[:16] + "...")
            return None

        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        return content

    async def retrieve_json(self, storage_key: str) -> Any | None:
        """Retrieve and parse JSON artifact.

        Args:
            storage_key: SHA256 content-addressed key.

        Returns:
            Parsed JSON data, or None if not found.
        """
        content = await self.retrieve(storage_key)
        if content is None:
            return None

        return json.loads(content.decode("utf-8"))

    async def retrieve_by_id(self, db: AsyncSession, artifact_id: UUID) -> bytes | None:
        """Retrieve artifact content by artifact ID.

        Args:
            db: Database session.
            artifact_id: Artifact UUID.

        Returns:
            Raw bytes content, or None if not found.
        """
        query = select(Artifact).where(Artifact.id == artifact_id)
        result = await db.execute(query)
        artifact = result.scalar_one_or_none()

        if not artifact:
            return None

        return await self.retrieve(artifact.storage_key)

    async def retrieve_json_by_id(self, db: AsyncSession, artifact_id: UUID) -> Any | None:
        """Retrieve and parse JSON artifact by artifact ID.

        Args:
            db: Database session.
            artifact_id: Artifact UUID.

        Returns:
            Parsed JSON data, or None if not found.
        """
        content = await self.retrieve_by_id(db, artifact_id)
        if content is None:
            return None

        return json.loads(content.decode("utf-8"))

    async def delete(self, db: AsyncSession, artifact_id: UUID) -> bool:
        """Delete an artifact from storage and database.

        Note: This bypasses deduplication - if other records reference the same
        storage_key, the file will still be deleted. Use with caution.

        Args:
            db: Database session.
            artifact_id: Artifact UUID.

        Returns:
            True if deleted, False if not found.
        """
        query = select(Artifact).where(Artifact.id == artifact_id)
        result = await db.execute(query)
        artifact = result.scalar_one_or_none()

        if not artifact:
            return False

        # Delete file
        file_path = self._get_file_path(artifact.storage_key)
        if file_path.exists():
            file_path.unlink()

        # Delete database record
        await db.delete(artifact)
        await db.flush()

        logger.info(
            "Deleted artifact",
            artifact_id=str(artifact_id),
            storage_key=artifact.storage_key[:16] + "...",
        )

        return True

    async def exists(self, storage_key: str) -> bool:
        """Check if an artifact exists in storage.

        Args:
            storage_key: SHA256 content-addressed key.

        Returns:
            True if the artifact file exists.
        """
        file_path = self._get_file_path(storage_key)
        return file_path.exists()

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        total_size = 0
        file_count = 0

        for subdir in self.base_path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1

        return {
            "base_path": str(self.base_path),
            "file_count": file_count,
            "total_size_bytes": total_size,
        }


# Singleton instance for dependency injection
_artifact_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Get or create artifact store singleton."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store
