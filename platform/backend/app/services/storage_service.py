"""Storage service for file management."""

import hashlib
import uuid
from pathlib import Path

import aiofiles

from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class StorageService:
    """Service for managing file storage operations."""

    def __init__(self, base_path: str | None = None) -> None:
        """Initialize storage service.

        Args:
            base_path: Base storage path. Defaults to settings.STORAGE_PATH.
        """
        self.base_path = Path(base_path or settings.STORAGE_PATH)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required storage directories exist."""
        for subdir in ["documents", "indexes", "artifacts", "reports"]:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)

    def get_documents_path(self, kb_id: uuid.UUID) -> Path:
        """Get the documents directory for a knowledge base.

        Args:
            kb_id: Knowledge base UUID.

        Returns:
            Path to the KB's documents directory.
        """
        path = self.base_path / "documents" / str(kb_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_index_path(self, kb_id: uuid.UUID) -> Path:
        """Get the index directory for a knowledge base.

        Args:
            kb_id: Knowledge base UUID.

        Returns:
            Path to the KB's index directory.
        """
        path = self.base_path / "indexes" / str(kb_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def generate_unique_filename(self, original_filename: str) -> str:
        """Generate a unique filename while preserving the extension.

        Args:
            original_filename: Original filename from upload.

        Returns:
            Unique filename with UUID prefix.
        """
        # Extract extension
        path = Path(original_filename)
        extension = path.suffix.lower()
        stem = path.stem

        # Sanitize stem (remove problematic characters)
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        # Truncate if too long
        safe_stem = safe_stem[:100]

        # Generate unique name
        unique_id = uuid.uuid4().hex[:8]
        return f"{unique_id}_{safe_stem}{extension}"

    @staticmethod
    def calculate_checksum(content: bytes) -> str:
        """Calculate SHA256 checksum of content.

        Args:
            content: File content as bytes.

        Returns:
            Hex-encoded SHA256 checksum.
        """
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    async def calculate_checksum_async(file_path: Path) -> str:
        """Calculate SHA256 checksum of a file asynchronously.

        Args:
            file_path: Path to the file.

        Returns:
            Hex-encoded SHA256 checksum.
        """
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            # Read in chunks for large files
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def save_file(
        self,
        kb_id: uuid.UUID,
        filename: str,
        content: bytes,
    ) -> tuple[Path, str, int]:
        """Save a file to storage.

        Args:
            kb_id: Knowledge base UUID.
            filename: Original filename.
            content: File content as bytes.

        Returns:
            Tuple of (file_path, checksum, size_bytes).
        """
        # Generate unique filename and get target directory
        unique_filename = self.generate_unique_filename(filename)
        docs_path = self.get_documents_path(kb_id)
        file_path = docs_path / unique_filename

        # Calculate checksum before saving
        checksum = self.calculate_checksum(content)
        size_bytes = len(content)

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        logger.info(
            "Saved file",
            kb_id=str(kb_id),
            filename=unique_filename,
            size_bytes=size_bytes,
            checksum=checksum[:16] + "...",
        )

        return file_path, checksum, size_bytes

    async def delete_file(self, file_path: str | Path) -> bool:
        """Delete a file from storage.

        Args:
            file_path: Path to the file to delete.

        Returns:
            True if deleted successfully, False if file didn't exist.
        """
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info("Deleted file", file_path=str(path))
            return True
        logger.warning("File not found for deletion", file_path=str(path))
        return False

    async def delete_kb_storage(self, kb_id: uuid.UUID) -> None:
        """Delete all storage for a knowledge base.

        Args:
            kb_id: Knowledge base UUID.
        """
        import shutil

        # Delete documents directory
        docs_path = self.base_path / "documents" / str(kb_id)
        if docs_path.exists():
            shutil.rmtree(docs_path)
            logger.info("Deleted KB documents", kb_id=str(kb_id))

        # Delete indexes directory
        index_path = self.base_path / "indexes" / str(kb_id)
        if index_path.exists():
            shutil.rmtree(index_path)
            logger.info("Deleted KB indexes", kb_id=str(kb_id))

    def get_relative_path(self, absolute_path: Path) -> str:
        """Convert absolute path to relative path from base storage.

        Args:
            absolute_path: Absolute file path.

        Returns:
            Relative path string from storage root.
        """
        try:
            return str(absolute_path.relative_to(self.base_path))
        except ValueError:
            # Path is not relative to base_path
            return str(absolute_path)


# Singleton instance for dependency injection
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
