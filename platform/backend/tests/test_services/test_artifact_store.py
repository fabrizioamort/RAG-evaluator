"""Tests for the artifact store service."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.artifact_store import ArtifactStore


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> str:
    """Create a temporary storage path for testing."""
    return str(tmp_path / "artifacts")


@pytest.fixture
def artifact_store(temp_storage_path: str) -> ArtifactStore:
    """Create an artifact store with temporary storage."""
    return ArtifactStore(base_path=temp_storage_path)


class TestArtifactStoreBasics:
    """Basic functionality tests for artifact store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_json(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test storing and retrieving JSON data."""
        test_data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

        # Store
        artifact = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RAW_METRICS
        )

        assert artifact is not None
        assert artifact.id is not None
        assert artifact.kind == ArtifactStore.KIND_RAW_METRICS
        assert artifact.content_type == "application/json"
        assert artifact.size_bytes is not None and artifact.size_bytes > 0
        assert len(artifact.storage_key) == 64  # SHA256 hex length

        # Retrieve
        retrieved = await artifact_store.retrieve_json(artifact.storage_key)
        assert retrieved == test_data

    @pytest.mark.asyncio
    async def test_store_and_retrieve_bytes(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test storing and retrieving raw bytes."""
        test_data = "Plain text content for testing"

        artifact = await artifact_store.store(
            db_session,
            test_data,
            kind=ArtifactStore.KIND_PROVENANCE,
            content_type="text/plain",
        )

        # Retrieve raw bytes
        content = await artifact_store.retrieve(artifact.storage_key)
        assert content is not None
        assert content.decode("utf-8") == test_data

    @pytest.mark.asyncio
    async def test_retrieve_by_id(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test retrieving artifact by ID."""
        test_data = {"test": "data"}

        artifact = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RETRIEVED_CONTEXT
        )

        # Retrieve by ID
        retrieved = await artifact_store.retrieve_json_by_id(db_session, artifact.id)
        assert retrieved == test_data

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_key(self, artifact_store: ArtifactStore) -> None:
        """Test retrieving with non-existent key returns None."""
        fake_key = "0" * 64  # Valid SHA256 format but doesn't exist

        result = await artifact_store.retrieve(fake_key)
        assert result is None

        result_json = await artifact_store.retrieve_json(fake_key)
        assert result_json is None

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_id(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test retrieving with non-existent ID returns None."""
        fake_id = uuid4()

        result = await artifact_store.retrieve_by_id(db_session, fake_id)
        assert result is None

        result_json = await artifact_store.retrieve_json_by_id(db_session, fake_id)
        assert result_json is None


class TestArtifactDeduplication:
    """Tests for content-addressed deduplication."""

    @pytest.mark.asyncio
    async def test_deduplication_same_content(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test that storing the same content twice returns the same artifact."""
        test_data = {"dedupe": "test", "value": 123}

        # Store first time
        artifact1 = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RAW_METRICS
        )

        # Store same content again
        artifact2 = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RAW_METRICS
        )

        # Should return the same artifact (deduplication)
        assert artifact1.id == artifact2.id
        assert artifact1.storage_key == artifact2.storage_key

    @pytest.mark.asyncio
    async def test_different_content_different_artifacts(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test that different content creates different artifacts."""
        data1 = {"content": "first"}
        data2 = {"content": "second"}

        artifact1 = await artifact_store.store_json(
            db_session, data1, kind=ArtifactStore.KIND_RAW_METRICS
        )
        artifact2 = await artifact_store.store_json(
            db_session, data2, kind=ArtifactStore.KIND_RAW_METRICS
        )

        # Should be different artifacts
        assert artifact1.id != artifact2.id
        assert artifact1.storage_key != artifact2.storage_key

    @pytest.mark.asyncio
    async def test_deduplication_different_kinds(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test that same content with different kind still deduplicates."""
        test_data = {"same": "content"}

        # Store with one kind
        artifact1 = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RAW_METRICS
        )

        # Store same content with different kind - should still deduplicate
        # because storage_key is based on content, not kind
        artifact2 = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RETRIEVED_CONTEXT
        )

        # Same storage key means same artifact
        assert artifact1.storage_key == artifact2.storage_key
        assert artifact1.id == artifact2.id


class TestArtifactLargeContent:
    """Tests for handling large content."""

    @pytest.mark.asyncio
    async def test_large_json_content(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test storing and retrieving large JSON content."""
        # Create large data (~1MB)
        large_data = {"items": [{"id": i, "data": "x" * 1000} for i in range(1000)]}

        artifact = await artifact_store.store_json(
            db_session, large_data, kind=ArtifactStore.KIND_RETRIEVED_CONTEXT
        )

        assert artifact.size_bytes is not None and artifact.size_bytes > 1_000_000  # > 1MB

        # Retrieve and verify
        retrieved = await artifact_store.retrieve_json(artifact.storage_key)
        assert retrieved == large_data

    @pytest.mark.asyncio
    async def test_unicode_content(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test storing and retrieving unicode content."""
        unicode_data = {
            "greeting": "Hello, 世界! 🌍",
            "arabic": "مرحبا بالعالم",
            "emoji": "🎉🚀✨",
        }

        artifact = await artifact_store.store_json(
            db_session, unicode_data, kind=ArtifactStore.KIND_PROVENANCE
        )

        retrieved = await artifact_store.retrieve_json(artifact.storage_key)
        assert retrieved == unicode_data


class TestArtifactDelete:
    """Tests for artifact deletion."""

    @pytest.mark.asyncio
    async def test_delete_artifact(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test deleting an artifact."""
        test_data = {"to_delete": True}

        artifact = await artifact_store.store_json(
            db_session, test_data, kind=ArtifactStore.KIND_RAW_METRICS
        )
        storage_key = artifact.storage_key
        artifact_id = artifact.id

        # Verify it exists
        assert await artifact_store.exists(storage_key)

        # Delete
        result = await artifact_store.delete(db_session, artifact_id)
        assert result is True

        # Verify it's gone from filesystem
        assert not await artifact_store.exists(storage_key)

        # Verify retrieval returns None
        assert await artifact_store.retrieve(storage_key) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test deleting non-existent artifact returns False."""
        fake_id = uuid4()

        result = await artifact_store.delete(db_session, fake_id)
        assert result is False


class TestArtifactExists:
    """Tests for existence checks."""

    @pytest.mark.asyncio
    async def test_exists_true(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test exists returns True for stored artifact."""
        artifact = await artifact_store.store_json(
            db_session, {"test": "data"}, kind=ArtifactStore.KIND_RAW_METRICS
        )

        assert await artifact_store.exists(artifact.storage_key)

    @pytest.mark.asyncio
    async def test_exists_false(self, artifact_store: ArtifactStore) -> None:
        """Test exists returns False for non-existent key."""
        fake_key = "a" * 64

        assert not await artifact_store.exists(fake_key)


class TestArtifactStorageStats:
    """Tests for storage statistics."""

    @pytest.mark.asyncio
    async def test_storage_stats_empty(self, artifact_store: ArtifactStore) -> None:
        """Test stats on empty storage."""
        stats = artifact_store.get_storage_stats()

        assert stats["file_count"] == 0
        assert stats["total_size_bytes"] == 0
        assert "base_path" in stats

    @pytest.mark.asyncio
    async def test_storage_stats_with_artifacts(
        self, artifact_store: ArtifactStore, db_session: AsyncSession
    ) -> None:
        """Test stats after storing artifacts."""
        # Store multiple artifacts
        await artifact_store.store_json(db_session, {"a": 1}, kind=ArtifactStore.KIND_RAW_METRICS)
        await artifact_store.store_json(
            db_session, {"b": 2}, kind=ArtifactStore.KIND_RETRIEVED_CONTEXT
        )
        await artifact_store.store_json(
            db_session, {"c": 3}, kind=ArtifactStore.KIND_RETRIEVAL_TRACE
        )

        stats = artifact_store.get_storage_stats()

        assert stats["file_count"] == 3
        assert stats["total_size_bytes"] > 0


class TestArtifactKinds:
    """Tests for artifact kind constants."""

    def test_kind_constants_defined(self) -> None:
        """Test that all expected kind constants are defined."""
        assert ArtifactStore.KIND_RETRIEVED_CONTEXT == "retrieved_context"
        assert ArtifactStore.KIND_RAW_METRICS == "raw_metrics"
        assert ArtifactStore.KIND_RETRIEVAL_TRACE == "retrieval_trace"
        assert ArtifactStore.KIND_PROVENANCE == "provenance"
