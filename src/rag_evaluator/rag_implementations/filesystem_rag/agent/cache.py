"""Session cache for Filesystem RAG agent.

This module provides caching of core index files to reduce
latency across queries within a session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SessionCache:
    """Cache for core index files loaded at session start.

    Caches frequently accessed files to reduce I/O latency:
    - corpus_overview.md
    - navigation_guide.md
    - _topic_map.md
    - _entity_registry.md

    Usage:
        cache = SessionCache("/path/to/prepared")
        cache.warm()  # Load core files
        content = cache.get("_meta/corpus_overview.md")
        context = cache.get_initial_context()  # For agent system prompt
    """

    # Files to cache at session start
    CORE_FILES = [
        "_meta/corpus_overview.md",
        "_meta/navigation_guide.md",
        "_index/topics/_topic_map.md",
        "_index/entities/_entity_registry.md",
    ]

    # Files to include in initial context (subset of core files)
    CONTEXT_FILES = [
        "_meta/corpus_overview.md",
        "_meta/navigation_guide.md",
    ]

    def __init__(self, prepared_path: str) -> None:
        """Initialize the session cache.

        Args:
            prepared_path: Path to the prepared filesystem root
        """
        self.prepared_path = Path(prepared_path)
        self._cache: dict[str, str] = {}
        self._loaded = False
        self._load_errors: list[str] = []

    @property
    def is_loaded(self) -> bool:
        """Check if the cache has been warmed."""
        return self._loaded

    @property
    def load_errors(self) -> list[str]:
        """Get any errors encountered during loading."""
        return self._load_errors.copy()

    def warm(self) -> bool:
        """Load core index files into cache.

        Returns:
            True if all core files loaded successfully, False otherwise

        Note:
            This method is idempotent - calling it multiple times
            has no effect after the first successful call.
        """
        if self._loaded:
            return len(self._load_errors) == 0

        self._load_errors = []

        for file_rel in self.CORE_FILES:
            file_path = self.prepared_path / file_rel
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    self._cache[file_rel] = content
                except Exception as e:
                    self._load_errors.append(f"Failed to load {file_rel}: {e}")
            else:
                self._load_errors.append(f"File not found: {file_rel}")

        self._loaded = True
        return len(self._load_errors) == 0

    def get(self, file_path: str) -> str | None:
        """Get cached file content.

        Args:
            file_path: Relative path to file (e.g., "_meta/corpus_overview.md")

        Returns:
            File content if cached, None otherwise

        Note:
            Call warm() before using get() to ensure cache is populated.
        """
        return self._cache.get(file_path)

    def get_initial_context(self) -> str:
        """Get concatenated core context for agent system prompt.

        Returns:
            Formatted string containing corpus overview and navigation guide,
            suitable for inclusion in the agent's system prompt.

        Note:
            Call warm() before using this method.
        """
        parts: list[str] = []

        for file_rel in self.CONTEXT_FILES:
            content = self._cache.get(file_rel)
            if content:
                parts.append(f"=== {file_rel} ===\n{content}")

        return "\n\n".join(parts)

    def get_topic_map(self) -> str | None:
        """Get cached topic map content.

        Returns:
            Topic map content if cached, None otherwise
        """
        return self._cache.get("_index/topics/_topic_map.md")

    def get_entity_registry(self) -> str | None:
        """Get cached entity registry content.

        Returns:
            Entity registry content if cached, None otherwise
        """
        return self._cache.get("_index/entities/_entity_registry.md")

    def get_corpus_overview(self) -> str | None:
        """Get cached corpus overview content.

        Returns:
            Corpus overview content if cached, None otherwise
        """
        return self._cache.get("_meta/corpus_overview.md")

    def get_navigation_guide(self) -> str | None:
        """Get cached navigation guide content.

        Returns:
            Navigation guide content if cached, None otherwise
        """
        return self._cache.get("_meta/navigation_guide.md")

    def invalidate(self) -> None:
        """Clear the cache and reset loaded state.

        Call this if the prepared filesystem has been modified
        and the cache needs to be reloaded.
        """
        self._cache.clear()
        self._loaded = False
        self._load_errors = []

    def reload(self) -> bool:
        """Invalidate and reload the cache.

        Returns:
            True if reload successful, False otherwise
        """
        self.invalidate()
        return self.warm()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "is_loaded": self._loaded,
            "files_cached": len(self._cache),
            "total_size_bytes": sum(len(v) for v in self._cache.values()),
            "cached_files": list(self._cache.keys()),
            "load_errors": self._load_errors,
        }

    def preload_file(self, file_path: str) -> bool:
        """Manually preload an additional file into cache.

        Args:
            file_path: Relative path to file

        Returns:
            True if file loaded successfully, False otherwise

        Note:
            This is useful for caching files discovered during queries
            that may be needed again.
        """
        full_path = self.prepared_path / file_path

        if not full_path.exists():
            return False

        try:
            content = full_path.read_text(encoding="utf-8")
            self._cache[file_path] = content
            return True
        except Exception:
            return False

    def is_cached(self, file_path: str) -> bool:
        """Check if a file is in the cache.

        Args:
            file_path: Relative path to file

        Returns:
            True if file is cached, False otherwise
        """
        return file_path in self._cache
