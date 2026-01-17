"""Token usage tracking utilities.

This module provides utilities for tracking token usage
across RAG operations for cost calculation and monitoring.
"""

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Track token usage for cost calculation.

    Accumulates token counts across prompt, completion, and embedding
    operations to enable accurate cost tracking.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        """Get total token count.

        Returns:
            Sum of all token types
        """
        with self._lock:
            return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Add another TokenUsage to this one.

        Creates a new TokenUsage instance with combined counts.

        Args:
            other: Another TokenUsage to add

        Returns:
            New TokenUsage with combined counts
        """
        with self._lock:
            with other._lock:
                return TokenUsage(
                    prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                    completion_tokens=self.completion_tokens + other.completion_tokens,
                    embedding_tokens=self.embedding_tokens + other.embedding_tokens,
                )

    def add_prompt_tokens(self, count: int) -> None:
        """Add prompt tokens.

        Args:
            count: Number of prompt tokens to add
        """
        with self._lock:
            self.prompt_tokens += count

    def add_completion_tokens(self, count: int) -> None:
        """Add completion tokens.

        Args:
            count: Number of completion tokens to add
        """
        with self._lock:
            self.completion_tokens += count

    def add_embedding_tokens(self, count: int) -> None:
        """Add embedding tokens.

        Args:
            count: Number of embedding tokens to add
        """
        with self._lock:
            self.embedding_tokens += count

    def reset(self) -> None:
        """Reset all token counts to zero."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.embedding_tokens = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with all token counts
        """
        with self._lock:
            # Calculate total_tokens directly while holding the lock
            # to avoid re-entering the lock via the property (though RLock allows it)
            prompt = self.prompt_tokens
            completion = self.completion_tokens
            embedding = self.embedding_tokens
            total = prompt + completion + embedding

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "embedding_tokens": embedding,
                "total_tokens": total,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        """Create TokenUsage from dictionary.

        Args:
            data: Dictionary with token counts

        Returns:
            New TokenUsage instance
        """
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            embedding_tokens=data.get("embedding_tokens", 0),
        )

    def __repr__(self) -> str:
        """String representation of token usage.

        Returns:
            Formatted string with token counts
        """
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"embedding={self.embedding_tokens}, "
            f"total={self.total_tokens})"
        )
