"""Job event logging service for tracking evaluation progress and streaming events via SSE."""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobEventLog:
    """Service for persisting and streaming job events."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """Initialize the job event log service.

        Args:
            storage_path: Base path for persisting event logs.
                         Defaults to settings.STORAGE_PATH / "logs" / "jobs".
        """
        self.storage_path = storage_path or (Path(settings.STORAGE_PATH) / "logs" / "jobs")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory queues for active SSE streams
        # evaluation_id -> list of asyncio.Queue
        self._queues: Dict[str, List[asyncio.Queue[Dict[str, Any]]]] = defaultdict(list)

        # Cache for events to allow late-joiners to see history if needed
        # In a production system, this would be backed by a DB or Redis
        self._cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _get_log_path(self, evaluation_id: UUID) -> Path:
        """Get the path to the log file for an evaluation."""
        return self.storage_path / f"{evaluation_id}.jsonl"

    async def log_event(self, evaluation_id: UUID, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event and broadcast it to active listeners.

        Args:
            evaluation_id: ID of the evaluation.
            event_type: Type of event (e.g., 'started', 'progress', 'completed').
            data: Event payload.
        """
        event = {
            "evaluation_id": str(evaluation_id),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }

        # 1. Persist to disk (JSONL)
        log_path = self._get_log_path(evaluation_id)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(
                "Failed to persist job event", evaluation_id=str(evaluation_id), error=str(e)
            )

        # 2. Add to cache
        eval_id_str = str(evaluation_id)
        self._cache[eval_id_str].append(event)

        # Limit cache size per evaluation (e.g., last 100 events)
        if len(self._cache[eval_id_str]) > 100:
            self._cache[eval_id_str].pop(0)

        # 3. Broadcast to all active queues
        if eval_id_str in self._queues:
            for queue in self._queues[eval_id_str]:
                await queue.put(event)

    async def subscribe(
        self, evaluation_id: UUID, last_event_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to events for an evaluation.

        Args:
            evaluation_id: ID of the evaluation to watch.
            last_event_id: Not implemented yet, for reconnection.

        Yields:
            Event dictionaries.
        """
        eval_id_str = str(evaluation_id)
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._queues[eval_id_str].append(queue)

        try:
            # First, yield cached events to catch up
            # In a more robust implementation, we'd use last_event_id and the file on disk
            for cached_event in self._cache[eval_id_str]:
                yield cached_event

            # Then wait for new events
            while True:
                event = await queue.get()
                yield event
        finally:
            # Clean up on disconnect
            if eval_id_str in self._queues:
                self._queues[eval_id_str].remove(queue)
                if not self._queues[eval_id_str]:
                    del self._queues[eval_id_str]

    def get_history(self, evaluation_id: UUID) -> List[Dict[str, Any]]:
        """Get the full event history from disk for an evaluation."""
        log_path = self._get_log_path(evaluation_id)
        if not log_path.exists():
            return []

        history = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
        except Exception as e:
            logger.error(
                "Failed to read job history", evaluation_id=str(evaluation_id), error=str(e)
            )

        return history


# Singleton instance
_job_event_log: Optional[JobEventLog] = None


def get_job_event_log() -> JobEventLog:
    """Get the job event log service singleton."""
    global _job_event_log
    if _job_event_log is None:
        _job_event_log = JobEventLog()
    return _job_event_log
