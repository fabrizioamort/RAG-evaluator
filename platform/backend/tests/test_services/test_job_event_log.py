"""Tests for JobEventLog service."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.services.job_event_log import JobEventLog


@pytest.fixture
def event_log(tmp_path: Path) -> JobEventLog:
    """Fixture for JobEventLog with a temporary storage path."""
    return JobEventLog(storage_path=tmp_path)


@pytest.mark.asyncio
async def test_log_event_persistence(event_log: JobEventLog) -> None:
    """Test that events are persisted to disk."""
    eval_id = uuid.uuid4()
    event_data = {"progress": 10, "message": "Hello"}

    await event_log.log_event(eval_id, "test_event", event_data)

    # Verify file exists
    log_path = event_log._get_log_path(eval_id)
    assert log_path.exists()

    # Verify content
    with open(log_path, "r", encoding="utf-8") as f:
        line = f.readline()
        event = json.loads(line)
        assert event["event_type"] == "test_event"
        assert event["progress"] == 10
        assert "timestamp" in event


@pytest.mark.asyncio
async def test_sse_streaming(event_log: JobEventLog) -> None:
    """Test that events are streamed to subscribers."""
    eval_id = uuid.uuid4()

    # Subscribe in a task
    events_received: list[dict[str, Any]] = []

    async def subscriber() -> None:
        async for event in event_log.subscribe(eval_id):
            events_received.append(event)
            if len(events_received) == 2:
                break

    sub_task = asyncio.create_task(subscriber())

    # Wait a bit for subscription to register
    await asyncio.sleep(0.1)

    # Log events
    await event_log.log_event(eval_id, "event1", {"val": 1})
    await event_log.log_event(eval_id, "event2", {"val": 2})

    # Wait for subscriber to finish
    await asyncio.wait_for(sub_task, timeout=2.0)

    assert len(events_received) == 2
    assert events_received[0]["event_type"] == "event1"
    assert events_received[1]["event_type"] == "event2"


@pytest.mark.asyncio
async def test_get_history(event_log: JobEventLog) -> None:
    """Test retrieving history from disk."""
    eval_id = uuid.uuid4()

    await event_log.log_event(eval_id, "e1", {"a": 1})
    await event_log.log_event(eval_id, "e2", {"a": 2})

    history = event_log.get_history(eval_id)
    assert len(history) == 2
    assert history[0]["event_type"] == "e1"
    assert history[1]["event_type"] == "e2"


@pytest.mark.asyncio
async def test_subscribe_replays_persisted_history(event_log: JobEventLog) -> None:
    """Subscribers should see persisted history after in-memory cache is gone."""
    eval_id = uuid.uuid4()

    await event_log.log_event(eval_id, "started", {"completed": 0})
    await event_log.log_event(eval_id, "test_case_error", {"error_message": "case failed"})
    event_log._cache.pop(str(eval_id))

    events_received = []
    async for event in event_log.subscribe(eval_id):
        events_received.append(event)
        if len(events_received) == 2:
            break

    assert [event["event_type"] for event in events_received] == ["started", "test_case_error"]
    assert events_received[1]["error_message"] == "case failed"


@pytest.mark.asyncio
async def test_reset_cache_removes_persisted_history(event_log: JobEventLog) -> None:
    """Retry reset should prevent old failure events from replaying."""
    eval_id = uuid.uuid4()

    await event_log.log_event(eval_id, "error", {"error_message": "old failure"})
    assert event_log._get_log_path(eval_id).exists()

    event_log.reset_cache(eval_id)

    assert str(eval_id) not in event_log._cache
    assert not event_log._get_log_path(eval_id).exists()
    assert event_log.get_history(eval_id) == []
