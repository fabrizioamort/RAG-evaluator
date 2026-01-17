"""Unit tests for WebhookService."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.webhook import Webhook
from app.services.webhook_service import WebhookService


@pytest.fixture
def webhook_service() -> WebhookService:
    return WebhookService()


@pytest.fixture
def mock_db() -> AsyncMock:
    return AsyncMock(spec=AsyncSession)


@pytest.mark.asyncio
async def test_sign_payload(webhook_service: WebhookService) -> None:
    """Test HMAC-SHA256 signing."""
    payload = {"event": "test", "data": 123}
    secret = "test_secret"

    signature = webhook_service.sign_payload(payload, secret)

    # Verify signature is a valid hex string of expected length
    assert len(signature) == 64
    assert all(c in "0123456789abcdef" for c in signature)

    # Verify same payload/secret produces same signature
    assert signature == webhook_service.sign_payload(payload, secret)

    # Verify different payload produces different signature
    assert signature != webhook_service.sign_payload({"event": "other"}, secret)


@pytest.mark.asyncio
async def test_send_test_event_success(webhook_service: WebhookService, mock_db: AsyncMock) -> None:
    """Test successful test event delivery."""
    webhook = Webhook(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        url="https://example.com/webhook",
        secret="secret",
        active=True,
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(status_code=200, text="OK")

        result = await webhook_service.send_test_event(mock_db, webhook)

        assert result["success"] is True
        assert result["status_code"] == 200
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_send_test_event_failure(webhook_service: WebhookService, mock_db: AsyncMock) -> None:
    """Test failed test event delivery."""
    webhook = Webhook(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        url="https://example.com/webhook",
        secret="secret",
        active=True,
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(status_code=500, text="Error")

        result = await webhook_service.send_test_event(mock_db, webhook)

        assert result["success"] is False
        assert result["status_code"] == 500
        assert result["error"] == "Error"


@pytest.mark.asyncio
async def test_trigger_event_no_webhooks(
    webhook_service: WebhookService, mock_db: AsyncMock
) -> None:
    """Test triggering an event when no webhooks are subscribed."""
    project_id = uuid.uuid4()

    # Mock DB response for empty webhooks
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    with patch.object(webhook_service, "_deliver_webhook") as mock_deliver:
        await webhook_service.trigger_event(mock_db, project_id, "evaluation.completed", {})

        mock_deliver.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_event_with_subscription(
    webhook_service: WebhookService, mock_db: AsyncMock
) -> None:
    """Test triggering an event with a subscribed webhook."""
    project_id = uuid.uuid4()
    webhook = Webhook(
        id=uuid.uuid4(),
        project_id=project_id,
        url="https://example.com/webhook",
        secret="secret",
        active=True,
        events=["evaluation.completed"],
    )

    # Mock DB response
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [webhook]
    mock_db.execute.return_value = mock_result

    with patch.object(webhook_service, "_deliver_webhook") as mock_deliver:
        await webhook_service.trigger_event(
            mock_db, project_id, "evaluation.completed", {"eval_id": "123"}
        )

        mock_deliver.assert_called_once()
        # Verify the payload passed to deliver
        call_args = mock_deliver.call_args[0]
        assert call_args[2]["event_type"] == "evaluation.completed"
        assert call_args[2]["data"] == {"eval_id": "123"}
        assert call_args[2]["project_id"] == str(project_id)
