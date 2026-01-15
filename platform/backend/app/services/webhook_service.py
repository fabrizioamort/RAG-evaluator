"""Webhook service for delivering event notifications."""

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.webhook import Webhook
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class WebhookService:
    """Service for managing and delivering webhooks."""

    def __init__(self) -> None:
        """Initialize the webhook service."""
        self._client = httpx.AsyncClient(timeout=10.0)

    @staticmethod
    def sign_payload(payload: dict[str, Any], secret: str) -> str:
        """Sign a payload with HMAC-SHA256.

        Args:
            payload: The payload to sign.
            secret: The webhook secret.

        Returns:
            The hex-encoded signature.
        """
        message = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
        return signature

    async def trigger_event(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Trigger a webhook event for a project.

        Args:
            db: Database session.
            project_id: Project ID.
            event_type: Type of event (e.g., 'evaluation.completed').
            data: Event-specific data.
        """
        # Find active webhooks for the project subscribed to this event
        stmt = select(Webhook).where(
            Webhook.project_id == project_id,
            Webhook.active.is_(True),
        )
        result = await db.execute(stmt)
        webhooks = result.scalars().all()

        # Filtering in Python as events are stored in a JSON list
        subscribed_webhooks = [w for w in webhooks if event_type in w.events]

        if not subscribed_webhooks:
            logger.debug(
                "No active webhooks subscribed to event",
                project_id=str(project_id),
                event_type=event_type,
            )
            return

        # Prepare payload with metadata
        payload = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": str(project_id),
            "data": data,
        }

        # Deliver to each webhook
        for webhook in subscribed_webhooks:
            await self._deliver_webhook(db, webhook, payload)

    async def _deliver_webhook(
        self, db: AsyncSession, webhook: Webhook, payload: dict[str, Any]
    ) -> bool:
        """Deliver a single webhook payload with optional retries.

        Args:
            db: Database session.
            webhook: Webhook model instance.
            payload: Payload to deliver.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        signature = self.sign_payload(payload, webhook.secret)
        headers = {
            "Content-Type": "application/json",
            "X-RAG-Signature": signature,
            "X-RAG-Event": payload["event_type"],
            "User-Agent": "RAG-Evaluator-Webhook/1.0",
        }

        max_retries = 3
        retry_delay = 1.0  # Initial delay in seconds

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(webhook.url, json=payload, headers=headers)

                if 200 <= response.status_code < 300:
                    webhook.failure_count = 0
                    webhook.last_triggered_at = datetime.now(timezone.utc)
                    await db.commit()
                    logger.debug(
                        "Webhook delivered successfully",
                        webhook_id=str(webhook.id),
                        status_code=response.status_code,
                    )
                    return True

                logger.warning(
                    "Webhook delivery failed",
                    webhook_id=str(webhook.id),
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )

            except Exception as e:
                logger.error(
                    "Error during webhook delivery",
                    webhook_id=str(webhook.id),
                    error=str(e),
                    attempt=attempt + 1,
                )

            if attempt < max_retries:
                import asyncio

                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        # All attempts failed
        webhook.failure_count += 1
        await db.commit()
        return False

    async def send_test_event(
        self, db: AsyncSession, webhook: Webhook, event_type: str = "evaluation.completed"
    ) -> dict[str, Any]:
        """Send a test event to a specific webhook.

        Args:
            db: Database session.
            webhook: Webhook to test.
            event_type: Event type to simulate.

        Returns:
            Dictionary with test results.
        """
        test_payload = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": str(webhook.project_id),
            "data": {
                "message": "This is a test event from RAG Evaluator",
                "test_id": str(uuid.uuid4()),
            },
        }

        start_time = datetime.now(timezone.utc)
        signature = self.sign_payload(test_payload, webhook.secret)
        headers = {
            "Content-Type": "application/json",
            "X-RAG-Signature": signature,
            "X-RAG-Event": event_type,
            "User-Agent": "RAG-Evaluator-Webhook-Test/1.0",
        }

        try:
            response = await self._client.post(
                webhook.url, json=test_payload, headers=headers, timeout=5.0
            )
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return {
                "success": 200 <= response.status_code < 300,
                "status_code": response.status_code,
                "response_time_ms": duration_ms,
                "error": response.text if not (200 <= response.status_code < 300) else None,
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "response_time_ms": None,
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Singleton instance
_webhook_service: WebhookService | None = None


def get_webhook_service() -> WebhookService:
    """Get or create the global WebhookService instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service
