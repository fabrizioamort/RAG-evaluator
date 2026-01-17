"""Webhook Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse

# Webhook event types
WEBHOOK_EVENTS = [
    "evaluation.started",
    "evaluation.completed",
    "evaluation.failed",
    "evaluation.paused",
    "generation.started",
    "generation.completed",
    "generation.failed",
]


class WebhookBase(BaseSchema):
    """Base webhook schema."""

    name: str = Field(min_length=1, max_length=255, description="Webhook name")
    url: str = Field(max_length=500, description="Webhook URL")
    events: list[str] = Field(
        min_length=1,
        description="Events to subscribe to",
    )
    active: bool = Field(default=True, description="Whether webhook is active")


class WebhookCreate(WebhookBase):
    """Schema for creating a webhook."""

    pass


class WebhookUpdate(BaseSchema):
    """Schema for updating a webhook."""

    name: str | None = Field(default=None, min_length=1, max_length=255, description="Webhook name")
    url: str | None = Field(default=None, max_length=500, description="Webhook URL")
    events: list[str] | None = Field(default=None, description="Events to subscribe to")
    active: bool | None = Field(default=None, description="Whether webhook is active")


class WebhookResponse(WebhookBase, BaseResponseSchema):
    """Schema for webhook response."""

    project_id: UUID = Field(description="Parent project ID")
    secret: str = Field(description="Webhook secret for signature verification")
    failure_count: int = Field(default=0, description="Consecutive failure count")
    last_triggered_at: datetime | None = Field(default=None, description="Last trigger time")


class WebhookResponseSafe(BaseSchema):
    """Webhook response without secret."""

    id: UUID
    project_id: UUID
    name: str
    url: str
    events: list[str]
    active: bool
    failure_count: int
    last_triggered_at: datetime | None
    created_at: datetime


class WebhookList(PaginatedResponse):
    """Paginated list of webhooks."""

    items: list[WebhookResponseSafe]


class WebhookTestRequest(BaseSchema):
    """Schema for testing a webhook."""

    event_type: str = Field(
        default="evaluation.completed",
        description="Event type to simulate",
    )


class WebhookTestResponse(BaseSchema):
    """Response from webhook test."""

    success: bool = Field(description="Whether test delivery succeeded")
    status_code: int | None = Field(default=None, description="HTTP status code")
    response_time_ms: float | None = Field(default=None, description="Response time")
    error: str | None = Field(default=None, description="Error message if failed")


class WebhookDelivery(BaseSchema):
    """Schema for a webhook delivery attempt."""

    id: UUID
    webhook_id: UUID
    event_type: str
    payload: dict[str, Any]
    status_code: int | None
    response_body: str | None
    error_message: str | None
    delivered_at: datetime
    success: bool


class WebhookPayload(BaseSchema):
    """Base schema for webhook payloads."""

    event_type: str = Field(description="Event type")
    timestamp: datetime = Field(description="Event timestamp")
    project_id: UUID = Field(description="Project ID")


class EvaluationWebhookPayload(WebhookPayload):
    """Payload for evaluation-related webhooks."""

    evaluation_id: UUID = Field(description="Evaluation ID")
    status: str = Field(description="Evaluation status")
    pass_rate: float | None = Field(default=None, description="Pass rate if completed")
    error_message: str | None = Field(default=None, description="Error if failed")
    summary_metrics: dict[str, float] | None = Field(
        default=None, description="Summary metrics if completed"
    )


class GenerationWebhookPayload(WebhookPayload):
    """Payload for test generation webhooks."""

    test_set_id: UUID = Field(description="Test set ID")
    job_id: UUID = Field(description="Generation job ID")
    status: str = Field(description="Job status")
    generated_count: int = Field(default=0, description="Number of tests generated")
    rejected_count: int = Field(default=0, description="Number of tests rejected")
    error_message: str | None = Field(default=None, description="Error if failed")
