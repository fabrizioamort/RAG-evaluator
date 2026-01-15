"""Webhooks API endpoints."""

import secrets
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.deps import DbSession, Pagination
from app.models.webhook import Webhook
from app.schemas.webhook import (
    WebhookCreate,
    WebhookList,
    WebhookResponse,
    WebhookResponseSafe,
    WebhookTestRequest,
    WebhookTestResponse,
    WebhookUpdate,
)
from app.services.webhook_service import get_webhook_service
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Webhooks"])
logger = get_logger(__name__)


@router.get(
    "/projects/{project_id}/webhooks",
    response_model=WebhookList,
    summary="List webhooks for a project",
)
async def list_webhooks(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
) -> WebhookList:
    """List all webhooks for a specific project."""
    query = select(Webhook).where(Webhook.project_id == project_id)

    # Get total count
    count_query = select(func.count()).select_from(Webhook).where(Webhook.project_id == project_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.offset(pagination.offset).limit(pagination.limit)
    result = await db.execute(query)
    webhooks = result.scalars().all()

    return WebhookList(
        items=[WebhookResponseSafe.model_validate(w, from_attributes=True) for w in webhooks],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "/projects/{project_id}/webhooks",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new webhook",
)
async def create_webhook(
    db: DbSession,
    project_id: UUID,
    webhook_data: WebhookCreate,
) -> WebhookResponse:
    """Create a new webhook for a project. Max 3 per project."""
    # Check limit
    count_query = select(func.count()).select_from(Webhook).where(Webhook.project_id == project_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    if total >= 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum of 3 webhooks per project allowed",
        )

    webhook = Webhook(
        project_id=project_id,
        name=webhook_data.name,
        url=webhook_data.url,
        events=webhook_data.events,
        active=webhook_data.active,
        secret=secrets.token_hex(24),
    )

    db.add(webhook)
    await db.commit()
    await db.refresh(webhook)

    logger.info("Created webhook", webhook_id=str(webhook.id), project_id=str(project_id))

    return WebhookResponse.model_validate(webhook, from_attributes=True)


@router.get(
    "/webhooks/{webhook_id}",
    response_model=WebhookResponse,
    summary="Get webhook details",
)
async def get_webhook(
    db: DbSession,
    webhook_id: UUID,
) -> WebhookResponse:
    """Get details of a specific webhook."""
    webhook = await db.get(Webhook, webhook_id)
    if not webhook:
        logger.warning("Webhook not found", webhook_id=str(webhook_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook with id {webhook_id} not found",
        )
    return WebhookResponse.model_validate(webhook, from_attributes=True)


@router.patch(
    "/webhooks/{webhook_id}",
    response_model=WebhookResponse,
    summary="Update a webhook",
)
async def update_webhook(
    db: DbSession,
    webhook_id: UUID,
    webhook_data: WebhookUpdate,
) -> WebhookResponse:
    """Update an existing webhook."""
    webhook = await db.get(Webhook, webhook_id)
    if not webhook:
        logger.warning("Webhook not found for update", webhook_id=str(webhook_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook with id {webhook_id} not found",
        )

    update_data = webhook_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(webhook, field, value)

    await db.commit()
    await db.refresh(webhook)

    logger.info("Updated webhook", webhook_id=str(webhook_id))

    return WebhookResponse.model_validate(webhook, from_attributes=True)


@router.delete(
    "/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a webhook",
)
async def delete_webhook(
    db: DbSession,
    webhook_id: UUID,
) -> None:
    """Delete a webhook."""
    webhook = await db.get(Webhook, webhook_id)
    if not webhook:
        logger.warning("Webhook not found for deletion", webhook_id=str(webhook_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook with id {webhook_id} not found",
        )
    await db.delete(webhook)
    await db.commit()

    logger.info("Deleted webhook", webhook_id=str(webhook_id))


@router.post(
    "/webhooks/{webhook_id}/test",
    response_model=WebhookTestResponse,
    summary="Test a webhook",
)
async def test_webhook(
    db: DbSession,
    webhook_id: UUID,
    test_request: WebhookTestRequest,
) -> WebhookTestResponse:
    """Send a test event to a specific webhook."""
    webhook = await db.get(Webhook, webhook_id)
    if not webhook:
        logger.warning("Webhook not found for test", webhook_id=str(webhook_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook with id {webhook_id} not found",
        )

    service = get_webhook_service()
    result = await service.send_test_event(db, webhook, test_request.event_type)

    logger.info(
        "Performed webhook test",
        webhook_id=str(webhook_id),
        success=result["success"],
        status_code=result["status_code"],
    )

    return WebhookTestResponse.model_validate(result)
