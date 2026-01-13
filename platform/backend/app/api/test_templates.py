"""Test Templates API endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.deps import DbSession, Pagination
from app.models.test_template import TestTemplate
from app.schemas.test_template import (
    TestTemplateCreate,
    TestTemplateList,
    TestTemplateResponse,
    TestTemplateUpdate,
)
from app.utils.logging_config import get_logger

router = APIRouter(tags=["Test Templates"])
logger = get_logger(__name__)


def _template_to_response(template: TestTemplate) -> TestTemplateResponse:
    """Convert TestTemplate model to TestTemplateResponse schema."""
    return TestTemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        category=template.category,
        question_template=template.question_template,
        answer_template=template.answer_template,
        entity_types=template.entity_types if isinstance(template.entity_types, list) else [],
        complexity_level=template.complexity_level,
        is_builtin=template.is_builtin,
        created_at=template.created_at,
    )


async def _get_template_or_404(db: DbSession, template_id: UUID) -> TestTemplate:
    """Get test template by ID or raise 404."""
    query = select(TestTemplate).where(TestTemplate.id == template_id)
    result = await db.execute(query)
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test template with id {template_id} not found",
        )
    return template


@router.get(
    "/test-templates",
    response_model=TestTemplateList,
    summary="List all test templates",
    description="Retrieve a paginated list of all test templates (builtin and custom).",
)
async def list_test_templates(
    db: DbSession,
    pagination: Pagination,
) -> TestTemplateList:
    """List all test templates."""
    query = select(TestTemplate)

    count_query = select(func.count()).select_from(TestTemplate)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(TestTemplate.is_builtin.desc(), TestTemplate.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    result = await db.execute(query)
    templates = result.scalars().all()

    logger.info("Listed test templates", count=len(templates), total=total)

    return TestTemplateList(
        items=[_template_to_response(t) for t in templates],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "/test-templates",
    response_model=TestTemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom test template",
    description="Create a new custom test template for generating test cases.",
)
async def create_test_template(
    db: DbSession,
    template_data: TestTemplateCreate,
) -> TestTemplateResponse:
    """Create a new custom test template."""
    template = TestTemplate(
        name=template_data.name,
        description=template_data.description,
        category=template_data.category,
        question_template=template_data.question_template,
        answer_template=template_data.answer_template,
        entity_types=template_data.entity_types,
        complexity_level=template_data.complexity_level,
        is_builtin=False,
    )
    db.add(template)
    await db.commit()
    await db.refresh(template)

    logger.info("Created test template", template_id=str(template.id), name=template.name)

    return _template_to_response(template)


@router.get(
    "/test-templates/{template_id}",
    response_model=TestTemplateResponse,
    summary="Get test template details",
    description="Retrieve details of a specific test template.",
)
async def get_test_template(
    db: DbSession,
    template_id: UUID,
) -> TestTemplateResponse:
    """Get a specific test template by ID."""
    template = await _get_template_or_404(db, template_id)
    return _template_to_response(template)


@router.put(
    "/test-templates/{template_id}",
    response_model=TestTemplateResponse,
    summary="Update a test template",
    description="Update a custom test template. Builtin templates cannot be modified.",
)
async def update_test_template(
    db: DbSession,
    template_id: UUID,
    template_data: TestTemplateUpdate,
) -> TestTemplateResponse:
    """Update an existing test template."""
    template = await _get_template_or_404(db, template_id)

    if template.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Builtin templates cannot be modified",
        )

    update_data = template_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(template, field, value)

    await db.commit()
    await db.refresh(template)

    logger.info(
        "Updated test template",
        template_id=str(template_id),
        updated_fields=list(update_data.keys()),
    )

    return _template_to_response(template)


@router.delete(
    "/test-templates/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a test template",
    description="Delete a custom test template. Builtin templates cannot be deleted.",
)
async def delete_test_template(
    db: DbSession,
    template_id: UUID,
) -> None:
    """Delete a test template."""
    template = await _get_template_or_404(db, template_id)

    if template.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Builtin templates cannot be deleted",
        )

    await db.delete(template)
    await db.commit()

    logger.info("Deleted test template", template_id=str(template_id))
