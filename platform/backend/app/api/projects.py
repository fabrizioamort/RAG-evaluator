"""Projects API endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, Pagination, StatusFilter, TagsFilter
from app.models.project import Project
from app.schemas.evaluation import EvaluationResponse
from app.schemas.project import (
    ProjectCreate,
    ProjectList,
    ProjectResponse,
    ProjectUpdate,
)
from app.utils.logging_config import get_logger

router = APIRouter(prefix="/projects", tags=["Projects"])
logger = get_logger(__name__)


def _project_to_response(project: Project) -> ProjectResponse:
    """Convert Project model to ProjectResponse schema."""
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status,
        tags=project.tags if isinstance(project.tags, list) else [],
        created_at=project.created_at,
        updated_at=project.updated_at,
        knowledge_base_count=len(project.knowledge_bases) if project.knowledge_bases else 0,
        test_set_count=len(project.test_sets) if project.test_sets else 0,
        rag_config_count=len(project.rag_configs) if project.rag_configs else 0,
        evaluation_count=len(project.evaluations) if project.evaluations else 0,
    )


@router.get(
    "",
    response_model=ProjectList,
    summary="List all projects",
    description="Retrieve a paginated list of projects with optional filters.",
)
async def list_projects(
    db: DbSession,
    pagination: Pagination,
    status_filter: StatusFilter = None,
    tags: TagsFilter = None,
) -> ProjectList:
    """List all projects with optional filtering and pagination."""
    # Build base query with relationship counts
    query = select(Project).options(
        selectinload(Project.knowledge_bases),
        selectinload(Project.test_sets),
        selectinload(Project.rag_configs),
        selectinload(Project.evaluations),
    )

    # Apply filters
    if status_filter:
        query = query.where(Project.status == status_filter)

    if tags:
        # Filter projects that have any of the specified tags
        # Using PostgreSQL JSONB contains operator or SQLite JSON functions
        for tag in tags:
            query = query.where(Project.tags.contains([tag]))

    # Get total count
    count_query = select(func.count()).select_from(Project)
    if status_filter:
        count_query = count_query.where(Project.status == status_filter)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(Project.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    projects = result.scalars().all()

    logger.info(
        "Listed projects",
        count=len(projects),
        total=total,
        status_filter=status_filter,
    )

    return ProjectList(
        items=[_project_to_response(p) for p in projects],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new project",
    description="Create a new project for organizing RAG evaluations.",
)
async def create_project(
    db: DbSession,
    project_data: ProjectCreate,
) -> ProjectResponse:
    """Create a new project."""
    project = Project(
        name=project_data.name,
        description=project_data.description,
        tags=project_data.tags,
        status="active",
    )

    db.add(project)
    await db.commit()
    await db.refresh(project)

    logger.info("Created project", project_id=str(project.id), name=project.name)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status,
        tags=project.tags if isinstance(project.tags, list) else [],
        created_at=project.created_at,
        updated_at=project.updated_at,
        knowledge_base_count=0,
        test_set_count=0,
        rag_config_count=0,
        evaluation_count=0,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project details",
    description="Retrieve details of a specific project by ID.",
    responses={
        404: {"description": "Project not found"},
    },
)
async def get_project(
    db: DbSession,
    project_id: UUID,
) -> ProjectResponse:
    """Get a specific project by ID."""
    query = (
        select(Project)
        .where(Project.id == project_id)
        .options(
            selectinload(Project.knowledge_bases),
            selectinload(Project.test_sets),
            selectinload(Project.rag_configs),
            selectinload(Project.evaluations),
        )
    )

    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        logger.warning("Project not found", project_id=str(project_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    return _project_to_response(project)


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Update a project",
    description="Update an existing project's details.",
    responses={
        404: {"description": "Project not found"},
    },
)
async def update_project(
    db: DbSession,
    project_id: UUID,
    project_data: ProjectUpdate,
) -> ProjectResponse:
    """Update an existing project."""
    # First check if project exists
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        logger.warning("Project not found for update", project_id=str(project_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    # Update only provided fields
    update_data = project_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    await db.commit()

    # Re-fetch with relationships to avoid lazy loading issues
    query_with_rels = (
        select(Project)
        .where(Project.id == project_id)
        .options(
            selectinload(Project.knowledge_bases),
            selectinload(Project.test_sets),
            selectinload(Project.rag_configs),
            selectinload(Project.evaluations),
        )
    )
    result = await db.execute(query_with_rels)
    project = result.scalar_one()

    logger.info(
        "Updated project", project_id=str(project_id), updated_fields=list(update_data.keys())
    )

    return _project_to_response(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project",
    description="Delete a project and all its related data (knowledge bases, test sets, evaluations, etc.).",
    responses={
        404: {"description": "Project not found"},
    },
)
async def delete_project(
    db: DbSession,
    project_id: UUID,
) -> None:
    """Delete a project and all related data (cascade)."""
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        logger.warning("Project not found for deletion", project_id=str(project_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    await db.delete(project)
    await db.commit()

    logger.info("Deleted project", project_id=str(project_id))


@router.post(
    "/{project_id}/archive",
    response_model=ProjectResponse,
    summary="Archive a project",
    description="Archive a project (sets status to 'archived').",
    responses={
        404: {"description": "Project not found"},
    },
)
async def archive_project(
    db: DbSession,
    project_id: UUID,
) -> ProjectResponse:
    """Archive a project by setting its status to 'archived'."""
    # First check if project exists
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        logger.warning("Project not found for archiving", project_id=str(project_id))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    project.status = "archived"
    await db.commit()

    # Re-fetch with relationships to avoid lazy loading issues
    query_with_rels = (
        select(Project)
        .where(Project.id == project_id)
        .options(
            selectinload(Project.knowledge_bases),
            selectinload(Project.test_sets),
            selectinload(Project.rag_configs),
            selectinload(Project.evaluations),
        )
    )
    result = await db.execute(query_with_rels)
    project = result.scalar_one()

    logger.info("Archived project", project_id=str(project_id))

    return _project_to_response(project)


@router.get(
    "/{project_id}/baseline",
    response_model=EvaluationResponse,
    summary="Get project baseline evaluation",
    description="Retrieve the evaluation marked as baseline for this project.",
)
async def get_project_baseline(
    db: DbSession,
    project_id: UUID,
) -> EvaluationResponse:
    """Get the baseline evaluation for a project."""
    from app.api.evaluations import _evaluation_to_response
    from app.models.evaluation import Evaluation

    query = select(Evaluation).where(
        Evaluation.project_id == project_id,
        Evaluation.is_baseline == True,  # noqa: E712
    )
    result = await db.execute(query)
    evaluation = result.scalar_one_or_none()

    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No baseline evaluation found for project {project_id}",
        )

    return _evaluation_to_response(evaluation)
