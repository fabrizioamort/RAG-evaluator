"""RAG configurations API endpoints."""

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from rag_evaluator.rag_implementations.graph_rag.neo4j_connection import (
    Neo4jConnectionError,
    resolve_neo4j_connection_params,
    test_neo4j_connection,
)
from sqlalchemy import func, select

from app.api.deps import DbSession, Pagination
from app.config import settings
from app.models.project import Project
from app.models.rag_config import RAGConfig
from app.schemas.rag_config import (
    LLMProviderInfo,
    RAGConfigCreate,
    RAGConfigList,
    RAGConfigResponse,
    RAGConfigUpdate,
    RAGTypeInfo,
    RAGTypeParameter,
)
from app.services.rag_registry import RAGRegistry
from app.utils.logging_config import get_logger

# We use two routers: one for project-nested resources and one for direct resource access
router = APIRouter(tags=["RAG Configs"])
logger = get_logger(__name__)


def _test_neo4j_connection(uri: str, username: str, password: str) -> None:
    """Test Neo4j connectivity and authentication. Raises on failure."""
    test_neo4j_connection(uri, username, password)


def _resolve_graph_neo4j_params(params: dict[str, Any] | None) -> tuple[str, str, str]:
    """Resolve graph_rag Neo4j params using backend settings fallback."""
    parameters = params or {}
    return resolve_neo4j_connection_params(
        parameters.get("neo4j_uri"),
        parameters.get("neo4j_username"),
        parameters.get("neo4j_password"),
        default_uri=settings.NEO4J_URI,
        default_username=settings.NEO4J_USERNAME,
        default_password=settings.NEO4J_PASSWORD,
    )


# --- Discovery Endpoints ---


@router.get(
    "/rag-types",
    response_model=list[RAGTypeInfo],
    summary="List available RAG types",
)
async def list_rag_types() -> list[RAGTypeInfo]:
    """Get all supported RAG implementation types and their metadata."""
    return RAGRegistry.get_rag_types()


@router.get(
    "/rag-types/{rag_type}/parameters",
    response_model=list[RAGTypeParameter],
    summary="Get parameters for a RAG type",
    responses={404: {"description": "RAG type not found"}},
)
async def get_rag_type_parameters(rag_type: str) -> list[RAGTypeParameter]:
    """Get the parameter schema for a specific RAG implementation type."""
    rag_types = RAGRegistry.get_rag_types()
    for t in rag_types:
        if t.name == rag_type:
            return t.parameters
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"RAG type '{rag_type}' not found",
    )


@router.get(
    "/llm-providers",
    response_model=list[LLMProviderInfo],
    summary="List available LLM providers",
)
async def list_llm_providers() -> list[LLMProviderInfo]:
    """Get all supported LLM providers and their available models."""
    return RAGRegistry.get_llm_providers()


# --- CRUD Endpoints ---


@router.get(
    "/projects/{project_id}/rag-configs",
    response_model=RAGConfigList,
    summary="List RAG configs for a project",
)
async def list_project_rag_configs(
    db: DbSession,
    project_id: UUID,
    pagination: Pagination,
) -> RAGConfigList:
    """Retrieve a paginated list of RAG configurations for a specific project."""
    # Check if project exists
    project_exists = await db.execute(select(Project.id).where(Project.id == project_id))
    if not project_exists.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    # Build query
    query = select(RAGConfig).where(RAGConfig.project_id == project_id)

    # Get total count
    count_query = (
        select(func.count()).select_from(RAGConfig).where(RAGConfig.project_id == project_id)
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(RAGConfig.created_at.desc())
    query = query.offset(pagination.offset).limit(pagination.limit)

    # Execute query
    result = await db.execute(query)
    configs = result.scalars().all()

    return RAGConfigList(
        items=[RAGConfigResponse.model_validate(c) for c in configs],
        offset=pagination.offset,
        limit=pagination.limit,
        total=total,
    )


@router.post(
    "/projects/{project_id}/rag-configs",
    response_model=RAGConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new RAG config",
)
async def create_rag_config(
    db: DbSession,
    project_id: UUID,
    config_data: RAGConfigCreate,
) -> RAGConfig:
    """Create a new RAG configuration within a project."""
    # Check if project exists
    project_exists = await db.execute(select(Project.id).where(Project.id == project_id))
    if not project_exists.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found",
        )

    # Validate RAG type
    valid_types = [t.name for t in RAGRegistry.get_rag_types()]
    if config_data.rag_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid RAG type '{config_data.rag_type}'. Valid types are: {', '.join(valid_types)}",
        )

    # Verify Neo4j connectivity for graph_rag configs
    if config_data.rag_type == "graph_rag":
        neo4j_uri, neo4j_username, neo4j_password = _resolve_graph_neo4j_params(
            config_data.parameters
        )
        try:
            await asyncio.to_thread(_test_neo4j_connection, neo4j_uri, neo4j_username, neo4j_password)
        except Neo4jConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot connect to Neo4j at '{neo4j_uri}': {exc}",
            ) from exc

    config = RAGConfig(
        project_id=project_id,
        name=config_data.name,
        rag_type=config_data.rag_type,
        parameters=config_data.parameters,
        llm_provider=config_data.llm_provider,
        llm_model=config_data.llm_model,
        llm_base_url=config_data.llm_base_url,
        embedding_model=config_data.embedding_model,
    )

    db.add(config)
    await db.commit()
    await db.refresh(config)

    logger.info("Created RAG config", config_id=str(config.id), project_id=str(project_id))
    return config


@router.get(
    "/rag-configs/{config_id}",
    response_model=RAGConfigResponse,
    summary="Get RAG config details",
)
async def get_rag_config(
    db: DbSession,
    config_id: UUID,
) -> RAGConfig:
    """Retrieve details of a specific RAG configuration by ID."""
    query = select(RAGConfig).where(RAGConfig.id == config_id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG config with id {config_id} not found",
        )

    return config


@router.put(
    "/rag-configs/{config_id}",
    response_model=RAGConfigResponse,
    summary="Update a RAG config",
)
async def update_rag_config(
    db: DbSession,
    config_id: UUID,
    config_data: RAGConfigUpdate,
) -> RAGConfig:
    """Update an existing RAG configuration."""
    query = select(RAGConfig).where(RAGConfig.id == config_id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG config with id {config_id} not found",
        )

    # Update only provided fields
    update_data = config_data.model_dump(exclude_unset=True)

    # Re-validate Neo4j connectivity on graph_rag parameter updates.
    if config.rag_type == "graph_rag" and "parameters" in update_data:
        neo4j_uri, neo4j_username, neo4j_password = _resolve_graph_neo4j_params(
            update_data.get("parameters")
        )
        try:
            await asyncio.to_thread(_test_neo4j_connection, neo4j_uri, neo4j_username, neo4j_password)
        except Neo4jConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot connect to Neo4j at '{neo4j_uri}': {exc}",
            ) from exc

    for field, value in update_data.items():
        setattr(config, field, value)

    await db.commit()
    await db.refresh(config)

    logger.info("Updated RAG config", config_id=str(config_id))
    return config


@router.delete(
    "/rag-configs/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a RAG config",
)
async def delete_rag_config(
    db: DbSession,
    config_id: UUID,
) -> None:
    """Delete a RAG configuration."""
    query = select(RAGConfig).where(RAGConfig.id == config_id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG config with id {config_id} not found",
        )

    await db.delete(config)
    await db.commit()

    logger.info("Deleted RAG config", config_id=str(config_id))
