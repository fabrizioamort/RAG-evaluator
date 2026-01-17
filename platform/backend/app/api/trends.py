"""Trends API router."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.trend import ProjectTrends, RAGConfigTrend
from app.services.trend_analysis_service import TrendAnalysisService, get_trend_analysis_service

router = APIRouter()


@router.get("/projects/{project_id}/trends", response_model=ProjectTrends)
async def get_project_trends(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    trend_service: TrendAnalysisService = Depends(get_trend_analysis_service),
) -> Any:
    """Get aggregate trends for all RAG configurations in a project."""
    return await trend_service.get_project_trends(db, project_id)


@router.get("/rag-configs/{rag_config_id}/trends", response_model=RAGConfigTrend)
async def get_rag_config_trends(
    rag_config_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    trend_service: TrendAnalysisService = Depends(get_trend_analysis_service),
) -> Any:
    """Get trends for a specific RAG configuration."""
    return await trend_service.get_rag_config_trends(db, rag_config_id)
