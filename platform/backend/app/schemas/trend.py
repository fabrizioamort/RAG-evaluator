"""Schemas for trend analysis."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class TrendDataPoint(BaseModel):
    """Individual data point in a trend."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    evaluation_id: uuid.UUID
    metrics: dict[str, Any]
    pass_rate: float | None = None


class RAGConfigTrend(BaseModel):
    """Trend data for a specific RAG configuration."""

    model_config = ConfigDict(from_attributes=True)

    rag_config_id: uuid.UUID | None
    rag_config_name: str | None
    data_points: list[TrendDataPoint]


class ProjectTrends(BaseModel):
    """Aggregated trend data for a project."""

    model_config = ConfigDict(from_attributes=True)

    project_id: uuid.UUID
    trends: list[RAGConfigTrend]
