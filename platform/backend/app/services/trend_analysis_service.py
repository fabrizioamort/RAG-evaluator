"""Service for trend analysis."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.evaluation import Evaluation
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.rag_config import RAGConfig
from app.schemas.trend import ProjectTrends, RAGConfigTrend, TrendDataPoint
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class TrendAnalysisService:
    """Service for computing evaluation trends over time."""

    def __init__(self) -> None:
        """Initialize trend analysis service."""
        pass

    async def get_project_trends(self, db: AsyncSession, project_id: uuid.UUID) -> ProjectTrends:
        """Get aggregate trends for all RAG configurations in a project."""
        # Fetch all completed evaluations for the project
        query = (
            select(Evaluation)
            .where(
                Evaluation.project_id == project_id,
                Evaluation.status == "completed",
            )
            .options(
                selectinload(Evaluation.rag_config),
                selectinload(Evaluation.index).selectinload(KnowledgeBaseIndex.rag_config),
            )
            .order_by(Evaluation.completed_at.asc())
        )
        result = await db.execute(query)
        evaluations = result.scalars().all()

        # Group by rag_config_id
        trends_map: dict[uuid.UUID | None, list[TrendDataPoint]] = {}
        config_names: dict[uuid.UUID | None, str | None] = {}

        for eval_model in evaluations:
            config_id = eval_model.rag_config_id
            config = eval_model.rag_config
            if config_id is None and eval_model.index:
                config_id = eval_model.index.rag_config_id
                config = eval_model.index.rag_config or config
            if config_id not in trends_map:
                trends_map[config_id] = []
                config_names[config_id] = config.name if config else "Unknown Config"

            if eval_model.completed_at is None:
                continue

            summary_metrics = eval_model.summary_metrics or {}
            performance_metrics = eval_model.performance_metrics or {}
            cost_metrics = eval_model.cost_metrics or {}
            combined_metrics = {
                **summary_metrics,
                "avg_latency_seconds": float(performance_metrics.get("avg_latency_seconds", 0) or 0),
                "total_cost_usd": float(cost_metrics.get("total_cost_usd", 0) or 0),
            }

            data_point = TrendDataPoint(
                timestamp=eval_model.completed_at,
                evaluation_id=eval_model.id,
                metrics=combined_metrics,
                pass_rate=eval_model.pass_rate,
            )
            trends_map[config_id].append(data_point)

        trends = [
            RAGConfigTrend(
                rag_config_id=config_id,
                rag_config_name=config_names[config_id],
                data_points=data_points,
            )
            for config_id, data_points in trends_map.items()
        ]

        return ProjectTrends(project_id=project_id, trends=trends)

    async def get_rag_config_trends(
        self, db: AsyncSession, rag_config_id: uuid.UUID
    ) -> RAGConfigTrend:
        """Get trends for a specific RAG configuration."""
        # Fetch RAG config to get its name
        rag_config = await db.get(RAGConfig, rag_config_id)
        config_name = rag_config.name if rag_config else "Unknown Config"

        # Fetch all completed evaluations for this config
        query = (
            select(Evaluation)
            .where(
                Evaluation.rag_config_id == rag_config_id,
                Evaluation.status == "completed",
            )
            .order_by(Evaluation.completed_at.asc())
        )
        result = await db.execute(query)
        evaluations = result.scalars().all()

        data_points = []
        for eval_model in evaluations:
            if eval_model.completed_at is None:
                continue
            summary_metrics = eval_model.summary_metrics or {}
            performance_metrics = eval_model.performance_metrics or {}
            cost_metrics = eval_model.cost_metrics or {}
            combined_metrics = {
                **summary_metrics,
                "avg_latency_seconds": float(performance_metrics.get("avg_latency_seconds", 0) or 0),
                "total_cost_usd": float(cost_metrics.get("total_cost_usd", 0) or 0),
            }

            data_points.append(
                TrendDataPoint(
                    timestamp=eval_model.completed_at,
                    evaluation_id=eval_model.id,
                    metrics=combined_metrics,
                    pass_rate=eval_model.pass_rate,
                )
            )

        return RAGConfigTrend(
            rag_config_id=rag_config_id,
            rag_config_name=config_name,
            data_points=data_points,
        )


# Singleton instance
_trend_analysis_service: TrendAnalysisService | None = None


def get_trend_analysis_service() -> TrendAnalysisService:
    """Get or create the trend analysis service instance."""
    global _trend_analysis_service
    if _trend_analysis_service is None:
        _trend_analysis_service = TrendAnalysisService()
    return _trend_analysis_service
