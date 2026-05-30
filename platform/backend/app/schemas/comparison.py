"""Comparison Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse
from app.schemas.evaluation import CostMetrics, PerformanceMetrics, SummaryMetrics


class MetricDelta(BaseSchema):
    """Schema for metric difference between evaluations."""

    baseline_value: float | None = Field(default=None, description="Baseline metric value")
    compared_value: float | None = Field(default=None, description="Compared metric value")
    absolute_delta: float | None = Field(default=None, description="Absolute difference")
    percentage_delta: float | None = Field(
        default=None, description="Percentage change from baseline"
    )
    improved: bool | None = Field(
        default=None, description="Whether the metric improved (higher is better)"
    )


class SummaryMetricsDelta(BaseSchema):
    """Schema for summary metrics comparison."""

    faithfulness: MetricDelta | None = Field(default=None)
    relevancy: MetricDelta | None = Field(default=None)
    precision: MetricDelta | None = Field(default=None)
    recall: MetricDelta | None = Field(default=None)
    overall: MetricDelta | None = Field(default=None)


class CostMetricsDelta(BaseSchema):
    """Schema for cost metrics comparison."""

    total_cost_usd: MetricDelta | None = Field(default=None)
    total_prompt_tokens: MetricDelta | None = Field(default=None)
    total_completion_tokens: MetricDelta | None = Field(default=None)
    avg_cost_per_query: MetricDelta | None = Field(default=None)


class PerformanceMetricsDelta(BaseSchema):
    """Schema for performance metrics comparison."""

    avg_latency: MetricDelta | None = Field(default=None)
    min_latency: MetricDelta | None = Field(default=None)
    max_latency: MetricDelta | None = Field(default=None)
    p95_latency: MetricDelta | None = Field(default=None)


class EvaluationComparisonResult(BaseSchema):
    """Schema for a single evaluation's comparison to baseline."""

    evaluation_id: UUID = Field(description="ID of the compared evaluation")
    evaluation_name: str | None = Field(default=None, description="Name/notes of evaluation")
    rag_config_name: str | None = Field(default=None, description="RAG config name")
    summary_metrics: SummaryMetrics | None = Field(default=None, description="Summary metrics")
    cost_metrics: CostMetrics | None = Field(default=None, description="Cost metrics")
    performance_metrics: PerformanceMetrics | None = Field(
        default=None, description="Performance metrics"
    )
    pass_rate: float | None = Field(default=None, description="Pass rate")
    summary_delta: SummaryMetricsDelta | None = Field(
        default=None, description="Difference from baseline"
    )
    cost_delta: CostMetricsDelta | None = Field(
        default=None, description="Cost difference from baseline"
    )
    performance_delta: PerformanceMetricsDelta | None = Field(
        default=None, description="Performance difference from baseline"
    )
    pass_rate_delta: MetricDelta | None = Field(
        default=None, description="Pass rate difference from baseline"
    )


class PerQuestionDelta(BaseSchema):
    """Schema for per-question comparison between evaluations."""

    test_case_id: UUID = Field(description="Test case ID")
    question: str | None = Field(default=None, description="The question")
    baseline_result: dict[str, Any] | None = Field(
        default=None, description="Baseline result scores"
    )
    compared_results: dict[UUID, dict[str, Any]] = Field(
        default_factory=dict, description="Compared results by evaluation ID"
    )


class AggregateMetrics(BaseSchema):
    """Schema for aggregate comparison metrics."""

    baseline_evaluation_id: UUID = Field(description="Baseline evaluation ID")
    baseline_evaluation_name: str | None = Field(
        default=None, description="Baseline evaluation name/notes"
    )
    baseline_rag_config_name: str | None = Field(
        default=None, description="Baseline RAG config name"
    )
    baseline_summary: SummaryMetrics | None = Field(default=None)
    baseline_cost: CostMetrics | None = Field(default=None)
    baseline_performance: PerformanceMetrics | None = Field(default=None)
    baseline_pass_rate: float | None = Field(default=None)
    comparison_results: list[EvaluationComparisonResult] = Field(
        default_factory=list, description="Comparison results for each evaluation"
    )


class ComparisonBase(BaseSchema):
    """Base comparison schema."""

    name: str | None = Field(default=None, max_length=255, description="Comparison name")
    description: str | None = Field(default=None, description="Comparison description")


class ComparisonCreate(ComparisonBase):
    """Schema for creating a comparison."""

    baseline_evaluation_id: UUID = Field(description="Baseline evaluation ID")
    compared_evaluation_ids: list[UUID] = Field(
        min_length=1,
        max_length=10,
        description="Evaluation IDs to compare against baseline",
    )

    @field_validator("compared_evaluation_ids")
    @classmethod
    def validate_unique_ids(cls, v: list[UUID]) -> list[UUID]:
        """Ensure all evaluation IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Compared evaluation IDs must be unique")
        return v


class ComparisonResponse(ComparisonBase, BaseResponseSchema):
    """Schema for comparison response."""

    project_id: UUID = Field(description="Project ID")
    baseline_evaluation_id: UUID = Field(description="Baseline evaluation ID")
    compared_evaluation_ids: list[UUID] = Field(description="Compared evaluation IDs")
    aggregate_metrics: AggregateMetrics | None = Field(
        default=None, description="Aggregate comparison metrics"
    )

    @field_validator("compared_evaluation_ids", mode="before")
    @classmethod
    def convert_string_uuids(cls, v: list[str] | list[UUID]) -> list[UUID]:
        """Convert string UUIDs to UUID objects."""
        if not v:
            return []
        return [UUID(str(item)) if isinstance(item, str) else item for item in v]


class ComparisonDetail(ComparisonResponse):
    """Schema for detailed comparison with per-question deltas."""

    per_question_deltas: list[PerQuestionDelta] | None = Field(
        default=None, description="Per-question comparison results"
    )


class ComparisonSummary(BaseSchema):
    """Minimal comparison info for lists."""

    id: UUID
    name: str | None
    baseline_evaluation_id: UUID
    compared_count: int = Field(description="Number of evaluations compared")
    created_at: datetime


class ComparisonList(PaginatedResponse):
    """Paginated list of comparisons."""

    items: list[ComparisonResponse]
