"""Evaluation and evaluation result Pydantic schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse
from app.schemas.query_overrides import QueryOverrides


class MetricScore(BaseSchema):
    """Schema for a single metric score with explanation."""

    score: float | None = Field(default=None, ge=0, le=1, description="Metric score")
    reason: str | None = Field(default=None, description="Explanation for the score")


class EvaluationResultBase(BaseSchema):
    """Base evaluation result schema."""

    generated_answer: str | None = Field(default=None, description="Generated answer from RAG")
    faithfulness_score: float | None = Field(
        default=None, ge=0, le=1, description="Faithfulness score"
    )
    faithfulness_reason: str | None = Field(default=None, description="Faithfulness explanation")
    relevancy_score: float | None = Field(
        default=None, ge=0, le=1, description="Answer relevancy score"
    )
    relevancy_reason: str | None = Field(default=None, description="Relevancy explanation")
    precision_score: float | None = Field(
        default=None, ge=0, le=1, description="Contextual precision score"
    )
    precision_reason: str | None = Field(default=None, description="Precision explanation")
    recall_score: float | None = Field(
        default=None, ge=0, le=1, description="Contextual recall score"
    )
    recall_reason: str | None = Field(default=None, description="Recall explanation")
    g_eval_score: float | None = Field(default=None, ge=0, le=1, description="G-Eval score")
    g_eval_reason: str | None = Field(default=None, description="G-Eval explanation")
    latency_seconds: float | None = Field(
        default=None, ge=0, description="Query latency in seconds"
    )
    prompt_tokens: int | None = Field(default=None, ge=0, description="Prompt tokens used")
    completion_tokens: int | None = Field(default=None, ge=0, description="Completion tokens used")
    cost_usd: Decimal | None = Field(default=None, ge=0, description="Cost in USD")


class EvaluationResultResponse(EvaluationResultBase, BaseResponseSchema):
    """Schema for evaluation result response."""

    evaluation_id: UUID = Field(description="Parent evaluation ID")
    test_case_id: UUID | None = Field(default=None, description="Source test case ID")
    retrieved_context_artifact_id: UUID | None = Field(
        default=None, description="Artifact with retrieved context"
    )
    retrieval_trace_artifact_id: UUID | None = Field(
        default=None, description="Artifact with retrieval trace"
    )
    raw_metrics_artifact_id: UUID | None = Field(
        default=None, description="Artifact with raw metric data"
    )

    # Computed metrics summary
    @property
    def average_score(self) -> float | None:
        """Calculate average of all metric scores."""
        scores = [
            self.faithfulness_score,
            self.relevancy_score,
            self.precision_score,
            self.recall_score,
            self.g_eval_score,
        ]
        valid_scores = [s for s in scores if s is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None


class EvaluationResultWithTestCase(EvaluationResultResponse):
    """Evaluation result with embedded test case info."""

    question: str | None = Field(default=None, description="Test question")
    expected_answer: str | None = Field(default=None, description="Expected answer")
    difficulty: str | None = Field(default=None, description="Question difficulty")
    category: str | None = Field(default=None, description="Question category")


class SummaryMetrics(BaseSchema):
    """Aggregate metrics for an evaluation."""

    faithfulness_avg: float | None = Field(default=None, description="Average faithfulness")
    relevancy_avg: float | None = Field(default=None, description="Average relevancy")
    precision_avg: float | None = Field(default=None, description="Average precision")
    recall_avg: float | None = Field(default=None, description="Average recall")
    g_eval_avg: float | None = Field(default=None, description="Average G-Eval")
    overall_avg: float | None = Field(default=None, description="Overall average")
    legal_rag_bench: dict[str, Any] | None = Field(
        default=None,
        description="Legal RAG Bench retrieval/judge/taxonomy summary",
    )


class CostMetrics(BaseSchema):
    """Cost metrics for an evaluation."""

    total_cost_usd: Decimal = Field(default=Decimal("0"), description="Total cost")
    total_prompt_tokens: int = Field(default=0, description="Total prompt tokens")
    total_completion_tokens: int = Field(default=0, description="Total completion tokens")
    avg_cost_per_query: Decimal | None = Field(default=None, description="Average cost per query")


class PerformanceMetrics(BaseSchema):
    """Performance metrics for an evaluation."""

    avg_latency_seconds: float | None = Field(default=None, description="Average latency")
    min_latency_seconds: float | None = Field(default=None, description="Min latency")
    max_latency_seconds: float | None = Field(default=None, description="Max latency")
    p95_latency_seconds: float | None = Field(default=None, description="95th percentile latency")


class EvaluationBase(BaseSchema):
    """Base evaluation schema."""

    name: str | None = Field(default=None, description="Human-readable name for the evaluation")
    notes: str | None = Field(default=None, description="Evaluation notes")
    tags: list[str] = Field(default_factory=list, description="Tags")


class EvaluationCreate(EvaluationBase):
    """Schema for creating an evaluation."""

    test_set_id: UUID = Field(description="Test set to use")
    knowledge_base_index_id: UUID = Field(description="Knowledge Base Index to evaluate against")
    metric_names: list[str] | None = Field(
        default=None, description="List of metrics to calculate (default: all)"
    )
    include_reason: bool | None = Field(
        default=None,
        description="Whether to include metric reasoning (overrides global setting)",
    )
    query_overrides: QueryOverrides | None = Field(
        default=None,
        description="Query-time overrides for the selected ready index",
    )
    eval_judge_model: str | None = Field(
        default=None,
        max_length=100,
        description="DeepEval judge model; defaults to the effective RAG generation model",
    )
    eval_judge_provider: str | None = Field(
        default=None,
        max_length=50,
        description="DeepEval judge provider; defaults to the generation provider",
    )
    # knowledge_base_id and rag_config_id are removed as they are derived from the index


class EvaluationResponse(EvaluationBase, BaseResponseSchema):
    """Schema for evaluation response."""

    project_id: UUID = Field(description="Parent project ID")
    knowledge_base_id: UUID | None = Field(default=None, description="Knowledge base ID")
    knowledge_base_index_id: UUID | None = Field(
        default=None, description="Knowledge base Index ID"
    )
    kb_version_id: UUID | None = Field(default=None, description="KB version ID")
    test_set_id: UUID | None = Field(default=None, description="Test set ID")
    # rag_config_id is deprecated/removed, but we might keep it for legacy if needed.
    # We'll remove it to align with new plan.
    run_manifest_id: UUID | None = Field(default=None, description="Run manifest ID")

    status: str = Field(description="Evaluation status")
    started_at: datetime | None = Field(default=None, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")

    summary_metrics: SummaryMetrics | None = Field(default=None, description="Summary metrics")
    cost_metrics: CostMetrics | None = Field(default=None, description="Cost metrics")
    performance_metrics: PerformanceMetrics | None = Field(
        default=None, description="Performance metrics"
    )

    pass_rate: float | None = Field(default=None, ge=0, le=1, description="Pass rate (0-1)")
    is_baseline: bool = Field(default=False, description="Whether this is the baseline")
    baseline_reason: str | None = Field(default=None, description="Reason for baseline")
    metric_config: dict[str, Any] | None = Field(default=None, description="Selected metric config")
    query_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Query-time overrides used by this evaluation",
    )
    eval_judge_model: str | None = Field(default=None, description="DeepEval judge model")
    eval_judge_provider: str | None = Field(default=None, description="DeepEval judge provider")
    error_message: str | None = Field(default=None, description="Error if failed")

    # Counts
    result_count: int = Field(default=0, description="Number of results")


class EvaluationWithDetails(EvaluationResponse):
    """Evaluation with related entity names."""

    knowledge_base_name: str | None = Field(default=None, description="KB name")
    index_name: str | None = Field(default=None, description="Index name")
    test_set_name: str | None = Field(default=None, description="Test set name")
    rag_config_name: str | None = Field(default=None, description="RAG config name (from index)")


class EvaluationSummary(BaseSchema):
    """Minimal evaluation info for lists."""

    id: UUID
    status: str
    pass_rate: float | None
    created_at: datetime
    completed_at: datetime | None


class EvaluationList(PaginatedResponse):
    """Paginated list of evaluations."""

    items: list[EvaluationResponse]


class EvaluationResultList(PaginatedResponse):
    """Paginated list of evaluation results."""

    items: Sequence[EvaluationResultWithTestCase]


# SSE Progress Events


class ProgressEvent(BaseSchema):
    """Base schema for SSE progress events."""

    event_type: str = Field(description="Event type identifier")
    evaluation_id: UUID = Field(description="Evaluation ID")
    timestamp: datetime = Field(description="Event timestamp")


class ProgressStartedEvent(ProgressEvent):
    """Event when evaluation starts."""

    event_type: str = Field(default="started")
    total_test_cases: int = Field(description="Total test cases to evaluate")


class ProgressUpdateEvent(ProgressEvent):
    """Event for progress updates."""

    event_type: str = Field(default="progress")
    completed: int = Field(description="Completed test cases")
    total: int = Field(description="Total test cases")
    current_question: str | None = Field(
        default=None, description="Current question being evaluated"
    )
    last_result: EvaluationResultResponse | None = Field(default=None, description="Last result")


class ProgressCompletedEvent(ProgressEvent):
    """Event when evaluation completes."""

    event_type: str = Field(default="completed")
    summary_metrics: SummaryMetrics = Field(description="Final summary metrics")
    pass_rate: float = Field(description="Final pass rate")
    duration_seconds: float = Field(description="Total duration")


class ProgressErrorEvent(ProgressEvent):
    """Event when evaluation fails."""

    event_type: str = Field(default="error")
    error_message: str = Field(description="Error message")
    test_case_index: int | None = Field(default=None, description="Test case that failed")


class ProgressPausedEvent(ProgressEvent):
    """Event when evaluation is paused."""

    event_type: str = Field(default="paused")
    completed: int = Field(description="Completed test cases so far")


class ProgressResumedEvent(ProgressEvent):
    """Event when evaluation resumes."""

    event_type: str = Field(default="resumed")
    resuming_from: int = Field(description="Test case index resuming from")


# Control schemas


class EvaluationControl(BaseSchema):
    """Schema for evaluation control actions."""

    action: str = Field(
        pattern="^(pause|resume|cancel|retry)$",
        description="Control action",
    )
    reason: str | None = Field(default=None, description="Reason for action")


class SetBaselineRequest(BaseSchema):
    """Schema for setting an evaluation as baseline."""

    reason: str = Field(min_length=1, description="Reason for marking as baseline")


class RunManifestResponse(BaseSchema):
    """Schema for run manifest response."""

    id: UUID
    rag_config_snapshot: dict[str, Any]
    build_config_snapshot: dict[str, Any]
    query_overrides: dict[str, Any]
    effective_config_snapshot: dict[str, Any]
    kb_version_snapshot: dict[str, Any]
    generation_model: str | None
    eval_judge_model: str | None
    prompt_templates: dict[str, Any]
    rag_evaluator_version: str | None
    platform_version: str | None
    created_at: datetime
