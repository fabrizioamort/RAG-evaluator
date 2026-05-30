"""Service for computing evaluation comparisons."""

from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.comparison import Comparison
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.schemas.comparison import (
    AggregateMetrics,
    CostMetricsDelta,
    EvaluationComparisonResult,
    MetricDelta,
    PerformanceMetricsDelta,
    PerQuestionDelta,
    SummaryMetricsDelta,
)
from app.schemas.evaluation import CostMetrics, PerformanceMetrics, SummaryMetrics
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ComparisonService:
    """Service for computing and managing evaluation comparisons."""

    def __init__(self) -> None:
        """Initialize comparison service."""
        pass

    def _compute_metric_delta(
        self,
        baseline_value: float | Decimal | int | None,
        compared_value: float | Decimal | int | None,
        higher_is_better: bool = True,
    ) -> MetricDelta:
        """Compute the delta between baseline and compared values."""
        # Convert to float for computation
        baseline = float(baseline_value) if baseline_value is not None else None
        compared = float(compared_value) if compared_value is not None else None

        if baseline is None or compared is None:
            return MetricDelta(
                baseline_value=baseline,
                compared_value=compared,
                absolute_delta=None,
                percentage_delta=None,
                improved=None,
            )

        absolute_delta = compared - baseline
        percentage_delta = (absolute_delta / baseline * 100) if baseline != 0 else None

        # Determine if improved based on whether higher is better
        if higher_is_better:
            improved = compared > baseline
        else:
            improved = compared < baseline

        return MetricDelta(
            baseline_value=baseline,
            compared_value=compared,
            absolute_delta=round(absolute_delta, 6),
            percentage_delta=round(percentage_delta, 2) if percentage_delta else None,
            improved=improved,
        )

    def _compute_summary_delta(
        self,
        baseline: SummaryMetrics | None,
        compared: SummaryMetrics | None,
    ) -> SummaryMetricsDelta:
        """Compute delta for summary metrics."""
        if baseline is None or compared is None:
            return SummaryMetricsDelta()

        return SummaryMetricsDelta(
            faithfulness=self._compute_metric_delta(
                baseline.faithfulness_avg, compared.faithfulness_avg, higher_is_better=True
            ),
            relevancy=self._compute_metric_delta(
                baseline.relevancy_avg, compared.relevancy_avg, higher_is_better=True
            ),
            precision=self._compute_metric_delta(
                baseline.precision_avg, compared.precision_avg, higher_is_better=True
            ),
            recall=self._compute_metric_delta(
                baseline.recall_avg, compared.recall_avg, higher_is_better=True
            ),
            overall=self._compute_metric_delta(
                baseline.overall_avg, compared.overall_avg, higher_is_better=True
            ),
        )

    def _compute_cost_delta(
        self,
        baseline: CostMetrics | None,
        compared: CostMetrics | None,
    ) -> CostMetricsDelta:
        """Compute delta for cost metrics (lower is better for costs)."""
        if baseline is None or compared is None:
            return CostMetricsDelta()

        return CostMetricsDelta(
            total_cost_usd=self._compute_metric_delta(
                baseline.total_cost_usd, compared.total_cost_usd, higher_is_better=False
            ),
            total_prompt_tokens=self._compute_metric_delta(
                baseline.total_prompt_tokens,
                compared.total_prompt_tokens,
                higher_is_better=False,
            ),
            total_completion_tokens=self._compute_metric_delta(
                baseline.total_completion_tokens,
                compared.total_completion_tokens,
                higher_is_better=False,
            ),
            avg_cost_per_query=self._compute_metric_delta(
                baseline.avg_cost_per_query,
                compared.avg_cost_per_query,
                higher_is_better=False,
            ),
        )

    def _compute_performance_delta(
        self,
        baseline: PerformanceMetrics | None,
        compared: PerformanceMetrics | None,
    ) -> PerformanceMetricsDelta:
        """Compute delta for performance metrics (lower latency is better)."""
        if baseline is None or compared is None:
            return PerformanceMetricsDelta()

        return PerformanceMetricsDelta(
            avg_latency=self._compute_metric_delta(
                baseline.avg_latency_seconds,
                compared.avg_latency_seconds,
                higher_is_better=False,
            ),
            min_latency=self._compute_metric_delta(
                baseline.min_latency_seconds,
                compared.min_latency_seconds,
                higher_is_better=False,
            ),
            max_latency=self._compute_metric_delta(
                baseline.max_latency_seconds,
                compared.max_latency_seconds,
                higher_is_better=False,
            ),
            p95_latency=self._compute_metric_delta(
                baseline.p95_latency_seconds,
                compared.p95_latency_seconds,
                higher_is_better=False,
            ),
        )

    def _eval_to_summary_metrics(self, eval_model: Evaluation) -> SummaryMetrics | None:
        """Convert evaluation model to SummaryMetrics schema."""
        if not eval_model.summary_metrics:
            return None
        return SummaryMetrics(**eval_model.summary_metrics)

    def _eval_to_cost_metrics(self, eval_model: Evaluation) -> CostMetrics | None:
        """Convert evaluation model to CostMetrics schema."""
        if not eval_model.cost_metrics:
            return None
        return CostMetrics(**eval_model.cost_metrics)

    def _eval_to_performance_metrics(self, eval_model: Evaluation) -> PerformanceMetrics | None:
        """Convert evaluation model to PerformanceMetrics schema."""
        if not eval_model.performance_metrics:
            return None
        return PerformanceMetrics(**eval_model.performance_metrics)

    async def compute_aggregate_metrics(
        self,
        db: AsyncSession,
        baseline_eval: Evaluation,
        compared_evals: list[Evaluation],
    ) -> AggregateMetrics:
        """Compute aggregate comparison metrics."""
        baseline_summary = self._eval_to_summary_metrics(baseline_eval)
        baseline_cost = self._eval_to_cost_metrics(baseline_eval)
        baseline_performance = self._eval_to_performance_metrics(baseline_eval)
        baseline_rag_config_name = (
            baseline_eval.rag_config.name if baseline_eval.rag_config else None
        )

        comparison_results: list[EvaluationComparisonResult] = []

        for compared_eval in compared_evals:
            compared_summary = self._eval_to_summary_metrics(compared_eval)
            compared_cost = self._eval_to_cost_metrics(compared_eval)
            compared_performance = self._eval_to_performance_metrics(compared_eval)

            # Get RAG config name if available
            rag_config_name = None
            if compared_eval.rag_config:
                rag_config_name = compared_eval.rag_config.name

            comparison_results.append(
                EvaluationComparisonResult(
                    evaluation_id=compared_eval.id,
                    evaluation_name=compared_eval.notes,
                    rag_config_name=rag_config_name,
                    summary_metrics=compared_summary,
                    cost_metrics=compared_cost,
                    performance_metrics=compared_performance,
                    pass_rate=compared_eval.pass_rate,
                    summary_delta=self._compute_summary_delta(baseline_summary, compared_summary),
                    cost_delta=self._compute_cost_delta(baseline_cost, compared_cost),
                    performance_delta=self._compute_performance_delta(
                        baseline_performance, compared_performance
                    ),
                    pass_rate_delta=self._compute_metric_delta(
                        baseline_eval.pass_rate,
                        compared_eval.pass_rate,
                        higher_is_better=True,
                    ),
                )
            )

        return AggregateMetrics(
            baseline_evaluation_id=baseline_eval.id,
            baseline_evaluation_name=baseline_eval.notes,
            baseline_rag_config_name=baseline_rag_config_name,
            baseline_summary=baseline_summary,
            baseline_cost=baseline_cost,
            baseline_performance=baseline_performance,
            baseline_pass_rate=baseline_eval.pass_rate,
            comparison_results=comparison_results,
        )

    async def compute_per_question_deltas(
        self,
        db: AsyncSession,
        baseline_eval_id: UUID,
        compared_eval_ids: list[UUID],
    ) -> list[PerQuestionDelta]:
        """Compute per-question comparison results."""
        all_eval_ids = [baseline_eval_id] + compared_eval_ids

        # Get all results for all evaluations
        query = (
            select(EvaluationResult)
            .where(EvaluationResult.evaluation_id.in_(all_eval_ids))
            .options(selectinload(EvaluationResult.test_case))
        )
        result = await db.execute(query)
        all_results = result.scalars().all()

        # Group results by test_case_id
        results_by_test_case: dict[UUID, dict[UUID, EvaluationResult]] = {}
        for r in all_results:
            if r.test_case_id is None:
                continue
            if r.test_case_id not in results_by_test_case:
                results_by_test_case[r.test_case_id] = {}
            results_by_test_case[r.test_case_id][r.evaluation_id] = r

        per_question_deltas: list[PerQuestionDelta] = []

        for test_case_id, eval_results in results_by_test_case.items():
            baseline_result = eval_results.get(baseline_eval_id)

            # Extract baseline scores
            baseline_scores: dict[str, Any] | None = None
            question: str | None = None

            if baseline_result:
                question = baseline_result.test_case.question if baseline_result.test_case else None
                baseline_scores = {
                    "faithfulness": baseline_result.faithfulness_score,
                    "relevancy": baseline_result.relevancy_score,
                    "precision": baseline_result.precision_score,
                    "recall": baseline_result.recall_score,
                    "g_eval": baseline_result.g_eval_score,
                    "latency_seconds": baseline_result.latency_seconds,
                    "generated_answer": baseline_result.generated_answer,
                }

            # Extract compared results
            compared_results: dict[UUID, dict[str, Any]] = {}
            for eval_id in compared_eval_ids:
                compared_result = eval_results.get(eval_id)
                if compared_result:
                    if question is None and compared_result.test_case:
                        question = compared_result.test_case.question
                    compared_results[eval_id] = {
                        "faithfulness": compared_result.faithfulness_score,
                        "relevancy": compared_result.relevancy_score,
                        "precision": compared_result.precision_score,
                        "recall": compared_result.recall_score,
                        "g_eval": compared_result.g_eval_score,
                        "latency_seconds": compared_result.latency_seconds,
                        "generated_answer": compared_result.generated_answer,
                    }

            per_question_deltas.append(
                PerQuestionDelta(
                    test_case_id=test_case_id,
                    question=question,
                    baseline_result=baseline_scores,
                    compared_results=compared_results,
                )
            )

        return per_question_deltas

    async def create_comparison(
        self,
        db: AsyncSession,
        baseline_eval: Evaluation,
        compared_evals: list[Evaluation],
        name: str | None = None,
        description: str | None = None,
    ) -> Comparison:
        """Create a new comparison and compute metrics."""
        # Compute aggregate metrics
        aggregate_metrics = await self.compute_aggregate_metrics(db, baseline_eval, compared_evals)

        # Compute per-question deltas
        compared_ids = [e.id for e in compared_evals]
        per_question_deltas = await self.compute_per_question_deltas(
            db, baseline_eval.id, compared_ids
        )

        # Create comparison record
        comparison = Comparison(
            project_id=baseline_eval.project_id,
            name=name,
            description=description,
            baseline_evaluation_id=baseline_eval.id,
            compared_evaluation_ids=[str(e.id) for e in compared_evals],
            aggregate_metrics=aggregate_metrics.model_dump(mode="json"),
            per_question_deltas=[d.model_dump(mode="json") for d in per_question_deltas],
        )

        db.add(comparison)
        await db.commit()
        await db.refresh(comparison)

        logger.info(
            "Created comparison",
            comparison_id=str(comparison.id),
            baseline_id=str(baseline_eval.id),
            compared_count=len(compared_evals),
        )

        return comparison


# Singleton instance
_comparison_service: ComparisonService | None = None


def get_comparison_service() -> ComparisonService:
    """Get or create the comparison service instance."""
    global _comparison_service
    if _comparison_service is None:
        _comparison_service = ComparisonService()
    return _comparison_service
