"""Evaluation runner for orchestrating RAG evaluation jobs."""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Set

# Import DeepEval metrics (we'll need them for individual scoring)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.models.test_case import TestCase
from app.models.test_set import TestSet
from app.services.artifact_store import ArtifactStore, get_artifact_store
from app.services.job_checkpoint_service import get_checkpoint_service
from app.services.job_event_log import get_job_event_log
from app.services.rag_adapter import get_rag_adapter_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """Orchestrates the execution of an evaluation job."""

    # Set of currently cancelled/paused evaluation IDs
    _cancelled_evaluations: Set[uuid.UUID] = set()
    _paused_evaluations: Set[uuid.UUID] = set()

    def __init__(self, db_session: AsyncSession, evaluation_id: uuid.UUID) -> None:
        """Initialize the evaluation runner.

        Args:
            db_session: Database session.
            evaluation_id: ID of the evaluation to run.
        """
        self.db = db_session
        self.evaluation_id = evaluation_id
        self.checkpoint_service = get_checkpoint_service(db_session)
        self.event_log = get_job_event_log()
        self.rag_adapter = get_rag_adapter_service()
        self.artifact_store = get_artifact_store()

        # Will be loaded during run()
        self.evaluation: Optional[Evaluation] = None
        self.test_cases: List[TestCase] = []

    @classmethod
    def cancel(cls, evaluation_id: uuid.UUID) -> None:
        """Signal an evaluation to cancel."""
        cls._cancelled_evaluations.add(evaluation_id)

    @classmethod
    def pause(cls, evaluation_id: uuid.UUID) -> None:
        """Signal an evaluation to pause."""
        cls._paused_evaluations.add(evaluation_id)

    @classmethod
    def resume(cls, evaluation_id: uuid.UUID) -> None:
        """Remove pause signal for an evaluation."""
        cls._paused_evaluations.discard(evaluation_id)

    async def _load_context(self) -> None:
        """Load evaluation and test cases from database."""
        result = await self.db.execute(
            select(Evaluation)
            .where(Evaluation.id == self.evaluation_id)
            .options(
                selectinload(Evaluation.rag_config),
                selectinload(Evaluation.test_set).selectinload(TestSet.test_cases),
            )
        )
        self.evaluation = result.scalars().first()
        if not self.evaluation:
            raise ValueError(f"Evaluation {self.evaluation_id} not found")

        if not self.evaluation.test_set:
            raise ValueError(f"Evaluation {self.evaluation_id} has no test set")

        self.test_cases = sorted(self.evaluation.test_set.test_cases, key=lambda x: x.created_at)

    def _initialize_metrics(self, llm_model: str) -> List[Any]:
        """Initialize DeepEval metrics."""
        # Note: We use the same model for all metrics as specified in the evaluation config
        # In a real scenario, we might want to use different models or settings
        include_reason = settings.EVAL_INCLUDE_REASON
        return [
            FaithfulnessMetric(threshold=0.7, model=llm_model, include_reason=include_reason),
            AnswerRelevancyMetric(threshold=0.7, model=llm_model, include_reason=include_reason),
            ContextualPrecisionMetric(
                threshold=0.7, model=llm_model, include_reason=include_reason
            ),
            ContextualRecallMetric(threshold=0.7, model=llm_model, include_reason=include_reason),
        ]

    async def run(self) -> None:
        """Execute the evaluation loop."""
        try:
            await self._load_context()

            # 1. Update status and log start
            await self.checkpoint_service.update_evaluation_status(self.evaluation_id, "running")
            job = await self.checkpoint_service.get_job(self.evaluation_id)
            if not job:
                job = await self.checkpoint_service.create_job(
                    self.evaluation_id, len(self.test_cases)
                )

            await self.checkpoint_service.update_progress(
                self.evaluation_id, job.progress_current, state="running"
            )

            await self.event_log.log_event(
                self.evaluation_id,
                "started",
                {"total_test_cases": len(self.test_cases), "resuming_from": job.progress_current},
            )

            # 2. Get RAG instance
            assert self.evaluation is not None
            assert self.evaluation.rag_config is not None
            rag = self.rag_adapter.get_or_create_rag(self.evaluation.rag_config)

            # 3. Initialize metrics
            metrics = self._initialize_metrics(self.evaluation.rag_config.llm_model)

            # 4. Loop through test cases
            start_index = job.progress_current
            total = len(self.test_cases)

            for i in range(start_index, total):
                # Check for cancellation
                if self.evaluation_id in self._cancelled_evaluations:
                    self._cancelled_evaluations.remove(self.evaluation_id)
                    await self.checkpoint_service.update_evaluation_status(
                        self.evaluation_id, "cancelled"
                    )
                    await self.checkpoint_service.update_progress(
                        self.evaluation_id, i, state="cancelled"
                    )
                    await self.event_log.log_event(
                        self.evaluation_id, "cancelled", {"completed": i}
                    )
                    return

                # Check for pause
                if self.evaluation_id in self._paused_evaluations:
                    await self.checkpoint_service.save_checkpoint(
                        self.evaluation_id, i, {"last_index": i}
                    )
                    await self.checkpoint_service.update_evaluation_status(
                        self.evaluation_id, "paused"
                    )
                    await self.checkpoint_service.update_progress(
                        self.evaluation_id, i, state="paused"
                    )
                    await self.event_log.log_event(self.evaluation_id, "paused", {"completed": i})
                    return

                test_case = self.test_cases[i]
                start_time = time.time()

                try:
                    # RAG Query with full trace
                    response = await self.rag_adapter.query_with_trace(rag, test_case.question)
                    latency = time.time() - start_time

                    # Create DeepEval test case for scoring
                    ground_truth_context: list[str] = (
                        test_case.ground_truth_context
                        if isinstance(test_case.ground_truth_context, list)
                        else []
                    )
                    retrieved_context = response.get("context", [])

                    llm_test_case = LLMTestCase(
                        input=test_case.question,
                        actual_output=response["answer"],
                        expected_output=test_case.expected_answer,
                        context=ground_truth_context,
                        retrieval_context=retrieved_context,
                    )

                    # Score metrics
                    # In a real app, we might want to run these in parallel or use an async evaluator
                    scores: Dict[str, Any] = {}
                    for metric in metrics:
                        # DeepEval metrics are synchronous in their measure() but can be used in threads
                        # Or we can use the async versions if available
                        await asyncio.get_event_loop().run_in_executor(
                            None, metric.measure, llm_test_case
                        )
                        name = metric.__class__.__name__.replace("Metric", "").lower()
                        # Clean up naming
                        if name == "answerrelevancy":
                            name = "relevancy"
                        elif name == "contextualprecision":
                            name = "precision"
                        elif name == "contextualrecall":
                            name = "recall"

                        scores[f"{name}_score"] = metric.score
                        scores[f"{name}_reason"] = getattr(metric, "reason", None)

                    # Token usage if available
                    prompt_tokens = (
                        response.get("metadata", {}).get("token_usage", {}).get("prompt_tokens")
                    )
                    completion_tokens = (
                        response.get("metadata", {}).get("token_usage", {}).get("completion_tokens")
                    )

                    # Calculate cost using CostTracker
                    from decimal import Decimal

                    from app.services.cost_tracker import get_cost_tracker

                    cost_tracker = get_cost_tracker()

                    cost_usd = Decimal("0")
                    if prompt_tokens is not None and completion_tokens is not None:
                        cost_usd = cost_tracker.calculate_cost(
                            self.evaluation.rag_config.llm_model, prompt_tokens, completion_tokens
                        )
                    else:
                        # Fallback to response cost if tokens are missing
                        cost_usd = Decimal(str(response.get("metadata", {}).get("cost", 0.0)))

                    # Store artifacts
                    retrieval_trace = response.get("retrieval_trace")
                    retrieval_trace_artifact = await self.artifact_store.store_json(
                        self.db, retrieval_trace, ArtifactStore.KIND_RETRIEVAL_TRACE
                    )

                    retrieved_context_data = response.get("context", [])
                    retrieved_context_artifact = await self.artifact_store.store_json(
                        self.db, retrieved_context_data, ArtifactStore.KIND_RETRIEVED_CONTEXT
                    )

                    # Collect raw metrics for artifact
                    raw_metrics = {
                        "metric_results": [
                            {
                                "name": metric.__class__.__name__,
                                "score": metric.score,
                                "reason": getattr(metric, "reason", None),
                            }
                            for metric in metrics
                        ]
                    }
                    raw_metrics_artifact = await self.artifact_store.store_json(
                        self.db, raw_metrics, ArtifactStore.KIND_RAW_METRICS
                    )

                    result_model = EvaluationResult(
                        evaluation_id=self.evaluation_id,
                        test_case_id=test_case.id,
                        generated_answer=response["answer"],
                        latency_seconds=latency,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        retrieved_context_artifact_id=retrieved_context_artifact.id,
                        retrieval_trace_artifact_id=retrieval_trace_artifact.id,
                        raw_metrics_artifact_id=raw_metrics_artifact.id,
                        **scores,
                    )
                    self.db.add(result_model)
                    await self.db.commit()

                    # Update progress
                    await self.checkpoint_service.update_progress(self.evaluation_id, i + 1)

                    # Log progress event
                    await self.event_log.log_event(
                        self.evaluation_id,
                        "progress",
                        {
                            "completed": i + 1,
                            "total": total,
                            "current_question": test_case.question,
                            # Minimal result info for SSE
                            "last_result": {
                                "test_case_id": str(test_case.id),
                                "latency": latency,
                                "average_score": sum(
                                    v for k, v in scores.items() if "_score" in k and v is not None
                                )
                                / 4.0,
                            },
                        },
                    )

                    # Save checkpoint every 5 items
                    if (i + 1) % 5 == 0:
                        await self.checkpoint_service.save_checkpoint(
                            self.evaluation_id, i + 1, {"last_index": i + 1}
                        )

                except Exception as e:
                    logger.error(
                        "Error processing test case",
                        i=i,
                        test_case_id=str(test_case.id),
                        error=str(e),
                    )
                    # We might want to continue or fail depending on settings
                    # For now, let's just log it in the event and keep going
                    await self.event_log.log_event(
                        self.evaluation_id,
                        "error",
                        {"error_message": f"Test case {i} failed: {str(e)}", "test_case_index": i},
                    )

            # 5. Calculate final summary and complete
            await self._finalize_evaluation()

        except Exception as e:
            logger.exception("Evaluation runner failed", evaluation_id=str(self.evaluation_id))
            await self.checkpoint_service.fail_job(self.evaluation_id, str(e))
            await self.event_log.log_event(self.evaluation_id, "error", {"error_message": str(e)})

    async def _finalize_evaluation(self) -> None:
        """Calculate aggregate metrics and mark evaluation as complete."""
        # Query all results for this evaluation
        from sqlalchemy import func

        from app.models.evaluation_result import EvaluationResult

        result = await self.db.execute(
            select(
                func.avg(EvaluationResult.faithfulness_score).label("faithfulness_avg"),
                func.avg(EvaluationResult.relevancy_score).label("relevancy_avg"),
                func.avg(EvaluationResult.precision_score).label("precision_avg"),
                func.avg(EvaluationResult.recall_score).label("recall_avg"),
                func.sum(EvaluationResult.prompt_tokens).label("total_prompt"),
                func.sum(EvaluationResult.completion_tokens).label("total_completion"),
                func.sum(EvaluationResult.cost_usd).label("total_cost"),
                func.avg(EvaluationResult.latency_seconds).label("avg_latency"),
                func.count(EvaluationResult.id).label("total_count"),
            ).where(EvaluationResult.evaluation_id == self.evaluation_id)
        )
        stats = result.one()

        # Calculate pass rate (simplified: all metrics > 0.7)
        pass_res = await self.db.execute(
            select(func.count(EvaluationResult.id)).where(
                EvaluationResult.evaluation_id == self.evaluation_id,
                EvaluationResult.faithfulness_score >= 0.7,
                EvaluationResult.relevancy_score >= 0.7,
                EvaluationResult.precision_score
                >= 0.7,  # Wait, why 8.7? Probably a typo in my thought, 0.7.
                EvaluationResult.recall_score >= 0.7,
            )
        )
        passed_count = pass_res.scalar_one()
        pass_rate = (passed_count / stats.total_count) if stats.total_count > 0 else 0.0

        summary_metrics = {
            "faithfulness_avg": float(stats.faithfulness_avg or 0),
            "relevancy_avg": float(stats.relevancy_avg or 0),
            "precision_avg": float(stats.precision_avg or 0),
            "recall_avg": float(stats.recall_avg or 0),
            "overall_avg": (
                (
                    float(stats.faithfulness_avg or 0)
                    + float(stats.relevancy_avg or 0)
                    + float(stats.precision_avg or 0)
                    + float(stats.recall_avg or 0)
                )
                / 4.0
            ),
        }

        await self.checkpoint_service.complete_job(
            self.evaluation_id, summary_metrics=summary_metrics, pass_rate=pass_rate
        )

        await self.event_log.log_event(
            self.evaluation_id,
            "completed",
            {
                "summary_metrics": summary_metrics,
                "pass_rate": pass_rate,
                "duration_seconds": 0.0,  # Placeholder
            },
        )


def get_evaluation_runner(db_session: AsyncSession, evaluation_id: uuid.UUID) -> EvaluationRunner:
    """Factory to get the evaluation runner."""
    return EvaluationRunner(db_session, evaluation_id)
