"""Evaluation runner for orchestrating RAG evaluation jobs."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set

# Import DeepEval metrics (we'll need them for individual scoring)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.test_case import TestCase
from app.models.test_set import TestSet
from app.services.artifact_store import ArtifactStore, get_artifact_store
from app.services.job_checkpoint_service import get_checkpoint_service
from app.services.job_event_log import get_job_event_log
from app.services.llm_provider import LLMProviderService
from app.services.rag_adapter import get_rag_adapter_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class SafeDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper that uses our safe LLMProviderService."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.provider = LLMProviderService()

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self, *args: Any, **kwargs: Any) -> DeepEvalBaseLLM:
        return self

    def generate(self, prompt: str, schema: Any = None) -> str:
        """Synchronous generate for DeepEval metrics."""
        # Use a fresh event loop since we are in a thread executor
        loop = asyncio.new_event_loop()
        try:
            # Re-use a_generate logic to avoid duplication and ensure same behavior
            return loop.run_until_complete(self.a_generate(prompt, schema))
        finally:
            loop.close()

    async def a_generate(self, prompt: str, schema: Any = None) -> str:
        """Asynchronous generate for DeepEval metrics."""
        logger.debug(
            "SafeDeepEvalLLM.a_generate starting",
            model=self.model_name,
            has_schema=schema is not None,
        )

        # Build kwargs, pass response_format if prompt suggests JSON or schema is present
        completion_kwargs: dict[str, Any] = {"temperature": 0.0}
        if schema or "json" in prompt.lower():
            completion_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.provider.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **completion_kwargs,
            )
            content = response.content

            # Patch JSON if needed (handle common model deviations)
            if completion_kwargs.get("response_format", {}).get("type") == "json_object":
                try:
                    data = json.loads(content)
                    needs_patch = False

                    # Fix: 'verdicts' -> 'verdict' (common DeepEval metric target)
                    if "verdicts" in data and "verdict" not in data:
                        data["verdict"] = data["verdicts"]
                        needs_patch = True

                    if needs_patch:
                        content = json.dumps(data)
                        logger.debug(
                            "SafeDeepEvalLLM.a_generate: Patched JSON model output",
                            original_keys=list(data.keys()),
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        "SafeDeepEvalLLM.a_generate: Model returned invalid JSON in JSON mode",
                        content_preview=content[:100],
                    )

            logger.debug("SafeDeepEvalLLM.a_generate successful", content_preview=content[:200])
            return content
        except Exception as e:
            logger.error("SafeDeepEvalLLM.a_generate failed", error=str(e))
            raise


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
        self.db_lock = asyncio.Lock()

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
                selectinload(Evaluation.knowledge_base),
                selectinload(Evaluation.test_set).selectinload(TestSet.test_cases),
                selectinload(Evaluation.index).selectinload(KnowledgeBaseIndex.knowledge_base),
                selectinload(Evaluation.index).selectinload(KnowledgeBaseIndex.rag_config),
            )
        )
        self.evaluation = result.scalars().first()
        if not self.evaluation:
            raise ValueError(f"Evaluation {self.evaluation_id} not found")

        if not self.evaluation.test_set:
            raise ValueError(f"Evaluation {self.evaluation_id} has no test set")

        self.test_cases = sorted(self.evaluation.test_set.test_cases, key=lambda x: x.created_at)

    def _initialize_metrics(self, llm_model: str) -> List[Any]:
        """Initialize DeepEval metrics based on evaluation configuration."""
        include_reason = settings.EVAL_INCLUDE_REASON
        if self.evaluation and self.evaluation.metric_config:
            config_value = self.evaluation.metric_config.get("include_reason")
            if config_value is not None:
                include_reason = bool(config_value)
        safe_model = SafeDeepEvalLLM(model_name=llm_model)

        # Get selected metrics from config, fallback to all
        selected_metrics = ["faithfulness", "relevancy", "precision", "recall", "g_eval"]
        if self.evaluation and self.evaluation.metric_config:
            selected_metrics = self.evaluation.metric_config.get("metrics", selected_metrics)

        metrics: List[Any] = []
        if "faithfulness" in selected_metrics:
            metrics.append(
                FaithfulnessMetric(threshold=0.7, model=safe_model, include_reason=include_reason)
            )
        if "relevancy" in selected_metrics or "answer_relevancy" in selected_metrics:
            metrics.append(
                AnswerRelevancyMetric(
                    threshold=0.7, model=safe_model, include_reason=include_reason
                )
            )
        if "precision" in selected_metrics or "contextual_precision" in selected_metrics:
            metrics.append(
                ContextualPrecisionMetric(
                    threshold=0.7, model=safe_model, include_reason=include_reason
                )
            )
        if "recall" in selected_metrics or "contextual_recall" in selected_metrics:
            metrics.append(
                ContextualRecallMetric(
                    threshold=0.7, model=safe_model, include_reason=include_reason
                )
            )
        if "g_eval" in selected_metrics:
            metrics.append(
                GEval(
                    name="Correctness",
                    criteria="""Determine if the actual output is semantically equivalent to the expected output.
                    The core information and facts should align, while minor differences in formatting,
                    punctuation, casing, or stylistic additions (like abbreviations in parentheses)
                    should be ignored. Focus on factual accuracy and semantic completeness.""",
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT,
                    ],
                    evaluation_steps=[
                        "Identify the core facts and entities in the expected output.",
                        "Check if the actual output conveys all these core facts accurately.",
                        "Ignore minor formatting differences like trailing punctuation (e.g., 'McGill University' vs 'McGill University.')",
                        "Ignore stylistic variations or parenthetical additions that don't change meaning (e.g., 'Artificial Intelligence' vs 'Artificial Intelligence (AI)')",
                        "Check for any contradictory information that would make the answer factually incorrect.",
                        "Score 1.0 if the semantic meaning is the same, even if the phrasing differs slightly.",
                    ],
                    threshold=settings.EVAL_G_EVAL_THRESHOLD,
                    model=safe_model,
                )
            )

        return metrics

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

            # Prefer using the index if available (new architecture)
            if self.evaluation.index:
                index = self.evaluation.index
                if index.status != "ready":
                    raise ValueError(f"Index is not ready: {index.status}")
                rag = self.rag_adapter.create_rag_for_index(index)
                documents_path = index.knowledge_base.storage_path
                if not documents_path:
                    raise ValueError("Knowledge base has no storage path")
                await self.rag_adapter.prepare_documents(rag, documents_path)
                llm_model = index.config_snapshot.get("llm_model", "gpt-4o-mini")
            else:
                # Fallback to legacy KB + RAG config approach
                assert self.evaluation.rag_config is not None
                rag = self.rag_adapter.get_or_create_rag(
                    self.evaluation.rag_config,
                    index_path=self.evaluation.knowledge_base.index_path
                    if self.evaluation.knowledge_base
                    else None,
                )
                llm_model = self.evaluation.rag_config.llm_model

            # 3. Initialize metrics
            metrics = self._initialize_metrics(llm_model)

            # 4. Process test cases
            start_index = job.progress_current
            total = len(self.test_cases)
            remaining_test_cases = self.test_cases[start_index:]

            if not remaining_test_cases:
                await self._finalize_evaluation()
                return

            if settings.DEEPEVAL_ASYNC_MODE:
                # Parallel execution
                semaphore = asyncio.Semaphore(settings.DEEPEVAL_MAX_CONCURRENCY)

                async def sem_process(idx: int, tc: TestCase) -> None:
                    async with semaphore:
                        # Check for cancellation/pause before starting
                        if (
                            self.evaluation_id in self._cancelled_evaluations
                            or self.evaluation_id in self._paused_evaluations
                        ):
                            return
                        await self._process_test_case(idx, tc, rag, metrics, total)

                tasks = [
                    sem_process(start_index + i, tc) for i, tc in enumerate(remaining_test_cases)
                ]
                await asyncio.gather(*tasks)
            else:
                # Sequential execution
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
                        await self.event_log.log_event(
                            self.evaluation_id, "paused", {"completed": i}
                        )
                        return

                    await self._process_test_case(i, self.test_cases[i], rag, metrics, total)

            # 5. Calculate final summary and complete
            # In parallel mode, we need to check if we were cancelled/paused
            if self.evaluation_id in self._cancelled_evaluations:
                # Status already handled in loop check/sem_process
                return
            if self.evaluation_id in self._paused_evaluations:
                return

            await self._finalize_evaluation()

        except Exception as e:
            logger.exception("Evaluation runner failed", evaluation_id=str(self.evaluation_id))
            await self.checkpoint_service.fail_job(self.evaluation_id, str(e))
            await self.event_log.log_event(self.evaluation_id, "error", {"error_message": str(e)})

            # Trigger webhook
            try:
                if self.evaluation:
                    from app.services.webhook_service import get_webhook_service

                    await get_webhook_service().trigger_event(
                        self.db,
                        self.evaluation.project_id,
                        "evaluation.failed",
                        {
                            "evaluation_id": str(self.evaluation_id),
                            "status": "failed",
                            "error_message": str(e),
                        },
                    )
            except Exception as webhook_err:
                logger.error("Failed to trigger failure webhook", error=str(webhook_err))

    async def _process_test_case(
        self, i: int, test_case: TestCase, rag: Any, metrics: List[Any], total: int
    ) -> None:
        """Process a single test case."""
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
            scores: Dict[str, Any] = {}
            for metric in metrics:
                await metric.a_measure(llm_test_case)
                class_name = metric.__class__.__name__
                if class_name == "FaithfulnessMetric":
                    name = "faithfulness"
                elif class_name == "AnswerRelevancyMetric":
                    name = "relevancy"
                elif class_name == "ContextualPrecisionMetric":
                    name = "precision"
                elif class_name == "ContextualRecallMetric":
                    name = "recall"
                elif class_name == "GEval":
                    name = "g_eval"
                else:
                    name = class_name.lower().replace("metric", "")

                scores[f"{name}_score"] = metric.score
                scores[f"{name}_reason"] = getattr(metric, "reason", None)

            # Token usage if available
            prompt_tokens = response.get("metadata", {}).get("token_usage", {}).get("prompt_tokens")
            completion_tokens = (
                response.get("metadata", {}).get("token_usage", {}).get("completion_tokens")
            )

            # Calculate cost
            from decimal import Decimal

            from app.services.cost_tracker import get_cost_tracker

            cost_tracker = get_cost_tracker()

            cost_usd = Decimal("0")
            eval_model = self.evaluation
            if eval_model and prompt_tokens is not None and completion_tokens is not None:
                # Determine LLM model from index (preferred) or rag_config (legacy)
                if eval_model.index:
                    llm_model_for_cost = eval_model.index.config_snapshot.get(
                        "llm_model", "gpt-4o-mini"
                    )
                elif eval_model.rag_config:
                    llm_model_for_cost = eval_model.rag_config.llm_model
                else:
                    llm_model_for_cost = "gpt-4o-mini"

                cost_usd = cost_tracker.calculate_cost(
                    llm_model_for_cost, prompt_tokens, completion_tokens
                )
            else:
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

            # CRITICAL: We need a new session or a lock if we want to commit in parallel
            # Since EvaluationRunner has self.db (AsyncSession), and sessions are not thread-safe/task-safe for concurrent commits
            # We use a lock for DB operations.
            async with self.db_lock:
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
            current_completed = await self._get_completed_count()
            await self.checkpoint_service.update_progress(self.evaluation_id, current_completed)

            # Log progress event
            await self.event_log.log_event(
                self.evaluation_id,
                "progress",
                {
                    "completed": current_completed,
                    "total": total,
                    "current_question": test_case.question,
                    "last_result": {
                        "test_case_id": str(test_case.id),
                        "latency": latency,
                        "average_score": sum(
                            v for k, v in scores.items() if "_score" in k and v is not None
                        )
                        / len([v for k, v in scores.items() if "_score" in k and v is not None])
                        if scores
                        else 0,
                    },
                },
            )

            # Save checkpoint every 5 items
            if current_completed % 5 == 0:
                await self.checkpoint_service.save_checkpoint(
                    self.evaluation_id, current_completed, {"last_index": i + 1}
                )

        except Exception as e:
            logger.error(
                "Error processing test case",
                i=i,
                test_case_id=str(test_case.id),
                error=str(e),
            )
            await self.event_log.log_event(
                self.evaluation_id,
                "error",
                {"error_message": f"Test case {i} failed: {str(e)}", "test_case_index": i},
            )

    async def _get_completed_count(self) -> int:
        """Get the number of completed results for this evaluation."""
        from sqlalchemy import func

        from app.models.evaluation_result import EvaluationResult

        result = await self.db.execute(
            select(func.count(EvaluationResult.id)).where(
                EvaluationResult.evaluation_id == self.evaluation_id
            )
        )
        return result.scalar_one()

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
                func.avg(EvaluationResult.g_eval_score).label("g_eval_avg"),
                func.sum(EvaluationResult.prompt_tokens).label("total_prompt"),
                func.sum(EvaluationResult.completion_tokens).label("total_completion"),
                func.sum(EvaluationResult.cost_usd).label("total_cost"),
                func.avg(EvaluationResult.latency_seconds).label("avg_latency"),
                func.min(EvaluationResult.latency_seconds).label("min_latency"),
                func.max(EvaluationResult.latency_seconds).label("max_latency"),
                func.count(EvaluationResult.id).label("total_count"),
            ).where(EvaluationResult.evaluation_id == self.evaluation_id)
        )
        stats = result.one()

        # Calculate pass rate based on selected metrics
        selected_metrics = (
            self.evaluation.metric_config.get("metrics", [])
            if self.evaluation and self.evaluation.metric_config
            else ["faithfulness", "relevancy", "precision", "recall", "g_eval"]
        )

        pass_conditions = [EvaluationResult.evaluation_id == self.evaluation_id]
        if "faithfulness" in selected_metrics:
            pass_conditions.append(EvaluationResult.faithfulness_score >= 0.7)
        if "relevancy" in selected_metrics:
            pass_conditions.append(EvaluationResult.relevancy_score >= 0.7)
        if "precision" in selected_metrics:
            pass_conditions.append(EvaluationResult.precision_score >= 0.7)
        if "recall" in selected_metrics:
            pass_conditions.append(EvaluationResult.recall_score >= 0.7)
        if "g_eval" in selected_metrics:
            pass_conditions.append(EvaluationResult.g_eval_score >= 0.7)

        pass_res = await self.db.execute(
            select(func.count(EvaluationResult.id)).where(*pass_conditions)
        )
        passed_count = pass_res.scalar_one()
        pass_rate = (passed_count / stats.total_count) if stats.total_count > 0 else 0.0

        # Build summary metrics dynamically
        summary_metrics = {}
        overall_scores = []

        if "faithfulness" in selected_metrics:
            summary_metrics["faithfulness_avg"] = float(stats.faithfulness_avg or 0)
            overall_scores.append(summary_metrics["faithfulness_avg"])

        if "relevancy" in selected_metrics:
            summary_metrics["relevancy_avg"] = float(stats.relevancy_avg or 0)
            overall_scores.append(summary_metrics["relevancy_avg"])

        if "precision" in selected_metrics:
            summary_metrics["precision_avg"] = float(stats.precision_avg or 0)
            overall_scores.append(summary_metrics["precision_avg"])

        if "recall" in selected_metrics:
            summary_metrics["recall_avg"] = float(stats.recall_avg or 0)
            overall_scores.append(summary_metrics["recall_avg"])

        if "g_eval" in selected_metrics:
            summary_metrics["g_eval_avg"] = float(stats.g_eval_avg or 0)
            overall_scores.append(summary_metrics["g_eval_avg"])

        # Final overall average
        summary_metrics["overall_avg"] = (
            round(sum(overall_scores) / len(overall_scores), 3) if overall_scores else 0.0
        )

        cost_metrics = {
            "total_cost_usd": float(stats.total_cost or 0),
            "total_prompt_tokens": int(stats.total_prompt or 0),
            "total_completion_tokens": int(stats.total_completion or 0),
            "avg_cost_per_query": float(stats.total_cost or 0) / stats.total_count
            if stats.total_count > 0
            else 0,
        }

        performance_metrics = {
            "avg_latency_seconds": float(stats.avg_latency or 0),
            "min_latency_seconds": float(stats.min_latency or 0),
            "max_latency_seconds": float(stats.max_latency or 0),
            "p95_latency_seconds": 0.0,  # Placeholder for now
        }

        await self.checkpoint_service.complete_job(
            self.evaluation_id,
            summary_metrics=summary_metrics,
            pass_rate=pass_rate,
            cost_metrics=cost_metrics,
            performance_metrics=performance_metrics,
        )

        await self.event_log.log_event(
            self.evaluation_id,
            "completed",
            {
                "summary_metrics": summary_metrics,
                "pass_rate": pass_rate,
                "cost_metrics": cost_metrics,
                "performance_metrics": performance_metrics,
                "duration_seconds": 0.0,  # Placeholder
            },
        )

        # Trigger webhook
        try:
            from app.services.webhook_service import get_webhook_service

            eval_model = self.evaluation
            if eval_model:
                await get_webhook_service().trigger_event(
                    self.db,
                    eval_model.project_id,
                    "evaluation.completed",
                    {
                        "evaluation_id": str(self.evaluation_id),
                        "status": "completed",
                        "pass_rate": pass_rate,
                        "summary_metrics": summary_metrics,
                    },
                )
        except Exception as webhook_err:
            logger.error("Failed to trigger completion webhook", error=str(webhook_err))


def get_evaluation_runner(db_session: AsyncSession, evaluation_id: uuid.UUID) -> EvaluationRunner:
    """Factory to get the evaluation runner."""
    return EvaluationRunner(db_session, evaluation_id)
