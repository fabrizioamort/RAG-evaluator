"""Evaluation runner for orchestrating RAG evaluation jobs."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set

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
from app.services.legal_rag_bench_judge import LegalRAGBenchJudge
from app.services.legal_rag_bench_metrics import (
    compute_legal_rag_retrieval_metrics,
    derive_success_signals,
    derive_taxonomy,
    extract_relevant_passage_id,
    is_legal_rag_judge_enabled,
    is_legal_rag_metric_enabled,
    summarize_legal_rag_metrics,
)
from app.services.llm_provider import LLMProviderService
from app.services.provider_resolver import resolve_provider_endpoint
from app.services.rag_adapter import get_rag_adapter_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

AnswerRelevancyMetric: Any | None = None
ContextualPrecisionMetric: Any | None = None
ContextualRecallMetric: Any | None = None
FaithfulnessMetric: Any | None = None
GEval: Any | None = None
LLMTestCase: Any | None = None
LLMTestCaseParams: Any | None = None
_DEEPEVAL_REAL_LOADED = False
_SAFE_DEEPEVAL_LLM_CLASS: Any | None = None


class _LocalLLMTestCase:
    """Minimal DeepEval-compatible test case used when metrics are test-patched."""

    def __init__(
        self,
        *,
        input: str,
        actual_output: str,
        expected_output: str,
        context: list[str],
        retrieval_context: list[str],
    ) -> None:
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.context = context
        self.retrieval_context = retrieval_context


class _LocalLLMTestCaseParams:
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"


def _ensure_deepeval_loaded() -> None:
    """Import DeepEval symbols on demand to keep pytest collection lightweight."""
    global AnswerRelevancyMetric
    global ContextualPrecisionMetric
    global ContextualRecallMetric
    global FaithfulnessMetric
    global GEval
    global LLMTestCase
    global LLMTestCaseParams
    global _DEEPEVAL_REAL_LOADED

    if all(
        symbol is not None
        for symbol in (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            GEval,
            LLMTestCase,
            LLMTestCaseParams,
        )
    ):
        return

    metrics_are_patched = all(
        symbol is not None
        for symbol in (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            GEval,
        )
    )
    if metrics_are_patched:
        if LLMTestCase is None:
            LLMTestCase = _LocalLLMTestCase
        if LLMTestCaseParams is None:
            LLMTestCaseParams = _LocalLLMTestCaseParams
        return

    from deepeval.metrics import (
        AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
    )
    from deepeval.metrics import (
        ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
    )
    from deepeval.metrics import (
        ContextualRecallMetric as DeepEvalContextualRecallMetric,
    )
    from deepeval.metrics import FaithfulnessMetric as DeepEvalFaithfulnessMetric
    from deepeval.metrics import GEval as DeepEvalGEval
    from deepeval.test_case import LLMTestCase as DeepEvalLLMTestCase
    from deepeval.test_case import LLMTestCaseParams as DeepEvalLLMTestCaseParams

    if AnswerRelevancyMetric is None:
        AnswerRelevancyMetric = DeepEvalAnswerRelevancyMetric
    if ContextualPrecisionMetric is None:
        ContextualPrecisionMetric = DeepEvalContextualPrecisionMetric
    if ContextualRecallMetric is None:
        ContextualRecallMetric = DeepEvalContextualRecallMetric
    if FaithfulnessMetric is None:
        FaithfulnessMetric = DeepEvalFaithfulnessMetric
    if GEval is None:
        GEval = DeepEvalGEval
    if LLMTestCase is None:
        LLMTestCase = DeepEvalLLMTestCase
    if LLMTestCaseParams is None:
        LLMTestCaseParams = DeepEvalLLMTestCaseParams
    _DEEPEVAL_REAL_LOADED = True


class _SafeDeepEvalLLMCore:
    """DeepEval LLM wrapper that uses our safe LLMProviderService."""

    def __init__(
        self,
        model_name: str,
        provider_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.base_url = base_url
        self.api_key = api_key
        self.provider = LLMProviderService()

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
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
                provider=self.provider_name,
                base_url=self.base_url,
                api_key=self.api_key,
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


def _safe_deepeval_llm_class() -> Any:
    """Return a DeepEvalBaseLLM subclass without importing DeepEval at module load."""
    global _SAFE_DEEPEVAL_LLM_CLASS
    if _SAFE_DEEVAL_CLASS := _SAFE_DEEPEVAL_LLM_CLASS:
        return _SAFE_DEEVAL_CLASS

    from deepeval.models.base_model import DeepEvalBaseLLM

    class _SafeDeepEvalLLM(_SafeDeepEvalLLMCore, DeepEvalBaseLLM):
        pass

    _SAFE_DEEPEVAL_LLM_CLASS = _SafeDeepEvalLLM
    return _SafeDeepEvalLLM


def SafeDeepEvalLLM(*args: Any, **kwargs: Any) -> Any:
    """Construct the model wrapper, subclassing DeepEval's base only when needed."""
    if _DEEPEVAL_REAL_LOADED:
        return _safe_deepeval_llm_class()(*args, **kwargs)
    return _SafeDeepEvalLLMCore(*args, **kwargs)


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
        self.legal_rag_judge = LegalRAGBenchJudge()
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

    def _initialize_metrics(
        self,
        llm_model: str,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> List[Any]:
        """Initialize DeepEval metrics based on evaluation configuration."""
        include_reason = settings.EVAL_INCLUDE_REASON
        if self.evaluation and self.evaluation.metric_config:
            config_value = self.evaluation.metric_config.get("include_reason")
            if config_value is not None:
                include_reason = bool(config_value)

        # Get selected metrics from config, fallback to all
        selected_metrics = ["faithfulness", "relevancy", "precision", "recall", "g_eval"]
        if self.evaluation and self.evaluation.metric_config:
            selected_metrics = self.evaluation.metric_config.get("metrics", selected_metrics)

        uses_deepeval = bool(
            {
                "faithfulness",
                "relevancy",
                "answer_relevancy",
                "precision",
                "contextual_precision",
                "recall",
                "contextual_recall",
                "g_eval",
            }
            & set(selected_metrics)
        )
        if not uses_deepeval:
            return []

        _ensure_deepeval_loaded()
        assert FaithfulnessMetric is not None
        assert AnswerRelevancyMetric is not None
        assert ContextualPrecisionMetric is not None
        assert ContextualRecallMetric is not None
        assert GEval is not None
        assert LLMTestCaseParams is not None

        safe_model = SafeDeepEvalLLM(
            model_name=llm_model,
            provider_name=provider,
            base_url=base_url,
            api_key=api_key,
        )

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

            completed_test_case_ids = await self._get_completed_test_case_ids()
            current_completed = len(completed_test_case_ids)

            await self.checkpoint_service.update_progress(
                self.evaluation_id, current_completed, state="running"
            )

            await self.event_log.log_event(
                self.evaluation_id,
                "started",
                {
                    "total_test_cases": len(self.test_cases),
                    "completed": current_completed,
                    "resuming_from": current_completed,
                },
            )

            # 2. Get RAG instance
            assert self.evaluation is not None

            # Prefer using the index if available (new architecture)
            top_k = 5
            if self.evaluation.index:
                index = self.evaluation.index
                if index.status != "ready":
                    raise ValueError(f"Index is not ready: {index.status}")
                rag, effective = self.rag_adapter.load_rag_for_index_query(
                    index, self.evaluation.query_overrides
                )
                top_k = effective.top_k
                llm_model = effective.generation_model
                gen_provider = effective.effective_config_snapshot.get("llm_provider", "openai")
                gen_base_url = effective.effective_config_snapshot.get("llm_base_url")
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
                gen_provider = self.evaluation.rag_config.llm_provider
                gen_base_url = self.evaluation.rag_config.llm_base_url

            # 3. Resolve judge bucket (own provider/model).
            # Resolve explicit credentials so litellm uses its well-tested
            # credential path (env-based pickup hits a broken OpenRouter path).
            # The judge follows the generation endpoint when it shares its provider.
            judge_model = self.evaluation.eval_judge_model or llm_model
            judge_provider = self.evaluation.eval_judge_provider or gen_provider
            base_override = gen_base_url if judge_provider == gen_provider else None
            endpoint = resolve_provider_endpoint(judge_provider, base_override)

            # 4. Process test cases
            total = len(self.test_cases)
            remaining_test_cases = [
                (i, tc)
                for i, tc in enumerate(self.test_cases)
                if tc.id not in completed_test_case_ids
            ]
            failed_case_indices: list[int] = []
            failed_case_errors: list[str] = []

            if not remaining_test_cases:
                await self._finalize_evaluation()
                return

            if settings.DEEPEVAL_ASYNC_MODE:
                # Parallel execution
                semaphore = asyncio.Semaphore(settings.DEEPEVAL_MAX_CONCURRENCY)

                async def sem_process(
                    idx: int, tc: TestCase
                ) -> tuple[int, bool, str | None] | None:
                    async with semaphore:
                        # Check for cancellation/pause before starting
                        if (
                            self.evaluation_id in self._cancelled_evaluations
                            or self.evaluation_id in self._paused_evaluations
                        ):
                            return None
                        success, error_message = await self._process_test_case(
                            idx,
                            tc,
                            rag,
                            total,
                            top_k,
                            llm_model,
                            judge_model,
                            judge_provider,
                            endpoint.base_url,
                            endpoint.api_key,
                        )
                        return idx, success, error_message

                tasks = [sem_process(i, tc) for i, tc in remaining_test_cases]
                results = await asyncio.gather(*tasks)
                for result in results:
                    if result is None:
                        continue
                    idx, success, error_message = result
                    if not success:
                        failed_case_indices.append(idx)
                        if error_message:
                            failed_case_errors.append(error_message)
            else:
                # Sequential execution
                for i, test_case in remaining_test_cases:
                    # Check for cancellation
                    if self.evaluation_id in self._cancelled_evaluations:
                        self._cancelled_evaluations.remove(self.evaluation_id)
                        current_completed = await self._get_completed_count()
                        await self.checkpoint_service.update_evaluation_status(
                            self.evaluation_id, "cancelled"
                        )
                        await self.checkpoint_service.update_progress(
                            self.evaluation_id, current_completed, state="cancelled"
                        )
                        await self.event_log.log_event(
                            self.evaluation_id, "cancelled", {"completed": current_completed}
                        )
                        return

                    # Check for pause
                    if self.evaluation_id in self._paused_evaluations:
                        current_completed = await self._get_completed_count()
                        await self.checkpoint_service.save_checkpoint(
                            self.evaluation_id,
                            current_completed,
                            {"last_index": i, "completed": current_completed},
                        )
                        await self.checkpoint_service.update_evaluation_status(
                            self.evaluation_id, "paused"
                        )
                        await self.checkpoint_service.update_progress(
                            self.evaluation_id, current_completed, state="paused"
                        )
                        await self.event_log.log_event(
                            self.evaluation_id, "paused", {"completed": current_completed}
                        )
                        return

                    success, error_message = await self._process_test_case(
                        i,
                        test_case,
                        rag,
                        total,
                        top_k,
                        llm_model,
                        judge_model,
                        judge_provider,
                        endpoint.base_url,
                        endpoint.api_key,
                    )
                    if not success:
                        failed_case_indices.append(i)
                        if error_message:
                            failed_case_errors.append(error_message)

            # 5. Calculate final summary and complete
            # In parallel mode, we need to check if we were cancelled/paused
            if self.evaluation_id in self._cancelled_evaluations:
                self._cancelled_evaluations.remove(self.evaluation_id)
                current_completed = await self._get_completed_count()
                await self.checkpoint_service.update_evaluation_status(
                    self.evaluation_id, "cancelled"
                )
                await self.checkpoint_service.update_progress(
                    self.evaluation_id, current_completed, state="cancelled"
                )
                await self.event_log.log_event(
                    self.evaluation_id, "cancelled", {"completed": current_completed}
                )
                return
            if self.evaluation_id in self._paused_evaluations:
                current_completed = await self._get_completed_count()
                await self.checkpoint_service.update_evaluation_status(
                    self.evaluation_id, "paused"
                )
                await self.checkpoint_service.update_progress(
                    self.evaluation_id, current_completed, state="paused"
                )
                await self.event_log.log_event(
                    self.evaluation_id, "paused", {"completed": current_completed}
                )
                return

            current_completed = await self._get_completed_count()
            if current_completed < total:
                failed_display = ", ".join(str(idx) for idx in failed_case_indices)
                failed_suffix = (
                    f" Failed test case indexes: {failed_display}." if failed_display else ""
                )
                failure_details = (
                    f" First failure: {failed_case_errors[0]}." if failed_case_errors else ""
                )
                error_message = (
                    f"Evaluation finished with {current_completed}/{total} test cases saved."
                    f"{failed_suffix}{failure_details} Retry will run only the missing test cases."
                )
                aggregate_metrics = (
                    await self._calculate_aggregate_metrics() if current_completed > 0 else None
                )
                await self._fail_evaluation(error_message, aggregate_metrics)
                return

            await self._finalize_evaluation()

        except Exception as e:
            logger.exception("Evaluation runner failed", evaluation_id=str(self.evaluation_id))
            await self._fail_evaluation(str(e))

    async def _process_test_case(
        self,
        i: int,
        test_case: TestCase,
        rag: Any,
        total: int,
        top_k: int,
        generation_model: str,
        judge_model: str,
        judge_provider: str | None,
        judge_base_url: str | None,
        judge_api_key: str | None,
    ) -> tuple[bool, str | None]:
        """Process a single test case."""
        start_time = time.time()
        try:
            async with self.db_lock:
                if await self._has_result_for_test_case(test_case.id):
                    return True, None

            # RAG Query with full trace
            response = await self.rag_adapter.query_with_trace(rag, test_case.question, top_k)
            latency = time.time() - start_time

            # Create DeepEval test case for scoring
            ground_truth_context: list[str] = (
                test_case.ground_truth_context
                if isinstance(test_case.ground_truth_context, list)
                else []
            )
            retrieved_context = response.get("context", [])
            metric_config = self.evaluation.metric_config if self.evaluation else None
            legal_rag_result: dict[str, Any] | None = None
            if is_legal_rag_metric_enabled(metric_config):
                rag_type = "unknown"
                if self.evaluation and self.evaluation.index:
                    rag_type = self.evaluation.index.config_snapshot.get("rag_type", "unknown")
                elif self.evaluation and self.evaluation.rag_config:
                    rag_type = self.evaluation.rag_config.rag_type

                legal_retrieval = compute_legal_rag_retrieval_metrics(
                    rag_type=rag_type,
                    top_k=top_k,
                    relevant_passage_id=extract_relevant_passage_id(
                        getattr(test_case, "metadata_", None),
                        ground_truth_context=ground_truth_context,
                    ),
                    response=response,
                )
                legal_judge = None
                if is_legal_rag_judge_enabled(metric_config):
                    legal_judge = await self.legal_rag_judge.judge(
                        question=test_case.question,
                        reference_answer=test_case.expected_answer,
                        generated_answer=response["answer"],
                        retrieved_context=[str(item) for item in retrieved_context],
                        model=judge_model,
                        provider=judge_provider,
                        base_url=judge_base_url,
                        api_key=judge_api_key,
                    )

                legal_rag_result = {
                    "retrieval": legal_retrieval,
                    "judge": legal_judge,
                    "taxonomy": derive_taxonomy(
                        retrieval_metrics=legal_retrieval,
                        judge_result=legal_judge,
                    ),
                }

            # Score metrics. DeepEval metric objects are mutable: a_measure()
            # writes score/reason onto the object. Create a fresh set per test
            # case so async evaluation cannot mix scores or reasons across
            # concurrent cases.
            metrics = self._initialize_metrics(
                judge_model, judge_provider, judge_base_url, judge_api_key
            )
            scores: Dict[str, Any] = {}
            metric_results: list[dict[str, Any]] = []
            if metrics:
                llm_test_case_cls = LLMTestCase or _LocalLLMTestCase
                llm_test_case = llm_test_case_cls(
                    input=test_case.question,
                    actual_output=response["answer"],
                    expected_output=test_case.expected_answer,
                    context=ground_truth_context,
                    retrieval_context=retrieved_context,
                )
                for metric in metrics:
                    await metric.a_measure(llm_test_case)
                    class_name = metric.__class__.__name__
                    name = self._metric_result_field_name(class_name)
                    score = getattr(metric, "score", None)
                    reason = getattr(metric, "reason", None)

                    scores[f"{name}_score"] = score
                    scores[f"{name}_reason"] = reason
                    metric_results.append(
                        {
                            "name": class_name,
                            "score": score,
                            "reason": reason,
                        }
                    )

            if legal_rag_result:
                taxonomy = legal_rag_result.get("taxonomy")
                legal_rag_result["success_signals"] = derive_success_signals(
                    retrieval_metrics=legal_rag_result.get("retrieval"),
                    judge_result=legal_rag_result.get("judge"),
                    taxonomy=taxonomy if isinstance(taxonomy, str) else None,
                    g_eval_score=scores.get("g_eval_score"),
                    g_eval_threshold=settings.EVAL_G_EVAL_THRESHOLD,
                )

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
                cost_usd = cost_tracker.calculate_cost(
                    generation_model, prompt_tokens, completion_tokens
                )
            else:
                cost_usd = Decimal(str(response.get("metadata", {}).get("cost", 0.0)))

            raw_metrics = {"metric_results": metric_results}
            if legal_rag_result:
                raw_metrics["legal_rag_bench"] = legal_rag_result

            # Concurrent test-case tasks share self.db, so every statement that
            # touches it (artifact flushes, the result commit, progress reads
            # and writes) must run under the lock. Otherwise SQLite raises
            # "cannot commit transaction - SQL statements in progress" when one
            # task commits while another has a flush/select in flight.
            async with self.db_lock:
                retrieval_trace = response.get("retrieval_trace")
                retrieval_trace_artifact = await self.artifact_store.store_json(
                    self.db, retrieval_trace, ArtifactStore.KIND_RETRIEVAL_TRACE
                )

                retrieved_context_data = response.get("context", [])
                retrieved_context_artifact = await self.artifact_store.store_json(
                    self.db, retrieved_context_data, ArtifactStore.KIND_RETRIEVED_CONTEXT
                )

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

                # Update progress (same shared session, so keep it inside the lock)
                current_completed = await self._get_completed_count()
                await self.checkpoint_service.update_progress(
                    self.evaluation_id, current_completed
                )

                if current_completed % 5 == 0:
                    await self.checkpoint_service.save_checkpoint(
                        self.evaluation_id, current_completed, {"last_index": current_completed}
                    )

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

            return True, None

        except Exception as e:
            error_message = f"Test case {i} failed: {str(e)}"
            logger.error(
                "Error processing test case",
                i=i,
                test_case_id=str(test_case.id),
                error=str(e),
            )
            await self.event_log.log_event(
                self.evaluation_id,
                "test_case_error",
                {"error_message": error_message, "test_case_index": i},
            )
            return False, error_message

    @staticmethod
    def _metric_result_field_name(class_name: str) -> str:
        if class_name == "FaithfulnessMetric":
            return "faithfulness"
        if class_name == "AnswerRelevancyMetric":
            return "relevancy"
        if class_name == "ContextualPrecisionMetric":
            return "precision"
        if class_name == "ContextualRecallMetric":
            return "recall"
        if class_name == "GEval":
            return "g_eval"
        return class_name.lower().replace("metric", "")

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

    async def _get_completed_test_case_ids(self) -> set[uuid.UUID]:
        """Get test case IDs that already have a saved result for this evaluation."""
        result = await self.db.execute(
            select(EvaluationResult.test_case_id).where(
                EvaluationResult.evaluation_id == self.evaluation_id,
                EvaluationResult.test_case_id.is_not(None),
            )
        )
        return {row[0] for row in result.all() if row[0] is not None}

    async def _has_result_for_test_case(self, test_case_id: uuid.UUID) -> bool:
        """Return whether this evaluation already has a result for a test case."""
        result = await self.db.execute(
            select(EvaluationResult.id)
            .where(
                EvaluationResult.evaluation_id == self.evaluation_id,
                EvaluationResult.test_case_id == test_case_id,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _collect_legal_rag_summary(self) -> dict[str, Any] | None:
        result = await self.db.execute(
            select(EvaluationResult.raw_metrics_artifact_id).where(
                EvaluationResult.evaluation_id == self.evaluation_id,
                EvaluationResult.raw_metrics_artifact_id.is_not(None),
            )
        )
        artifact_ids = [row[0] for row in result.all() if row[0]]
        legal_results: list[dict[str, Any]] = []
        for artifact_id in artifact_ids:
            raw_metrics = await self.artifact_store.retrieve_json_by_id(self.db, artifact_id)
            if isinstance(raw_metrics, dict) and isinstance(
                raw_metrics.get("legal_rag_bench"), dict
            ):
                legal_results.append(raw_metrics["legal_rag_bench"])
        return summarize_legal_rag_metrics(legal_results)

    async def _calculate_aggregate_metrics(self) -> dict[str, Any]:
        """Calculate aggregate metrics from saved evaluation results."""
        from sqlalchemy import func

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

        legal_rag_summary = await self._collect_legal_rag_summary()
        if legal_rag_summary:
            summary_metrics["legal_rag_bench"] = legal_rag_summary

        return {
            "summary_metrics": summary_metrics,
            "pass_rate": pass_rate,
            "cost_metrics": cost_metrics,
            "performance_metrics": performance_metrics,
        }

    async def _finalize_evaluation(self) -> None:
        """Calculate aggregate metrics and mark evaluation as complete."""
        aggregate_metrics = await self._calculate_aggregate_metrics()

        await self.checkpoint_service.complete_job(
            self.evaluation_id,
            summary_metrics=aggregate_metrics["summary_metrics"],
            pass_rate=aggregate_metrics["pass_rate"],
            cost_metrics=aggregate_metrics["cost_metrics"],
            performance_metrics=aggregate_metrics["performance_metrics"],
        )

        await self.event_log.log_event(
            self.evaluation_id,
            "completed",
            {
                "summary_metrics": aggregate_metrics["summary_metrics"],
                "pass_rate": aggregate_metrics["pass_rate"],
                "cost_metrics": aggregate_metrics["cost_metrics"],
                "performance_metrics": aggregate_metrics["performance_metrics"],
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
                        "pass_rate": aggregate_metrics["pass_rate"],
                        "summary_metrics": aggregate_metrics["summary_metrics"],
                    },
                )
        except Exception as webhook_err:
            logger.error("Failed to trigger completion webhook", error=str(webhook_err))

    async def _fail_evaluation(
        self,
        error_message: str,
        aggregate_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Mark evaluation as failed, optionally preserving aggregate partial results."""
        fail_kwargs = aggregate_metrics or {}
        await self.checkpoint_service.fail_job(
            self.evaluation_id,
            error_message,
            **fail_kwargs,
        )

        event_payload = {"error_message": error_message}
        if aggregate_metrics:
            event_payload.update(aggregate_metrics)
        await self.event_log.log_event(self.evaluation_id, "error", event_payload)

        try:
            if self.evaluation:
                from app.services.webhook_service import get_webhook_service

                webhook_payload = {
                    "evaluation_id": str(self.evaluation_id),
                    "status": "failed",
                    "error_message": error_message,
                }
                if aggregate_metrics:
                    webhook_payload["pass_rate"] = aggregate_metrics["pass_rate"]
                    webhook_payload["summary_metrics"] = aggregate_metrics["summary_metrics"]

                await get_webhook_service().trigger_event(
                    self.db,
                    self.evaluation.project_id,
                    "evaluation.failed",
                    webhook_payload,
                )
        except Exception as webhook_err:
            logger.error("Failed to trigger failure webhook", error=str(webhook_err))


def get_evaluation_runner(db_session: AsyncSession, evaluation_id: uuid.UUID) -> EvaluationRunner:
    """Factory to get the evaluation runner."""
    return EvaluationRunner(db_session, evaluation_id)
