"""Evaluation framework using DeepEval."""

import json
import time
from pathlib import Path
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.config import settings

# DeepEval is imported lazily to avoid import-time side effects during pytest collection.
evaluate: Any | None = None
AsyncConfig: Any | None = None
AnswerRelevancyMetric: Any | None = None
ContextualPrecisionMetric: Any | None = None
ContextualRecallMetric: Any | None = None
FaithfulnessMetric: Any | None = None
GEval: Any | None = None
LLMTestCase: Any | None = None
LLMTestCaseParams: Any | None = None


def _ensure_deepeval_loaded() -> None:
    """Import DeepEval symbols on demand.

    Imports are assigned only when the symbol is still None, so existing test patches
    (e.g. patching FaithfulnessMetric) are preserved.
    """
    global evaluate
    global AsyncConfig
    global AnswerRelevancyMetric
    global ContextualPrecisionMetric
    global ContextualRecallMetric
    global FaithfulnessMetric
    global GEval
    global LLMTestCase
    global LLMTestCaseParams

    if all(
        symbol is not None
        for symbol in (
            evaluate,
            AsyncConfig,
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

    from deepeval import evaluate as deepeval_evaluate
    from deepeval.evaluate import AsyncConfig as DeepEvalAsyncConfig
    from deepeval.metrics import (
        AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
    )
    from deepeval.metrics import (
        ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
    )
    from deepeval.metrics import (
        ContextualRecallMetric as DeepEvalContextualRecallMetric,
    )
    from deepeval.metrics import (
        FaithfulnessMetric as DeepEvalFaithfulnessMetric,
    )
    from deepeval.metrics import (
        GEval as DeepEvalGEval,
    )
    from deepeval.test_case import (
        LLMTestCase as DeepEvalLLMTestCase,
    )
    from deepeval.test_case import (
        LLMTestCaseParams as DeepEvalLLMTestCaseParams,
    )

    if evaluate is None:
        evaluate = deepeval_evaluate
    if AsyncConfig is None:
        AsyncConfig = DeepEvalAsyncConfig
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


class RAGEvaluator:
    """Evaluator for RAG implementations using DeepEval."""

    def __init__(
        self,
        test_set_path: str | None = None,
        selected_metrics: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            test_set_path: Path to test dataset JSON file.
                          Defaults to settings.eval_test_set_path.
            selected_metrics: List of metric names to use.
                             Available: faithfulness, answer_relevancy,
                             contextual_precision, contextual_recall, g_eval.
                             If None, all current metrics are used.
        """
        self.test_set_path = test_set_path or settings.eval_test_set_path
        self.test_cases = self._load_test_set()
        self.selected_metrics = selected_metrics or [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
            "g_eval",
        ]

        # Initialize DeepEval metrics with thresholds from settings
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> list[Any]:
        """Initialize selected DeepEval metrics."""
        _ensure_deepeval_loaded()
        assert FaithfulnessMetric is not None
        assert AnswerRelevancyMetric is not None
        assert ContextualPrecisionMetric is not None
        assert ContextualRecallMetric is not None
        assert GEval is not None
        assert LLMTestCaseParams is not None

        metrics: list[Any] = []

        if "faithfulness" in self.selected_metrics:
            metrics.append(
                FaithfulnessMetric(
                    threshold=settings.eval_faithfulness_threshold,
                    model=settings.openai_model,
                    include_reason=settings.eval_include_reason,
                    async_mode=settings.deepeval_async_mode,
                )
            )

        if "answer_relevancy" in self.selected_metrics:
            metrics.append(
                AnswerRelevancyMetric(
                    threshold=settings.eval_answer_relevancy_threshold,
                    model=settings.openai_model,
                    include_reason=settings.eval_include_reason,
                    async_mode=settings.deepeval_async_mode,
                )
            )

        if "contextual_precision" in self.selected_metrics:
            metrics.append(
                ContextualPrecisionMetric(
                    threshold=settings.eval_contextual_precision_threshold,
                    model=settings.openai_model,
                    include_reason=settings.eval_include_reason,
                    async_mode=settings.deepeval_async_mode,
                )
            )

        if "contextual_recall" in self.selected_metrics:
            metrics.append(
                ContextualRecallMetric(
                    threshold=settings.eval_contextual_recall_threshold,
                    model=settings.openai_model,
                    include_reason=settings.eval_include_reason,
                    async_mode=settings.deepeval_async_mode,
                )
            )

        if "g_eval" in self.selected_metrics:
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
                    threshold=settings.eval_g_eval_threshold,
                    model=settings.openai_model,
                    async_mode=settings.deepeval_async_mode,
                )
            )

        return metrics

    def _load_test_set(self) -> list[dict[str, Any]]:
        """Load test cases from JSON file.

        Returns:
            List of test case dictionaries.

        Raises:
            FileNotFoundError: If test set file doesn't exist.
            ValueError: If test set JSON is invalid.
        """
        test_set_file = Path(self.test_set_path)

        if not test_set_file.exists():
            raise FileNotFoundError(f"Test set file not found: {self.test_set_path}")

        try:
            with open(test_set_file, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("test_cases", [])  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in test set file: {e}") from e

    def evaluate(self, rag_impl: BaseRAG, verbose: bool = False) -> dict[str, Any]:
        """Evaluate a RAG implementation.

        Args:
            rag_impl: The RAG implementation to evaluate
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing evaluation results:
                - rag_implementation: Name of the RAG implementation
                - test_cases_count: Number of test cases evaluated
                - timestamp: Evaluation timestamp
                - metrics_summary: Aggregated metric scores
                - detailed_results: Per-question results
                - performance_metrics: Speed and cost metrics from RAG
                - pass_rate: Percentage of test cases passing all thresholds
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {rag_impl.name}")
            print(f"Test cases: {len(self.test_cases)}")
            print(f"{'=' * 60}\n")

        _ensure_deepeval_loaded()
        assert LLMTestCase is not None
        assert AsyncConfig is not None
        assert evaluate is not None

        start_time = time.time()
        deepeval_test_cases: list[Any] = []
        detailed_results: list[dict[str, Any]] = []

        def process_test_case(i_test_case: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            i, test_case = i_test_case
            if verbose:
                print(f"[{i}/{len(self.test_cases)}] Querying: {test_case['question'][:60]}...")

            # Query the RAG implementation
            response = rag_impl.query(test_case["question"])

            # Extract context strings from response
            context_list = response.get("context", [])
            # Ensure context is never None (DeepEval requires a list)
            if context_list is None:
                context_list = []
            # Ensure all context items are strings
            context_list = [str(item) if item is not None else "" for item in context_list]

            # Extract ground truth context for evaluation
            ground_truth_context = test_case.get("ground_truth_context", [])
            if ground_truth_context is None:
                ground_truth_context = []

            # Create DeepEval test case
            llm_test_case = LLMTestCase(
                input=test_case["question"],
                actual_output=response["answer"],
                expected_output=test_case.get("expected_answer", ""),
                context=ground_truth_context,  # Ground truth context
                retrieval_context=context_list,  # Actually retrieved context
            )

            # Return both the test case and detailed result
            return {
                "index": i - 1,
                "llm_test_case": llm_test_case,
                "detailed_result": {
                    "test_case_id": test_case.get("id", f"tc_{i:03d}"),
                    "question": test_case["question"],
                    "answer": response["answer"],
                    "expected_answer": test_case.get("expected_answer", ""),
                    "context_chunks_retrieved": len(context_list),
                    "retrieval_time": response.get("metadata", {}).get("retrieval_time", 0.0),
                    "difficulty": test_case.get("difficulty", "unknown"),
                    "category": test_case.get("category", "general"),
                },
            }

        # Execute queries (parallel or sequential)
        indexed_test_cases = list(enumerate(self.test_cases, 1))

        if settings.eval_parallel_queries:
            if verbose:
                print(
                    f"Running RAG queries in parallel (max workers: {settings.eval_max_workers})..."
                )
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=settings.eval_max_workers) as executor:
                query_results = list(executor.map(process_test_case, indexed_test_cases))
        else:
            query_results = [process_test_case(tc) for tc in indexed_test_cases]

        # Sort results back to original order and extract components
        query_results.sort(key=lambda x: x["index"])
        for res in query_results:
            deepeval_test_cases.append(res["llm_test_case"])
            detailed_results.append(res["detailed_result"])

        if verbose:
            print(f"\n{'=' * 60}")
            print("Running DeepEval metrics evaluation...")
            print(f"Async mode: {settings.deepeval_async_mode}")
            if settings.deepeval_async_mode:
                print(f"Max concurrent tasks: {settings.deepeval_max_concurrent}")
                print(f"Throttle value: {settings.deepeval_throttle_value}")
            print(f"{'=' * 60}\n")

        # Run DeepEval evaluation
        # Use AsyncConfig to control parallel/sequential execution
        async_config = AsyncConfig(
            run_async=settings.deepeval_async_mode,
            max_concurrent=settings.deepeval_max_concurrent,
            throttle_value=settings.deepeval_throttle_value,
        )
        evaluation_results = evaluate(  # type: ignore[operator]
            deepeval_test_cases,
            self.metrics,
            async_config=async_config,
        )

        # Extract metric scores from evaluation results
        # evaluation_results.test_results contains individual test results with metrics_data
        if hasattr(evaluation_results, "test_results"):
            for i, test_result in enumerate(evaluation_results.test_results):
                if i < len(detailed_results):
                    # Extract scores from metrics_data
                    metrics_dict: dict[str, float | None] = {
                        name: None for name in self.selected_metrics
                    }

                    # Iterate through metrics_data to extract scores and reasoning
                    if hasattr(test_result, "metrics_data"):
                        for metric_data in test_result.metrics_data:
                            metric_name = metric_data.name.lower().replace(" ", "_")
                            # Map DeepEval metric names to our naming convention
                            target_key = None
                            if "faithfulness" in metric_name:
                                target_key = "faithfulness"
                            elif "answer" in metric_name and "relevancy" in metric_name:
                                target_key = "answer_relevancy"
                            elif "contextual" in metric_name and "precision" in metric_name:
                                target_key = "contextual_precision"
                            elif "contextual" in metric_name and "recall" in metric_name:
                                target_key = "contextual_recall"
                            elif "correctness" in metric_name or "g_eval" in metric_name:
                                target_key = "g_eval"

                            if target_key and target_key in metrics_dict:
                                metrics_dict[target_key] = metric_data.score
                                # Capture reasoning if available
                                if hasattr(metric_data, "reason") and metric_data.reason:
                                    detailed_results[i][f"{target_key}_reason"] = metric_data.reason

                    detailed_results[i]["metrics"] = metrics_dict

        # Calculate aggregate metrics from detailed results
        metrics_summary = self._calculate_metrics_summary_from_detailed(detailed_results)

        # Calculate pass rate
        pass_rate = self._calculate_pass_rate(detailed_results)

        total_time = time.time() - start_time

        results = {
            "rag_implementation": rag_impl.name,
            "test_cases_count": len(self.test_cases),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluation_time": round(total_time, 2),
            "metrics_summary": metrics_summary,
            "detailed_results": detailed_results,
            "performance_metrics": rag_impl.get_metrics(),
            "pass_rate": pass_rate,
            "thresholds": {
                name: getattr(settings, f"eval_{name}_threshold")
                for name in self.selected_metrics
                if hasattr(settings, f"eval_{name}_threshold")
            },
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Evaluation complete in {total_time:.2f}s")
            print(f"Pass rate: {pass_rate:.1f}%")
            print(f"{'=' * 60}\n")

        return results

    def _calculate_metrics_summary_from_detailed(
        self, detailed_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate aggregate metrics from detailed results.

        Args:
            detailed_results: List of detailed test case results

        Returns:
            Dictionary with average scores for each metric
        """
        metric_scores: dict[str, list[float]] = {name: [] for name in self.selected_metrics}

        # Extract scores from detailed results
        for result in detailed_results:
            metrics = result.get("metrics", {})
            for metric_name in metric_scores.keys():
                score = metrics.get(metric_name)
                if score is not None:
                    metric_scores[metric_name].append(score)

        # Calculate averages
        summary = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                summary[f"{metric_name}_avg"] = round(sum(scores) / len(scores), 3)
                summary[f"{metric_name}_min"] = round(min(scores), 3)
                summary[f"{metric_name}_max"] = round(max(scores), 3)
            else:
                summary[f"{metric_name}_avg"] = 0.0
                summary[f"{metric_name}_min"] = 0.0
                summary[f"{metric_name}_max"] = 0.0

        return summary

    def _calculate_pass_rate(self, detailed_results: list[dict[str, Any]]) -> float:
        """Calculate percentage of test cases passing all metric thresholds.

        Args:
            detailed_results: List of detailed test case results

        Returns:
            Pass rate as percentage (0-100)
        """
        if not detailed_results:
            return 0.0

        passed_count = 0
        thresholds = {
            name: getattr(settings, f"eval_{name}_threshold")
            for name in self.selected_metrics
            if hasattr(settings, f"eval_{name}_threshold")
        }

        for result in detailed_results:
            metrics = result.get("metrics", {})
            all_passed = True

            for metric_name, threshold in thresholds.items():
                score = metrics.get(metric_name)
                if score is None or score < threshold:
                    all_passed = False
                    break

            if all_passed:
                passed_count += 1

        return round((passed_count / len(detailed_results)) * 100, 1)

    def compare_implementations(
        self, implementations: list[BaseRAG], verbose: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple RAG implementations.

        Args:
            implementations: List of RAG implementations to compare
            verbose: Whether to print detailed progress

        Returns:
            Dictionary mapping implementation names to their evaluation results
        """
        comparison = {}

        for impl in implementations:
            comparison[impl.name] = self.evaluate(impl, verbose=verbose)

        return comparison
