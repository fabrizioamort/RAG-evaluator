"""Evaluation framework using DeepEval."""

import json
import time
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.config import settings


class RAGEvaluator:
    """Evaluator for RAG implementations using DeepEval."""

    def __init__(self, test_set_path: str | None = None) -> None:
        """Initialize the evaluator.

        Args:
            test_set_path: Path to test dataset JSON file.
                          Defaults to settings.eval_test_set_path.
        """
        self.test_set_path = test_set_path or settings.eval_test_set_path
        self.test_cases = self._load_test_set()

        # Initialize DeepEval metrics with thresholds from settings
        self.metrics = [
            FaithfulnessMetric(
                threshold=settings.eval_faithfulness_threshold,
                model=settings.openai_model,
                include_reason=True,
                async_mode=settings.deepeval_async_mode,
            ),
            AnswerRelevancyMetric(
                threshold=settings.eval_answer_relevancy_threshold,
                model=settings.openai_model,
                include_reason=True,
                async_mode=settings.deepeval_async_mode,
            ),
            ContextualPrecisionMetric(
                threshold=settings.eval_contextual_precision_threshold,
                model=settings.openai_model,
                include_reason=True,
                async_mode=settings.deepeval_async_mode,
            ),
            ContextualRecallMetric(
                threshold=settings.eval_contextual_recall_threshold,
                model=settings.openai_model,
                include_reason=True,
                async_mode=settings.deepeval_async_mode,
            ),
        ]

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

        start_time = time.time()
        deepeval_test_cases: list[LLMTestCase] = []
        detailed_results: list[dict[str, Any]] = []

        # Execute queries and create DeepEval test cases
        for i, test_case in enumerate(self.test_cases, 1):
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
            deepeval_test_cases.append(llm_test_case)

            # Store detailed result
            detailed_results.append(
                {
                    "test_case_id": test_case.get("id", f"tc_{i:03d}"),
                    "question": test_case["question"],
                    "answer": response["answer"],
                    "expected_answer": test_case.get("expected_answer", ""),
                    "context_chunks_retrieved": len(context_list),
                    "retrieval_time": response.get("metadata", {}).get("retrieval_time", 0.0),
                    "difficulty": test_case.get("difficulty", "unknown"),
                    "category": test_case.get("category", "general"),
                }
            )

        if verbose:
            print(f"\n{'=' * 60}")
            print("Running DeepEval metrics evaluation...")
            print(f"{'=' * 60}\n")

        # Run DeepEval evaluation
        evaluation_results = evaluate(deepeval_test_cases, self.metrics)  # type: ignore[operator]

        # Extract metric scores from evaluation results
        # evaluation_results.test_results contains individual test results with metrics_data
        if hasattr(evaluation_results, "test_results"):
            for i, test_result in enumerate(evaluation_results.test_results):
                if i < len(detailed_results):
                    # Extract scores from metrics_data
                    metrics_dict: dict[str, float | None] = {
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "contextual_precision": None,
                        "contextual_recall": None,
                    }

                    # Iterate through metrics_data to extract scores
                    if hasattr(test_result, "metrics_data"):
                        for metric_data in test_result.metrics_data:
                            metric_name = metric_data.name.lower().replace(" ", "_")
                            # Map DeepEval metric names to our naming convention
                            if "faithfulness" in metric_name:
                                metrics_dict["faithfulness"] = metric_data.score
                            elif "answer" in metric_name and "relevancy" in metric_name:
                                metrics_dict["answer_relevancy"] = metric_data.score
                            elif "contextual" in metric_name and "precision" in metric_name:
                                metrics_dict["contextual_precision"] = metric_data.score
                            elif "contextual" in metric_name and "recall" in metric_name:
                                metrics_dict["contextual_recall"] = metric_data.score

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
                "faithfulness": settings.eval_faithfulness_threshold,
                "answer_relevancy": settings.eval_answer_relevancy_threshold,
                "contextual_precision": settings.eval_contextual_precision_threshold,
                "contextual_recall": settings.eval_contextual_recall_threshold,
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
        metric_scores: dict[str, list[float]] = {
            "faithfulness": [],
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
        }

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
            "faithfulness": settings.eval_faithfulness_threshold,
            "answer_relevancy": settings.eval_answer_relevancy_threshold,
            "contextual_precision": settings.eval_contextual_precision_threshold,
            "contextual_recall": settings.eval_contextual_recall_threshold,
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
