"""Evaluation framework using DeepEval."""

from typing import Any

from deepeval.test_case import LLMTestCase

from rag_evaluator.common.base_rag import BaseRAG


class RAGEvaluator:
    """Evaluator for RAG implementations using DeepEval."""

    def __init__(self, test_cases: list[dict[str, str]]) -> None:
        """Initialize the evaluator.

        Args:
            test_cases: List of test cases, each containing:
                - question: The question to ask
                - expected_answer: The expected answer
                - context: Optional reference context
        """
        self.test_cases = test_cases

    def evaluate(self, rag_impl: BaseRAG) -> dict[str, Any]:
        """Evaluate a RAG implementation.

        Args:
            rag_impl: The RAG implementation to evaluate

        Returns:
            Dictionary containing evaluation results:
                - accuracy_metrics: Faithfulness, answer relevance, context precision
                - performance_metrics: Speed, cost
                - detailed_results: Per-question results
        """
        results = {
            "rag_implementation": rag_impl.name,
            "test_cases_count": len(self.test_cases),
            "detailed_results": [],
            "aggregate_metrics": {},
        }

        for test_case in self.test_cases:
            # Query the RAG implementation
            response = rag_impl.query(test_case["question"])

            # Create DeepEval test case (TODO: use for actual metric evaluation)
            _llm_test_case = LLMTestCase(
                input=test_case["question"],
                actual_output=response["answer"],
                expected_output=test_case.get("expected_answer", ""),
                context=response.get("context", []),
            )

            # Store detailed results
            results["detailed_results"].append(
                {
                    "question": test_case["question"],
                    "answer": response["answer"],
                    "expected": test_case.get("expected_answer", ""),
                    "retrieval_time": response.get("metadata", {}).get("retrieval_time", 0),
                }
            )

        # Get RAG-specific metrics
        results["aggregate_metrics"] = rag_impl.get_metrics()

        return results

    def compare_implementations(self, implementations: list[BaseRAG]) -> dict[str, dict[str, Any]]:
        """Compare multiple RAG implementations.

        Args:
            implementations: List of RAG implementations to compare

        Returns:
            Dictionary mapping implementation names to their evaluation results
        """
        comparison = {}

        for impl in implementations:
            comparison[impl.name] = self.evaluate(impl)

        return comparison
