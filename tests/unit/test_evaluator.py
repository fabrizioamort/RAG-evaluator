"""Unit tests for RAG evaluator."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.evaluation.evaluator import RAGEvaluator


class MockRAG(BaseRAG):
    """Mock RAG implementation for testing."""

    def __init__(self) -> None:
        """Initialize mock RAG."""
        super().__init__("Mock RAG")
        self.query_count = 0

    def prepare_documents(self, documents_path: str) -> None:
        """Mock prepare_documents."""
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Mock query method."""
        self.query_count += 1
        return {
            "answer": f"Mock answer to: {question}",
            "context": [
                f"Mock context chunk 1 for {question}",
                f"Mock context chunk 2 for {question}",
            ],
            "metadata": {"retrieval_time": 0.1},
        }

    def get_metrics(self) -> dict[str, Any]:
        """Mock get_metrics."""
        return {
            "total_queries": self.query_count,
            "avg_retrieval_time": 0.1,
        }


@pytest.fixture
def temp_test_set(tmp_path: Path) -> Path:
    """Create temporary test set file.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to temporary test set file
    """
    test_set = {
        "metadata": {"version": "1.0", "description": "Test dataset"},
        "test_cases": [
            {
                "id": "tc_001",
                "question": "What is RAG?",
                "expected_answer": "RAG is a technique...",
                "ground_truth_context": ["Context 1"],
                "difficulty": "easy",
                "category": "definition",
            },
            {
                "id": "tc_002",
                "question": "How does RAG work?",
                "expected_answer": "RAG works by...",
                "ground_truth_context": ["Context 2"],
                "difficulty": "medium",
                "category": "explanation",
            },
        ],
    }

    test_set_path = tmp_path / "test_set.json"
    with open(test_set_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f)

    return test_set_path


def test_evaluator_initialization(temp_test_set: Path) -> None:
    """Test evaluator initialization."""
    with patch("rag_evaluator.evaluation.evaluator.FaithfulnessMetric"), \
         patch("rag_evaluator.evaluation.evaluator.AnswerRelevancyMetric"), \
         patch("rag_evaluator.evaluation.evaluator.ContextualPrecisionMetric"), \
         patch("rag_evaluator.evaluation.evaluator.ContextualRecallMetric"), \
         patch("rag_evaluator.evaluation.evaluator.HallucinationMetric"):
        evaluator = RAGEvaluator(test_set_path=str(temp_test_set))

        assert evaluator.test_set_path == str(temp_test_set)
        assert len(evaluator.test_cases) == 2
        assert len(evaluator.metrics) == 5  # 5 DeepEval metrics


def test_evaluator_initialization_default_path() -> None:
    """Test evaluator initialization with default path."""
    with patch("rag_evaluator.evaluation.evaluator.settings") as mock_settings:
        mock_settings.eval_test_set_path = "default/path.json"

        # Should raise FileNotFoundError since default path doesn't exist
        with pytest.raises(FileNotFoundError):
            RAGEvaluator()


def test_load_test_set_invalid_json(tmp_path: Path) -> None:
    """Test loading invalid JSON test set."""
    invalid_json_path = tmp_path / "invalid.json"
    with open(invalid_json_path, "w", encoding="utf-8") as f:
        f.write("{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON"):
        RAGEvaluator(test_set_path=str(invalid_json_path))


def test_load_test_set_missing_file() -> None:
    """Test loading non-existent test set file."""
    with pytest.raises(FileNotFoundError):
        RAGEvaluator(test_set_path="nonexistent.json")


def test_calculate_pass_rate_all_pass() -> None:
    """Test pass rate calculation when all tests pass."""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)

    # Mock settings for thresholds
    with patch("rag_evaluator.evaluation.evaluator.settings") as mock_settings:
        mock_settings.eval_faithfulness_threshold = 0.7
        mock_settings.eval_answer_relevancy_threshold = 0.7
        mock_settings.eval_contextual_precision_threshold = 0.7
        mock_settings.eval_contextual_recall_threshold = 0.7
        mock_settings.eval_hallucination_threshold = 0.7

        detailed_results = [
            {
                "metrics": {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.9,
                    "contextual_precision": 0.85,
                    "contextual_recall": 0.75,
                    "hallucination": 0.8,
                }
            },
            {
                "metrics": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.95,
                    "contextual_precision": 0.9,
                    "contextual_recall": 0.85,
                    "hallucination": 0.9,
                }
            },
        ]

        pass_rate = evaluator._calculate_pass_rate(detailed_results)
        assert pass_rate == 100.0


def test_calculate_pass_rate_partial_pass() -> None:
    """Test pass rate calculation with partial passes."""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)

    with patch("rag_evaluator.evaluation.evaluator.settings") as mock_settings:
        mock_settings.eval_faithfulness_threshold = 0.7
        mock_settings.eval_answer_relevancy_threshold = 0.7
        mock_settings.eval_contextual_precision_threshold = 0.7
        mock_settings.eval_contextual_recall_threshold = 0.7
        mock_settings.eval_hallucination_threshold = 0.7

        detailed_results = [
            {
                "metrics": {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.9,
                    "contextual_precision": 0.85,
                    "contextual_recall": 0.75,
                    "hallucination": 0.8,
                }
            },
            {
                "metrics": {
                    "faithfulness": 0.5,  # Below threshold
                    "answer_relevancy": 0.95,
                    "contextual_precision": 0.9,
                    "contextual_recall": 0.85,
                    "hallucination": 0.9,
                }
            },
        ]

        pass_rate = evaluator._calculate_pass_rate(detailed_results)
        assert pass_rate == 50.0


def test_calculate_pass_rate_empty_results() -> None:
    """Test pass rate calculation with empty results."""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    pass_rate = evaluator._calculate_pass_rate([])
    assert pass_rate == 0.0


def test_calculate_metrics_summary() -> None:
    """Test metrics summary calculation."""
    evaluator = RAGEvaluator.__new__(RAGEvaluator)

    # Create mock detailed results with metric scores
    detailed_results = []
    for i in range(3):
        detailed_results.append(
            {
                "test_case_id": f"tc_{i}",
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "metrics": {
                    "faithfulness": 0.8 + (i * 0.05),
                    "answer_relevancy": 0.75 + (i * 0.05),
                    "contextual_precision": 0.7 + (i * 0.05),
                    "contextual_recall": 0.85 + (i * 0.05),
                    "hallucination": 0.9 + (i * 0.05),
                },
            }
        )

    summary = evaluator._calculate_metrics_summary_from_detailed(detailed_results)

    # Check that averages are calculated
    assert "faithfulness_avg" in summary
    assert "faithfulness_min" in summary
    assert "faithfulness_max" in summary

    # Verify faithfulness calculations
    assert summary["faithfulness_avg"] == pytest.approx(0.85, abs=0.01)
    assert summary["faithfulness_min"] == pytest.approx(0.8, abs=0.01)
    assert summary["faithfulness_max"] == pytest.approx(0.9, abs=0.01)


def test_compare_implementations(temp_test_set: Path) -> None:
    """Test comparing multiple implementations."""
    with patch("rag_evaluator.evaluation.evaluator.evaluate") as mock_evaluate, \
         patch("rag_evaluator.evaluation.evaluator.FaithfulnessMetric"), \
         patch("rag_evaluator.evaluation.evaluator.AnswerRelevancyMetric"), \
         patch("rag_evaluator.evaluation.evaluator.ContextualPrecisionMetric"), \
         patch("rag_evaluator.evaluation.evaluator.ContextualRecallMetric"), \
         patch("rag_evaluator.evaluation.evaluator.HallucinationMetric"):
        # Mock the DeepEval evaluate function to return success
        mock_evaluate.return_value = None

        evaluator = RAGEvaluator(test_set_path=str(temp_test_set))

        # Create mock RAG implementations
        mock_rag1 = MockRAG()
        mock_rag1.name = "Mock RAG 1"

        mock_rag2 = MockRAG()
        mock_rag2.name = "Mock RAG 2"

        # Run comparison (with mocked DeepEval)
        comparison = evaluator.compare_implementations([mock_rag1, mock_rag2], verbose=False)

        # Verify both implementations were evaluated
        assert "Mock RAG 1" in comparison
        assert "Mock RAG 2" in comparison
        assert comparison["Mock RAG 1"]["test_cases_count"] == 2
        assert comparison["Mock RAG 2"]["test_cases_count"] == 2
