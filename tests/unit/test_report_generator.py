"""Unit tests for report generator."""

import json
from pathlib import Path

import pytest

from rag_evaluator.evaluation.report_generator import ReportGenerator


@pytest.fixture
def sample_evaluation_results() -> dict:
    """Create sample evaluation results for testing.

    Returns:
        Sample evaluation results dictionary
    """
    return {
        "rag_implementation": "Test RAG Implementation",
        "test_cases_count": 3,
        "timestamp": "2026-01-02 10:30:00",
        "total_evaluation_time": 45.5,
        "pass_rate": 66.7,
        "thresholds": {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "contextual_precision": 0.7,
            "contextual_recall": 0.7,
            "hallucination": 0.7,
        },
        "metrics_summary": {
            "faithfulness_avg": 0.85,
            "faithfulness_min": 0.75,
            "faithfulness_max": 0.95,
            "answer_relevancy_avg": 0.82,
            "answer_relevancy_min": 0.70,
            "answer_relevancy_max": 0.90,
            "contextual_precision_avg": 0.78,
            "contextual_precision_min": 0.65,
            "contextual_precision_max": 0.88,
            "contextual_recall_avg": 0.80,
            "contextual_recall_min": 0.72,
            "contextual_recall_max": 0.85,
            "hallucination_avg": 0.88,
            "hallucination_min": 0.80,
            "hallucination_max": 0.95,
        },
        "performance_metrics": {
            "total_queries": 3,
            "avg_retrieval_time": 0.502,
            "total_chunks_retrieved": 15,
        },
        "detailed_results": [
            {
                "test_case_id": "tc_001",
                "question": "What is RAG?",
                "answer": "RAG is a technique...",
                "expected_answer": "Expected answer...",
                "context_chunks_retrieved": 5,
                "retrieval_time": 0.5,
                "difficulty": "easy",
                "category": "definition",
                "metrics": {
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.90,
                    "contextual_precision": 0.88,
                    "contextual_recall": 0.85,
                    "hallucination": 0.95,
                },
            },
            {
                "test_case_id": "tc_002",
                "question": "How does it work?",
                "answer": "It works by...",
                "expected_answer": "Expected answer 2...",
                "context_chunks_retrieved": 5,
                "retrieval_time": 0.52,
                "difficulty": "medium",
                "category": "explanation",
                "metrics": {
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.85,
                    "contextual_precision": 0.78,
                    "contextual_recall": 0.80,
                    "hallucination": 0.90,
                },
            },
            {
                "test_case_id": "tc_003",
                "question": "Why use RAG?",
                "answer": "Because...",
                "expected_answer": "Expected answer 3...",
                "context_chunks_retrieved": 5,
                "retrieval_time": 0.49,
                "difficulty": "easy",
                "category": "explanation",
                "metrics": {
                    "faithfulness": 0.75,
                    "answer_relevancy": 0.70,
                    "contextual_precision": 0.65,
                    "contextual_recall": 0.72,
                    "hallucination": 0.80,
                },
            },
        ],
    }


@pytest.fixture
def sample_comparison_results(sample_evaluation_results: dict) -> dict:
    """Create sample comparison results for testing.

    Args:
        sample_evaluation_results: Sample evaluation results

    Returns:
        Sample comparison results dictionary
    """
    results_copy = sample_evaluation_results.copy()
    results_copy["rag_implementation"] = "RAG Implementation 2"

    return {
        "Test RAG Implementation": sample_evaluation_results,
        "RAG Implementation 2": results_copy,
    }


def test_report_generator_initialization(tmp_path: Path) -> None:
    """Test report generator initialization."""
    output_dir = tmp_path / "reports"
    generator = ReportGenerator(output_dir=str(output_dir))

    assert generator.output_dir == output_dir
    assert output_dir.exists()


def test_generate_json_report(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test JSON report generation."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_report(sample_evaluation_results, output_format="json")

    assert "json" in files
    json_path = Path(files["json"])
    assert json_path.exists()
    assert json_path.suffix == ".json"

    # Verify JSON content
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["rag_implementation"] == "Test RAG Implementation"
        assert data["test_cases_count"] == 3


def test_generate_markdown_report(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test Markdown report generation."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_report(sample_evaluation_results, output_format="markdown")

    assert "markdown" in files
    md_path = Path(files["markdown"])
    assert md_path.exists()
    assert md_path.suffix == ".md"

    # Verify Markdown content
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
        assert "# RAG Evaluation Report" in content
        assert "Test RAG Implementation" in content
        assert "66.7%" in content  # Pass rate
        assert "tc_001" in content
        assert "What is RAG?" in content


def test_generate_both_reports(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test generating both JSON and Markdown reports."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_report(sample_evaluation_results, output_format="both")

    assert "json" in files
    assert "markdown" in files
    assert Path(files["json"]).exists()
    assert Path(files["markdown"]).exists()


def test_generate_report_invalid_format(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test report generation with invalid format."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid format"):
        generator.generate_report(sample_evaluation_results, output_format="invalid")


def test_generate_comparison_json_report(tmp_path: Path, sample_comparison_results: dict) -> None:
    """Test comparison JSON report generation."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_comparison_report(sample_comparison_results, output_format="json")

    assert "json" in files
    json_path = Path(files["json"])
    assert json_path.exists()

    # Verify JSON content
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        assert "Test RAG Implementation" in data
        assert "RAG Implementation 2" in data


def test_generate_comparison_markdown_report(
    tmp_path: Path, sample_comparison_results: dict
) -> None:
    """Test comparison Markdown report generation."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_comparison_report(
        sample_comparison_results, output_format="markdown"
    )

    assert "markdown" in files
    md_path = Path(files["markdown"])
    assert md_path.exists()

    # Verify Markdown content
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
        assert "# RAG Implementations Comparison Report" in content
        assert "Test RAG Implementation" in content
        assert "RAG Implementation 2" in content
        assert "Summary Comparison" in content


def test_generate_comparison_report_invalid_format(
    tmp_path: Path, sample_comparison_results: dict
) -> None:
    """Test comparison report generation with invalid format."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid format"):
        generator.generate_comparison_report(sample_comparison_results, output_format="html")


def test_markdown_report_contains_metrics(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test that Markdown report contains all expected metrics."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_report(sample_evaluation_results, output_format="markdown")
    md_path = Path(files["markdown"])

    with open(md_path, encoding="utf-8") as f:
        content = f.read()

        # Check for metric names
        assert "Faithfulness" in content
        assert "Answer Relevancy" in content
        assert "Contextual Precision" in content
        assert "Contextual Recall" in content
        assert "Hallucination" in content

        # Check for test case details
        assert "tc_001" in content
        assert "tc_002" in content
        assert "tc_003" in content

        # Check for pass/fail indicators
        assert "✅ PASS" in content or "❌ FAIL" in content


def test_filename_generation(tmp_path: Path, sample_evaluation_results: dict) -> None:
    """Test that filenames are generated correctly."""
    generator = ReportGenerator(output_dir=str(tmp_path))

    files = generator.generate_report(sample_evaluation_results, output_format="both")

    json_filename = Path(files["json"]).name
    md_filename = Path(files["markdown"]).name

    # Both should start with eval_
    assert json_filename.startswith("eval_")
    assert md_filename.startswith("eval_")

    # Both should contain implementation name (sanitized)
    assert "test_rag_implementation" in json_filename
    assert "test_rag_implementation" in md_filename

    # Both should contain timestamp
    assert "2026-01-02" in json_filename
    assert "2026-01-02" in md_filename
