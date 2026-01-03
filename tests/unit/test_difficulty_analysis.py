"""Unit tests for difficulty_analysis module."""

from typing import Any

import pytest

from rag_evaluator.evaluation.difficulty_analysis import (
    analyze_by_difficulty,
    compare_difficulty_performance,
)


def test_analyze_by_difficulty_with_valid_data() -> None:
    """Test analyze_by_difficulty with valid test results."""
    detailed_results = [
        {
            "test_case_id": "test_1",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.9, "answer_relevancy": 0.85},
        },
        {
            "test_case_id": "test_2",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.95, "answer_relevancy": 0.9},
        },
        {
            "test_case_id": "test_3",
            "difficulty": "medium",
            "metrics": {"faithfulness": 0.75, "answer_relevancy": 0.7},
        },
        {
            "test_case_id": "test_4",
            "difficulty": "hard",
            "metrics": {"faithfulness": 0.6, "answer_relevancy": 0.55},
        },
    ]

    result = analyze_by_difficulty(detailed_results, metric_name="faithfulness")

    assert "easy" in result
    assert "medium" in result
    assert "hard" in result

    # Check easy difficulty
    assert result["easy"]["count"] == 2
    assert result["easy"]["mean"] == pytest.approx(0.925, abs=0.01)
    assert result["easy"]["min"] == 0.9
    assert result["easy"]["max"] == 0.95

    # Check medium difficulty
    assert result["medium"]["count"] == 1
    assert result["medium"]["mean"] == 0.75

    # Check hard difficulty
    assert result["hard"]["count"] == 1
    assert result["hard"]["mean"] == 0.6


def test_analyze_by_difficulty_with_missing_difficulty() -> None:
    """Test analyze_by_difficulty with missing difficulty labels."""
    detailed_results = [
        {
            "test_case_id": "test_1",
            "metrics": {"faithfulness": 0.9},
        },
        {
            "test_case_id": "test_2",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.95},
        },
    ]

    result = analyze_by_difficulty(detailed_results, metric_name="faithfulness")

    assert "easy" in result
    assert "unknown" in result
    assert result["unknown"]["count"] == 1


def test_analyze_by_difficulty_with_missing_metrics() -> None:
    """Test analyze_by_difficulty with missing metric values."""
    detailed_results = [
        {
            "test_case_id": "test_1",
            "difficulty": "easy",
            "metrics": {"answer_relevancy": 0.85},  # Missing faithfulness
        },
        {
            "test_case_id": "test_2",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.95},
        },
    ]

    result = analyze_by_difficulty(detailed_results, metric_name="faithfulness")

    # Should only count the one with faithfulness metric
    assert result["easy"]["count"] == 1
    assert result["easy"]["mean"] == 0.95


def test_analyze_by_difficulty_with_empty_list() -> None:
    """Test analyze_by_difficulty with empty list."""
    detailed_results: list[dict[str, Any]] = []

    result = analyze_by_difficulty(detailed_results, metric_name="faithfulness")

    assert result == {}


def test_analyze_by_difficulty_std_dev_calculation() -> None:
    """Test standard deviation calculation."""
    detailed_results = [
        {
            "test_case_id": "test_1",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.5},
        },
        {
            "test_case_id": "test_2",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.7},
        },
        {
            "test_case_id": "test_3",
            "difficulty": "easy",
            "metrics": {"faithfulness": 0.9},
        },
    ]

    result = analyze_by_difficulty(detailed_results, metric_name="faithfulness")

    # With multiple values, std should be > 0
    assert result["easy"]["std"] > 0
    assert result["easy"]["mean"] == pytest.approx(0.7, abs=0.01)


def test_compare_difficulty_performance_with_multiple_implementations() -> None:
    """Test compare_difficulty_performance with multiple implementations."""
    comparison_results = {
        "Implementation A": {
            "detailed_results": [
                {
                    "test_case_id": "test_1",
                    "difficulty": "easy",
                    "metrics": {"faithfulness": 0.9},
                },
                {
                    "test_case_id": "test_2",
                    "difficulty": "medium",
                    "metrics": {"faithfulness": 0.8},
                },
            ]
        },
        "Implementation B": {
            "detailed_results": [
                {
                    "test_case_id": "test_1",
                    "difficulty": "easy",
                    "metrics": {"faithfulness": 0.85},
                },
                {
                    "test_case_id": "test_2",
                    "difficulty": "medium",
                    "metrics": {"faithfulness": 0.75},
                },
            ]
        },
    }

    result = compare_difficulty_performance(comparison_results)

    assert "easy" in result
    assert "medium" in result

    assert "Implementation A" in result["easy"]
    assert "Implementation B" in result["easy"]

    # Implementation A should have higher scores
    assert result["easy"]["Implementation A"] > result["easy"]["Implementation B"]
    assert result["medium"]["Implementation A"] > result["medium"]["Implementation B"]


def test_compare_difficulty_performance_with_empty_results() -> None:
    """Test compare_difficulty_performance with empty results."""
    comparison_results: dict[str, dict[str, Any]] = {}

    result = compare_difficulty_performance(comparison_results)

    assert result == {}


def test_compare_difficulty_performance_with_missing_difficulties() -> None:
    """Test compare_difficulty_performance with implementations having different difficulties."""
    comparison_results = {
        "Implementation A": {
            "detailed_results": [
                {
                    "test_case_id": "test_1",
                    "difficulty": "easy",
                    "metrics": {"faithfulness": 0.9},
                },
            ]
        },
        "Implementation B": {
            "detailed_results": [
                {
                    "test_case_id": "test_1",
                    "difficulty": "hard",
                    "metrics": {"faithfulness": 0.6},
                },
            ]
        },
    }

    result = compare_difficulty_performance(comparison_results)

    # Should have separate entries for each difficulty
    assert "easy" in result
    assert "hard" in result
    assert "Implementation A" in result["easy"]
    assert "Implementation B" in result["hard"]
