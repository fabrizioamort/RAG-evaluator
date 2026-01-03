"""Unit tests for statistics module."""

import pytest

from rag_evaluator.evaluation.statistics import (
    StatisticalSummary,
    calculate_statistics,
    compare_implementations_statistically,
)


def test_calculate_statistics_with_valid_scores() -> None:
    """Test calculate_statistics with valid score list."""
    scores = [0.8, 0.85, 0.9, 0.75, 0.95]

    result = calculate_statistics(scores)

    assert isinstance(result, StatisticalSummary)
    assert result.mean == pytest.approx(0.85, abs=0.01)
    assert result.median == pytest.approx(0.85, abs=0.01)
    assert result.min == 0.75
    assert result.max == 0.95
    assert result.std_dev > 0


def test_calculate_statistics_with_empty_list() -> None:
    """Test calculate_statistics with empty list."""
    scores: list[float] = []

    result = calculate_statistics(scores)

    assert result.mean == 0.0
    assert result.median == 0.0
    assert result.std_dev == 0.0
    assert result.min == 0.0
    assert result.max == 0.0
    assert result.confidence_interval_95 == (0.0, 0.0)


def test_calculate_statistics_with_single_value() -> None:
    """Test calculate_statistics with single value."""
    scores = [0.8]

    result = calculate_statistics(scores)

    assert result.mean == 0.8
    assert result.median == 0.8
    assert result.std_dev == 0.0
    assert result.min == 0.8
    assert result.max == 0.8


def test_calculate_statistics_confidence_interval() -> None:
    """Test confidence interval calculation."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]

    result = calculate_statistics(scores)

    # CI should be around the mean
    assert result.confidence_interval_95[0] <= result.mean
    assert result.confidence_interval_95[1] >= result.mean
    # CI should be symmetric
    lower_margin = result.mean - result.confidence_interval_95[0]
    upper_margin = result.confidence_interval_95[1] - result.mean
    assert lower_margin == pytest.approx(upper_margin, abs=0.01)


def test_compare_implementations_statistically_with_valid_data() -> None:
    """Test statistical comparison with valid data."""
    results_a = [0.8, 0.85, 0.9, 0.75, 0.95]
    results_b = [0.7, 0.75, 0.8, 0.65, 0.85]

    comparison = compare_implementations_statistically(
        results_a, results_b, "Implementation A", "Implementation B"
    )

    assert "t_statistic" in comparison
    assert "p_value" in comparison
    assert "significant" in comparison
    assert "better_implementation" in comparison
    assert "interpretation" in comparison
    assert isinstance(comparison["significant"], bool)


def test_compare_implementations_statistically_with_different_lengths() -> None:
    """Test statistical comparison with different length arrays."""
    results_a = [0.8, 0.85, 0.9]
    results_b = [0.7, 0.75]

    comparison = compare_implementations_statistically(
        results_a, results_b, "Implementation A", "Implementation B"
    )

    assert "error" in comparison
    assert "Cannot compare" in comparison["error"]


def test_compare_implementations_statistically_with_insufficient_data() -> None:
    """Test statistical comparison with insufficient data."""
    results_a = [0.8]
    results_b = [0.7]

    comparison = compare_implementations_statistically(
        results_a, results_b, "Implementation A", "Implementation B"
    )

    assert "error" in comparison
    assert "Insufficient data" in comparison["error"]


def test_compare_implementations_statistically_no_difference() -> None:
    """Test statistical comparison with identical results."""
    results_a = [0.8, 0.8, 0.8, 0.8, 0.8]
    results_b = [0.8, 0.8, 0.8, 0.8, 0.8]

    comparison = compare_implementations_statistically(
        results_a, results_b, "Implementation A", "Implementation B"
    )

    # With identical results, there should be no significant difference
    assert comparison["significant"] is False
    assert "No statistically significant difference" in comparison["interpretation"]
