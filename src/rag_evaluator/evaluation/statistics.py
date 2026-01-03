"""Statistical analysis module for evaluation results."""

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class StatisticalSummary:
    """Statistical summary for a metric."""

    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    confidence_interval_95: tuple[float, float]


def calculate_statistics(scores: list[float]) -> StatisticalSummary:
    """Calculate statistical summary for a list of scores.

    Args:
        scores: List of numerical scores to analyze

    Returns:
        StatisticalSummary with mean, median, std_dev, min, max, and 95% CI
    """
    if not scores:
        return StatisticalSummary(0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0))

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # 95% confidence interval (t-distribution approximation)
    n = len(scores)
    margin = 1.96 * (std_dev / (n**0.5)) if n > 1 else 0.0
    ci_95 = (mean - margin, mean + margin)

    return StatisticalSummary(
        mean=round(mean, 3),
        median=round(median, 3),
        std_dev=round(std_dev, 3),
        min=round(min(scores), 3),
        max=round(max(scores), 3),
        confidence_interval_95=(round(ci_95[0], 3), round(ci_95[1], 3)),
    )


def compare_implementations_statistically(
    results_a: list[float],
    results_b: list[float],
    impl_a_name: str,
    impl_b_name: str,
) -> dict[str, Any]:
    """Perform statistical comparison between two RAG implementations.

    Uses paired t-test to determine if there's a statistically significant
    difference between the two implementations.

    Args:
        results_a: Scores from implementation A
        results_b: Scores from implementation B
        impl_a_name: Name of implementation A
        impl_b_name: Name of implementation B

    Returns:
        Dictionary with t_statistic, p_value, significance, and interpretation
    """
    from scipy import stats

    if len(results_a) != len(results_b):
        return {
            "error": "Cannot compare - different number of test cases",
            "interpretation": "Implementations must be evaluated on same test set",
        }

    if len(results_a) < 2:
        return {
            "error": "Insufficient data for statistical comparison",
            "interpretation": "Need at least 2 test cases for comparison",
        }

    # Paired t-test (same test cases evaluated by both)
    t_statistic, p_value = stats.ttest_rel(results_a, results_b)

    # Interpretation
    significant = bool(p_value < 0.05)  # Convert numpy bool to Python bool
    mean_a = statistics.mean(results_a)
    mean_b = statistics.mean(results_b)
    better = impl_a_name if mean_a > mean_b else impl_b_name

    return {
        "t_statistic": round(float(t_statistic), 3),
        "p_value": round(float(p_value), 4),
        "significant": significant,
        "better_implementation": better if significant else "No significant difference",
        "interpretation": (
            f"{better} performs significantly better (p={p_value:.4f})"
            if significant
            else f"No statistically significant difference (p={p_value:.4f})"
        ),
    }
