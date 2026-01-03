"""Difficulty analysis module for evaluating performance by question difficulty."""

import statistics
from collections import defaultdict
from typing import Any


def analyze_by_difficulty(
    detailed_results: list[dict[str, Any]], metric_name: str = "faithfulness"
) -> dict[str, dict[str, float]]:
    """Group results by difficulty and calculate average scores.

    Args:
        detailed_results: List of test case results with metrics and difficulty
        metric_name: Name of the metric to analyze (default: "faithfulness")

    Returns:
        Dictionary mapping difficulty levels to statistics:
        {
            "easy": {"mean": 0.85, "count": 30, "std": 0.08, "min": 0.70, "max": 0.95},
            "medium": {"mean": 0.75, "count": 50, "std": 0.12, "min": 0.55, "max": 0.90},
            "hard": {"mean": 0.62, "count": 20, "std": 0.15, "min": 0.40, "max": 0.85}
        }
    """
    difficulty_groups: dict[str, list[float]] = defaultdict(list)

    for result in detailed_results:
        difficulty = result.get("difficulty", "unknown")
        score = result.get("metrics", {}).get(metric_name)
        if score is not None:
            difficulty_groups[difficulty].append(score)

    analysis: dict[str, dict[str, float]] = {}
    for difficulty, scores in difficulty_groups.items():
        if scores:
            analysis[difficulty] = {
                "mean": round(statistics.mean(scores), 3),
                "count": len(scores),
                "std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
            }

    return analysis


def compare_difficulty_performance(
    comparison_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compare all implementations across difficulty levels.

    Args:
        comparison_results: Dictionary mapping implementation names to their results

    Returns:
        Dictionary mapping difficulty levels to implementation scores:
        {
            "easy": {"Vector": 0.85, "Hybrid": 0.87, ...},
            "medium": {"Vector": 0.75, "Hybrid": 0.78, ...},
            "hard": {"Vector": 0.62, "Hybrid": 0.68, ...}
        }
    """
    difficulty_comparison: dict[str, dict[str, float]] = defaultdict(dict)

    for impl_name, results in comparison_results.items():
        difficulty_analysis = analyze_by_difficulty(
            results["detailed_results"], metric_name="faithfulness"
        )

        for difficulty, stats in difficulty_analysis.items():
            difficulty_comparison[difficulty][impl_name] = stats["mean"]

    return dict(difficulty_comparison)
