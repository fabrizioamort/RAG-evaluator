"""Report generation for RAG evaluation results."""

import json
from pathlib import Path
from typing import Any

from rag_evaluator.evaluation.difficulty_analysis import (
    analyze_by_difficulty,
    compare_difficulty_performance,
)
from rag_evaluator.evaluation.statistics import (
    calculate_statistics,
    compare_implementations_statistically,
)


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def __init__(self, output_dir: str = "reports") -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self, evaluation_results: dict[str, Any], output_format: str = "both"
    ) -> dict[str, str]:
        """Generate evaluation report.

        Args:
            evaluation_results: Results from RAGEvaluator.evaluate()
            output_format: Format to generate ('json', 'markdown', or 'both')

        Returns:
            Dictionary with paths to generated report files

        Raises:
            ValueError: If output_format is invalid
        """
        if output_format not in ["json", "markdown", "both"]:
            raise ValueError(
                f"Invalid format: {output_format}. Must be 'json', 'markdown', or 'both'"
            )

        # Generate filename based on implementation name and timestamp
        impl_name = evaluation_results["rag_implementation"].replace(" ", "_").lower()
        timestamp = evaluation_results["timestamp"].replace(" ", "_").replace(":", "")
        base_filename = f"eval_{impl_name}_{timestamp}"

        generated_files = {}

        if output_format in ["json", "both"]:
            json_path = self._generate_json_report(evaluation_results, base_filename)
            generated_files["json"] = str(json_path)

        if output_format in ["markdown", "both"]:
            md_path = self._generate_markdown_report(evaluation_results, base_filename)
            generated_files["markdown"] = str(md_path)

        return generated_files

    def generate_comparison_report(
        self, comparison_results: dict[str, dict[str, Any]], output_format: str = "both"
    ) -> dict[str, str]:
        """Generate comparison report for multiple implementations.

        Args:
            comparison_results: Results from RAGEvaluator.compare_implementations()
            output_format: Format to generate ('json', 'markdown', or 'both')

        Returns:
            Dictionary with paths to generated report files
        """
        if output_format not in ["json", "markdown", "both"]:
            raise ValueError(
                f"Invalid format: {output_format}. Must be 'json', 'markdown', or 'both'"
            )

        # Use first implementation's timestamp for filename
        first_impl = next(iter(comparison_results.values()))
        timestamp = first_impl["timestamp"].replace(" ", "_").replace(":", "")
        base_filename = f"comparison_{timestamp}"

        generated_files = {}

        if output_format in ["json", "both"]:
            json_path = self._generate_comparison_json(comparison_results, base_filename)
            generated_files["json"] = str(json_path)

        if output_format in ["markdown", "both"]:
            md_path = self._generate_comparison_markdown(comparison_results, base_filename)
            generated_files["markdown"] = str(md_path)

        return generated_files

    def generate_statistical_analysis_section(
        self, evaluation_results: dict[str, Any]
    ) -> list[str]:
        """Generate statistical analysis section with confidence intervals.

        Args:
            evaluation_results: Evaluation results dictionary

        Returns:
            List of markdown lines for the statistical analysis section
        """
        lines = ["## Statistical Analysis", ""]

        # For each metric, show statistical summary
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        ]:
            # Extract individual scores from detailed_results
            scores = [
                r["metrics"].get(metric)
                for r in evaluation_results["detailed_results"]
                if r["metrics"].get(metric) is not None
            ]

            if scores:
                stats = calculate_statistics(scores)
                metric_name = metric.replace("_", " ").title()

                lines.extend(
                    [
                        f"### {metric_name}",
                        "",
                        f"- **Mean:** {stats.mean} (95% CI: [{stats.confidence_interval_95[0]}, {stats.confidence_interval_95[1]}])",
                        f"- **Median:** {stats.median}",
                        f"- **Std Dev:** {stats.std_dev}",
                        f"- **Range:** [{stats.min}, {stats.max}]",
                        "",
                    ]
                )

        return lines

    def generate_difficulty_breakdown_section(
        self, evaluation_results: dict[str, Any]
    ) -> list[str]:
        """Generate difficulty breakdown section.

        Args:
            evaluation_results: Evaluation results dictionary

        Returns:
            List of markdown lines for the difficulty breakdown section
        """
        lines = ["## Performance by Difficulty", ""]

        # Analyze for each metric
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        ]:
            difficulty_analysis = analyze_by_difficulty(
                evaluation_results["detailed_results"], metric_name=metric
            )

            if difficulty_analysis:
                metric_name = metric.replace("_", " ").title()
                lines.extend([f"### {metric_name}", ""])

                # Create table
                lines.append("| Difficulty | Mean | Count | Std Dev | Range |")
                lines.append("|------------|------|-------|---------|-------|")

                for difficulty in ["easy", "medium", "hard"]:
                    if difficulty in difficulty_analysis:
                        stats = difficulty_analysis[difficulty]
                        lines.append(
                            f"| {difficulty.capitalize()} | "
                            f"{stats['mean']:.3f} | "
                            f"{stats['count']} | "
                            f"{stats['std']:.3f} | "
                            f"[{stats['min']:.3f}, {stats['max']:.3f}] |"
                        )

                lines.append("")

        return lines

    def generate_failure_analysis_section(self, evaluation_results: dict[str, Any]) -> list[str]:
        """Analyze questions where performance was poor.

        Args:
            evaluation_results: Evaluation results dictionary

        Returns:
            List of markdown lines for the failure analysis section
        """
        lines = ["## Failure Analysis", ""]

        # Find questions with low scores
        low_score_threshold = 0.5
        failures = [
            r
            for r in evaluation_results["detailed_results"]
            if any(
                r["metrics"].get(metric, 1.0) < low_score_threshold
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "contextual_precision",
                    "contextual_recall",
                ]
            )
        ]

        if not failures:
            lines.append("_No significant failures detected (all scores >0.5)_")
            return lines

        lines.append(f"Found {len(failures)} test cases with scores below {low_score_threshold}:")
        lines.append("")

        for failure in failures[:5]:  # Show top 5 failures
            lines.extend(
                [
                    f"### {failure['test_case_id']}: {failure['question']}",
                    "",
                    f"**Difficulty:** {failure.get('difficulty', 'unknown')}  ",
                    f"**Category:** {failure.get('category', 'unknown')}  ",
                    "",
                ]
            )

            # Show which metrics failed
            failed_metrics = [
                (metric, failure["metrics"].get(metric, 0.0))
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "contextual_precision",
                    "contextual_recall",
                ]
                if failure["metrics"].get(metric, 1.0) < low_score_threshold
            ]

            lines.append("**Failed Metrics:**")
            for metric, score in failed_metrics:
                lines.append(f"- {metric.replace('_', ' ').title()}: {score:.3f}")

            lines.append("")

        return lines

    def generate_comparison_statistical_section(
        self, comparison_results: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Generate statistical comparison between implementations.

        Args:
            comparison_results: Comparison results dictionary

        Returns:
            List of markdown lines for the statistical comparison section
        """
        lines = ["## Statistical Comparison", ""]

        # Compare implementations pairwise for each metric
        impl_names = list(comparison_results.keys())

        if len(impl_names) < 2:
            lines.append("_Need at least 2 implementations for statistical comparison_")
            return lines

        # For faithfulness metric (can extend to others)
        metric = "faithfulness"
        lines.extend([f"### {metric.replace('_', ' ').title()} Comparisons", ""])

        # Extract scores for each implementation
        impl_scores: dict[str, list[float]] = {}
        for impl_name, results in comparison_results.items():
            scores = [
                r["metrics"].get(metric)
                for r in results["detailed_results"]
                if r["metrics"].get(metric) is not None
            ]
            impl_scores[impl_name] = scores

        # Pairwise comparisons
        for i, impl_a in enumerate(impl_names):
            for impl_b in impl_names[i + 1 :]:
                comparison = compare_implementations_statistically(
                    impl_scores[impl_a], impl_scores[impl_b], impl_a, impl_b
                )

                lines.append(f"**{impl_a} vs {impl_b}:** {comparison['interpretation']}")

        lines.append("")
        return lines

    def _generate_json_report(self, evaluation_results: dict[str, Any], base_filename: str) -> Path:
        """Generate JSON report.

        Args:
            evaluation_results: Evaluation results
            base_filename: Base filename without extension

        Returns:
            Path to generated JSON file
        """
        json_path = self.output_dir / f"{base_filename}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        return json_path

    def _generate_markdown_report(
        self, evaluation_results: dict[str, Any], base_filename: str
    ) -> Path:
        """Generate Markdown report.

        Args:
            evaluation_results: Evaluation results
            base_filename: Base filename without extension

        Returns:
            Path to generated Markdown file
        """
        md_path = self.output_dir / f"{base_filename}.md"

        # Build markdown content
        lines = [
            "# RAG Evaluation Report",
            "",
            f"**Implementation:** {evaluation_results['rag_implementation']}  ",
            f"**Timestamp:** {evaluation_results['timestamp']}  ",
            f"**Test Cases:** {evaluation_results['test_cases_count']}  ",
            f"**Evaluation Time:** {evaluation_results['total_evaluation_time']}s  ",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"**Overall Pass Rate:** {evaluation_results['pass_rate']}%",
            "",
            "### Metrics Thresholds",
            "",
            "| Metric | Threshold |",
            "|--------|-----------|",
        ]

        # Add thresholds table
        for metric, threshold in evaluation_results["thresholds"].items():
            lines.append(f"| {metric.replace('_', ' ').title()} | {threshold} |")

        lines.extend(["", "### Metric Scores", ""])

        # Create metrics summary table
        metrics_summary = evaluation_results["metrics_summary"]
        lines.extend(
            [
                "| Metric | Average | Min | Max |",
                "|--------|---------|-----|-----|",
            ]
        )

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        ]:
            avg = metrics_summary.get(f"{metric}_avg", 0.0)
            min_val = metrics_summary.get(f"{metric}_min", 0.0)
            max_val = metrics_summary.get(f"{metric}_max", 0.0)
            metric_name = metric.replace("_", " ").title()
            lines.append(f"| {metric_name} | {avg:.3f} | {min_val:.3f} | {max_val:.3f} |")

        # Add performance metrics
        lines.extend(["", "### Performance Metrics", ""])
        perf_metrics = evaluation_results["performance_metrics"]

        if perf_metrics:
            lines.extend(
                [
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )
            for key, value in perf_metrics.items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    lines.append(f"| {formatted_key} | {value:.3f} |")
                else:
                    lines.append(f"| {formatted_key} | {value} |")

        # Add new sections
        lines.extend(["", "---", ""])
        lines.extend(self.generate_statistical_analysis_section(evaluation_results))
        lines.extend(["", "---", ""])
        lines.extend(self.generate_difficulty_breakdown_section(evaluation_results))
        lines.extend(["", "---", ""])
        lines.extend(self.generate_failure_analysis_section(evaluation_results))

        # Add detailed results
        lines.extend(["", "---", "", "## Detailed Test Case Results", ""])

        for result in evaluation_results["detailed_results"]:
            lines.extend(
                [
                    f"### {result['test_case_id']} - {result['category'].title()}",
                    "",
                    f"**Difficulty:** {result['difficulty'].title()}  ",
                    f"**Question:** {result['question']}  ",
                    "",
                    "**Metrics:**",
                    "",
                ]
            )

            # Add metrics for this test case
            metrics = result.get("metrics", {})
            for metric_name in [
                "faithfulness",
                "answer_relevancy",
                "contextual_precision",
                "contextual_recall",
            ]:
                score = metrics.get(metric_name)
                if score is not None:
                    threshold = evaluation_results["thresholds"][metric_name]
                    status = "✅ PASS" if score >= threshold else "❌ FAIL"
                    formatted_name = metric_name.replace("_", " ").title()
                    lines.append(f"- **{formatted_name}:** {score:.3f} {status}")
                else:
                    formatted_name = metric_name.replace("_", " ").title()
                    lines.append(f"- **{formatted_name}:** N/A")

            lines.extend(
                [
                    "",
                    "**Performance:**",
                    "",
                    f"- Retrieval Time: {result['retrieval_time']:.3f}s",
                    f"- Chunks Retrieved: {result['context_chunks_retrieved']}",
                    "",
                    "**Answer:**",
                    "",
                    f"> {result['answer']}",
                    "",
                ]
            )

        # Write to file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return md_path

    def _generate_comparison_json(
        self, comparison_results: dict[str, dict[str, Any]], base_filename: str
    ) -> Path:
        """Generate comparison JSON report.

        Args:
            comparison_results: Comparison results
            base_filename: Base filename without extension

        Returns:
            Path to generated JSON file
        """
        json_path = self.output_dir / f"{base_filename}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)

        return json_path

    def _generate_comparison_markdown(
        self, comparison_results: dict[str, dict[str, Any]], base_filename: str
    ) -> Path:
        """Generate comparison Markdown report.

        Args:
            comparison_results: Comparison results
            base_filename: Base filename without extension

        Returns:
            Path to generated Markdown file
        """
        md_path = self.output_dir / f"{base_filename}.md"

        # Build markdown content
        lines = [
            "# RAG Implementations Comparison Report",
            "",
            f"**Number of Implementations:** {len(comparison_results)}  ",
            f"**Timestamp:** {next(iter(comparison_results.values()))['timestamp']}  ",
            "",
            "---",
            "",
            "## Summary Comparison",
            "",
            "| Implementation | Pass Rate | Faithfulness | Answer Relevancy | Ctx Precision | Ctx Recall |",
            "|----------------|-----------|--------------|------------------|---------------|------------|",
        ]

        # Add summary row for each implementation
        for impl_name, results in comparison_results.items():
            metrics = results["metrics_summary"]
            pass_rate = results["pass_rate"]
            lines.append(
                f"| {impl_name} | {pass_rate}% | "
                f"{metrics.get('faithfulness_avg', 0.0):.3f} | "
                f"{metrics.get('answer_relevancy_avg', 0.0):.3f} | "
                f"{metrics.get('contextual_precision_avg', 0.0):.3f} | "
                f"{metrics.get('contextual_recall_avg', 0.0):.3f} |"
            )

        # Add performance comparison
        lines.extend(["", "## Performance Comparison", ""])

        # Collect performance metrics
        perf_keys = set()
        for results in comparison_results.values():
            perf_keys.update(results["performance_metrics"].keys())

        if perf_keys:
            header = (
                "| Implementation | "
                + " | ".join(k.replace("_", " ").title() for k in sorted(perf_keys))
                + " |"
            )
            separator = "|" + "|".join(["---"] * (len(perf_keys) + 1)) + "|"
            lines.extend([header, separator])

            for impl_name, results in comparison_results.items():
                perf = results["performance_metrics"]
                values = []
                for key in sorted(perf_keys):
                    value = perf.get(key, "N/A")
                    if isinstance(value, float):
                        values.append(f"{value:.3f}")
                    else:
                        values.append(str(value))
                row = f"| {impl_name} | " + " | ".join(values) + " |"
                lines.append(row)

        # Add difficulty breakdown comparison
        lines.extend(["", "---", "", "## Performance by Difficulty", ""])

        difficulty_comparison = compare_difficulty_performance(comparison_results)

        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in difficulty_comparison:
                lines.extend([f"### {difficulty.capitalize()} Questions", ""])

                impl_scores = difficulty_comparison[difficulty]
                sorted_impls = sorted(impl_scores.items(), key=lambda x: x[1], reverse=True)

                for impl_name, score in sorted_impls:
                    lines.append(f"- **{impl_name}:** {score:.3f}")

                lines.append("")

        # Add statistical comparison section
        lines.extend(["", "---", ""])
        lines.extend(self.generate_comparison_statistical_section(comparison_results))

        # Add individual implementation details
        lines.extend(["", "---", "", "## Individual Implementation Details", ""])

        for impl_name, results in comparison_results.items():
            lines.extend(
                [
                    f"### {impl_name}",
                    "",
                    f"**Test Cases:** {results['test_cases_count']}  ",
                    f"**Pass Rate:** {results['pass_rate']}%  ",
                    f"**Evaluation Time:** {results['total_evaluation_time']}s  ",
                    "",
                ]
            )

            # Add metrics for this implementation
            metrics = results["metrics_summary"]
            lines.extend(["**Metrics:**", ""])
            for metric in [
                "faithfulness",
                "answer_relevancy",
                "contextual_precision",
                "contextual_recall",
            ]:
                avg = metrics.get(f"{metric}_avg", 0.0)
                metric_name = metric.replace("_", " ").title()
                lines.append(f"- **{metric_name}:** {avg:.3f}")

            lines.append("")

        # Write to file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return md_path
