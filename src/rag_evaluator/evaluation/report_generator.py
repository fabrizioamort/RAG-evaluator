"""Report generation for RAG evaluation results."""

import json
from pathlib import Path
from typing import Any


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
