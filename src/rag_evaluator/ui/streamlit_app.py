"""Streamlit web interface for RAG Evaluator."""

import json
import statistics
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_latest_report() -> dict[str, Any] | None:
    """Load the most recent evaluation report.

    Returns:
        Dictionary containing comparison results or None if no reports found
    """
    reports_dir = Path("reports")

    if not reports_dir.exists():
        return None

    # Find latest comparison report
    comparison_reports = sorted(
        reports_dir.glob("comparison_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if comparison_reports:
        with open(comparison_reports[0]) as f:
            return cast(dict[str, Any], json.load(f))

    # Fallback to single implementation report
    single_reports = sorted(
        reports_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if single_reports:
        with open(single_reports[0]) as f:
            report = cast(dict[str, Any], json.load(f))
            # Wrap single report in comparison format
            return {report["rag_implementation"]: report}

    return None


def render_overview_tab(report: dict[str, Any]) -> None:
    """Render Overview tab with summary statistics and charts.

    Args:
        report: Dictionary containing evaluation results for one or more implementations
    """
    st.title("RAG Evaluation - Overview")

    # Winner announcement
    if len(report) > 1:
        # Find best implementation by overall score
        best_impl = max(
            report.items(), key=lambda x: x[1]["metrics_summary"].get("faithfulness_avg", 0)
        )

        st.success(
            f"🏆 **Best Overall:** {best_impl[0]} "
            f"(Faithfulness: {best_impl[1]['metrics_summary']['faithfulness_avg']:.3f})"
        )

    # Summary statistics
    st.subheader("Summary Statistics")

    cols = st.columns(len(report))
    for idx, (impl_name, results) in enumerate(report.items()):
        with cols[idx]:
            st.metric(
                label=impl_name,
                value=f"{results['pass_rate']:.1f}%",
                delta=f"{results['test_cases_count']} tests",
            )

    # Metrics comparison bar chart
    st.subheader("Metrics Comparison")

    metrics_data = []
    for impl_name, results in report.items():
        metrics = results["metrics_summary"]
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        ]:
            metrics_data.append(
                {
                    "Implementation": impl_name,
                    "Metric": metric.replace("_", " ").title(),
                    "Score": metrics.get(f"{metric}_avg", 0),
                }
            )

    df = pd.DataFrame(metrics_data)

    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Implementation",
        barmode="group",
        title="Evaluation Metrics Comparison",
        labels={"Score": "Average Score"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Performance scatter plot (if cost data available)
    st.subheader("Accuracy vs Performance")

    # Create scatter: Accuracy (faithfulness) vs Latency
    scatter_data = []
    for impl_name, results in report.items():
        perf = results.get("performance_metrics", {})
        metrics = results["metrics_summary"]

        scatter_data.append(
            {
                "Implementation": impl_name,
                "Accuracy": metrics.get("faithfulness_avg", 0),
                "Latency (s)": perf.get("avg_retrieval_time", 0),
                "Queries": perf.get("total_queries", 0),
            }
        )

    df_scatter = pd.DataFrame(scatter_data)

    fig_scatter = px.scatter(
        df_scatter,
        x="Latency (s)",
        y="Accuracy",
        size="Queries",
        text="Implementation",
        title="Accuracy vs Latency Trade-off",
        labels={"Accuracy": "Faithfulness Score"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Key findings
    st.subheader("Key Findings")

    for impl_name, results in report.items():
        with st.expander(f"📊 {impl_name}"):
            metrics = results["metrics_summary"]

            st.markdown(
                f"""
            **Overall Performance:**
            - Pass Rate: {results["pass_rate"]:.1f}%
            - Faithfulness: {metrics.get("faithfulness_avg", 0):.3f}
            - Answer Relevancy: {metrics.get("answer_relevancy_avg", 0):.3f}
            - Contextual Precision: {metrics.get("contextual_precision_avg", 0):.3f}
            - Contextual Recall: {metrics.get("contextual_recall_avg", 0):.3f}
            """
            )


def render_comparison_tab(report: dict[str, Any]) -> None:
    """Render Detailed Comparison tab.

    Args:
        report: Dictionary containing evaluation results for one or more implementations
    """
    st.title("Detailed Comparison")

    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric to Analyze",
        ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Score distribution histograms
    st.subheader(f"{selected_metric.replace('_', ' ').title()} Score Distribution")

    fig = go.Figure()

    for impl_name, results in report.items():
        scores = [
            r["metrics"].get(selected_metric, 0)
            for r in results["detailed_results"]
            if r["metrics"].get(selected_metric) is not None
        ]

        fig.add_trace(go.Histogram(x=scores, name=impl_name, opacity=0.7, nbinsx=20))

    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"{selected_metric.replace('_', ' ').title()} Score",
        yaxis_title="Frequency",
        title=f"Distribution of {selected_metric.replace('_', ' ').title()} Scores",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Difficulty breakdown
    if len(report) > 1:
        st.subheader("Performance by Difficulty")

        difficulty_data = []
        for impl_name, results in report.items():
            for difficulty in ["easy", "medium", "hard"]:
                scores = [
                    r["metrics"].get(selected_metric, 0)
                    for r in results["detailed_results"]
                    if r.get("difficulty") == difficulty
                    and r["metrics"].get(selected_metric) is not None
                ]

                if scores:
                    difficulty_data.append(
                        {
                            "Implementation": impl_name,
                            "Difficulty": difficulty.capitalize(),
                            "Average Score": statistics.mean(scores),
                            "Count": len(scores),
                        }
                    )

        if difficulty_data:
            df_diff = pd.DataFrame(difficulty_data)

            fig_diff = px.bar(
                df_diff,
                x="Difficulty",
                y="Average Score",
                color="Implementation",
                barmode="group",
                title=f"{selected_metric.replace('_', ' ').title()} by Question Difficulty",
                text="Count",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_diff.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_diff, use_container_width=True)

    # Side-by-side comparison table
    st.subheader("Comparison Table")

    comparison_data = []
    for impl_name, results in report.items():
        metrics = results["metrics_summary"]
        perf = results.get("performance_metrics", {})

        comparison_data.append(
            {
                "Implementation": impl_name,
                "Pass Rate": f"{results['pass_rate']:.1f}%",
                "Faithfulness": f"{metrics.get('faithfulness_avg', 0):.3f}",
                "Relevancy": f"{metrics.get('answer_relevancy_avg', 0):.3f}",
                "Precision": f"{metrics.get('contextual_precision_avg', 0):.3f}",
                "Recall": f"{metrics.get('contextual_recall_avg', 0):.3f}",
                "Avg Latency": f"{perf.get('avg_retrieval_time', 0):.2f}s",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def render_query_explorer_tab(report: dict[str, Any]) -> None:
    """Render Query Explorer tab.

    Args:
        report: Dictionary containing evaluation results for one or more implementations
    """
    st.title("Query Explorer")

    # Get all test cases from first implementation
    first_impl = next(iter(report.values()))
    test_cases = first_impl["detailed_results"]

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        difficulty_filter = st.multiselect(
            "Difficulty",
            options=["easy", "medium", "hard"],
            default=["easy", "medium", "hard"],
        )

    with col2:
        # Get unique categories
        categories = list(set(tc.get("category", "unknown") for tc in test_cases))
        category_filter = st.multiselect("Category", options=categories, default=categories)

    with col3:
        # Score filter
        min_score = st.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # Filter test cases
    filtered_cases = [
        tc
        for tc in test_cases
        if tc.get("difficulty") in difficulty_filter
        and tc.get("category") in category_filter
        and tc["metrics"].get("faithfulness", 0) >= min_score
    ]

    st.write(f"Showing {len(filtered_cases)} of {len(test_cases)} test cases")

    # Question selector
    question_options = {
        f"{tc['test_case_id']}: {tc['question'][:60]}...": tc["test_case_id"]
        for tc in filtered_cases
    }

    if not question_options:
        st.warning("No test cases match the current filters")
        return

    selected_question_display = st.selectbox(
        "Select Question", options=list(question_options.keys())
    )

    selected_id = question_options[selected_question_display]

    # Find the selected test case in each implementation
    selected_cases = {}
    for impl_name, results in report.items():
        for tc in results["detailed_results"]:
            if tc["test_case_id"] == selected_id:
                selected_cases[impl_name] = tc
                break

    if not selected_cases:
        st.error("Test case not found")
        return

    # Display question details
    first_case = next(iter(selected_cases.values()))

    st.subheader("Question Details")
    st.markdown(f"**Question:** {first_case['question']}")
    st.markdown(f"**Difficulty:** {first_case.get('difficulty', 'unknown').capitalize()}")
    st.markdown(f"**Category:** {first_case.get('category', 'unknown')}")

    if first_case.get("expected_answer"):
        with st.expander("Ground Truth Answer"):
            st.write(first_case["expected_answer"])

    # Comparison table for this question
    st.subheader("Implementation Comparison")

    for impl_name, tc in selected_cases.items():
        with st.expander(f"📋 {impl_name}", expanded=True):
            metrics = tc["metrics"]

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}")
            with col2:
                st.metric("Relevancy", f"{metrics.get('answer_relevancy', 0):.3f}")
            with col3:
                st.metric("Precision", f"{metrics.get('contextual_precision', 0):.3f}")
            with col4:
                st.metric("Recall", f"{metrics.get('contextual_recall', 0):.3f}")

            # Answer
            st.markdown("**Answer:**")
            st.write(tc["answer"])

            # Context
            if tc.get("context_chunks_retrieved", 0) > 0:
                with st.expander("Retrieved Context"):
                    st.write(f"Retrieved {tc['context_chunks_retrieved']} chunks")

            # Performance
            retrieval_time = tc.get("retrieval_time", 0)
            st.caption(f"Retrieval time: {retrieval_time:.3f}s")


def apply_custom_css() -> None:
    """Apply custom CSS for better styling."""
    st.markdown(
        """
    <style>
    /* Color-coded scores */
    .metric-high {
        color: #28a745;
        font-weight: bold;
    }
    .metric-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .metric-low {
        color: #dc3545;
        font-weight: bold;
    }

    /* Better table styling */
    .dataframe {
        font-size: 14px;
    }

    /* Cleaner expanders */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="RAG Evaluator", page_icon="🔍", layout="wide")

    apply_custom_css()

    # Load report
    report = load_latest_report()

    if not report:
        st.error("No evaluation reports found. Please run an evaluation first:")
        st.code("uv run rag-eval evaluate --rag-type all")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Detailed Comparison", "🔍 Query Explorer"])

    with tab1:
        render_overview_tab(report)

    with tab2:
        render_comparison_tab(report)

    with tab3:
        render_query_explorer_tab(report)


if __name__ == "__main__":
    main()
