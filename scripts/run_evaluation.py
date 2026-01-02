"""Script to run RAG evaluation with detailed reporting.

This script evaluates a RAG implementation using the test dataset and generates
comprehensive reports in JSON and Markdown formats.

Usage:
    uv run python scripts/run_evaluation.py --rag-type vector_semantic
    uv run python scripts/run_evaluation.py --rag-type vector_semantic --verbose
"""

import argparse
import sys
from pathlib import Path

from rag_evaluator.config import settings
from rag_evaluator.evaluation.evaluator import RAGEvaluator
from rag_evaluator.evaluation.report_generator import ReportGenerator
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


def get_rag_implementation(rag_type: str):
    """Get RAG implementation by type.

    Args:
        rag_type: Type of RAG implementation to use

    Returns:
        RAG implementation instance

    Raises:
        ValueError: If rag_type is not supported
    """
    if rag_type == "vector_semantic":
        return ChromaSemanticRAG()
    else:
        raise ValueError(f"Unsupported RAG type: {rag_type}. Currently supported: vector_semantic")


def main() -> int:
    """Run RAG evaluation.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RAG implementation using DeepEval metrics"
    )
    parser.add_argument(
        "--rag-type",
        type=str,
        default="vector_semantic",
        choices=["vector_semantic"],
        help="Type of RAG implementation to evaluate",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help=f"Path to test set JSON file (default: {settings.eval_test_set_path})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for output reports (default: {settings.eval_reports_dir})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--prepare-documents",
        action="store_true",
        help="Prepare documents before evaluation",
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help=f"Directory containing documents to prepare (default: {settings.raw_data_dir})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RAG Evaluation Script")
    print("=" * 70)
    print(f"RAG Type: {args.rag_type}")
    print(f"Test Set: {args.test_set or settings.eval_test_set_path}")
    print(f"Output Dir: {args.output_dir or settings.eval_reports_dir}")
    print("=" * 70)

    try:
        # Check if test set exists
        test_set_path = args.test_set or settings.eval_test_set_path
        if not Path(test_set_path).exists():
            print(f"\n❌ Error: Test set file not found: {test_set_path}")
            print(
                "\nPlease ensure the test set file exists or specify a different path with --test-set"
            )
            return 1

        # Initialize RAG implementation
        print(f"\n📦 Initializing {args.rag_type} RAG implementation...")
        rag_impl = get_rag_implementation(args.rag_type)

        # Prepare documents if requested
        if args.prepare_documents:
            documents_dir = args.documents_dir or settings.raw_data_dir
            print(f"\n📄 Preparing documents from: {documents_dir}")
            if not Path(documents_dir).exists():
                print(f"❌ Error: Documents directory not found: {documents_dir}")
                return 1

            rag_impl.prepare_documents(documents_dir)
            print("✅ Documents prepared successfully")

        # Initialize evaluator
        print(f"\n🔧 Initializing evaluator with test set: {test_set_path}")
        evaluator = RAGEvaluator(test_set_path=test_set_path)
        print(f"✅ Loaded {len(evaluator.test_cases)} test cases")

        # Run evaluation
        print("\n🚀 Starting evaluation...")
        print("This may take several minutes depending on the number of test cases...")
        print()

        results = evaluator.evaluate(rag_impl, verbose=args.verbose)

        # Print summary
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Implementation: {results['rag_implementation']}")
        print(f"Test Cases: {results['test_cases_count']}")
        print(f"Pass Rate: {results['pass_rate']:.1f}%")
        print(f"Total Time: {results['total_evaluation_time']:.2f}s")
        print()

        # Print metrics summary
        print("Metrics:")
        metrics = results["metrics_summary"]
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
            "hallucination",
        ]:
            avg_key = f"{metric_name}_avg"
            if avg_key in metrics:
                threshold = results["thresholds"][metric_name]
                avg_value = metrics[avg_key]
                status = "✅" if avg_value >= threshold else "⚠️"
                formatted_name = metric_name.replace("_", " ").title()
                print(f"  {status} {formatted_name}: {avg_value:.3f} (threshold: {threshold})")

        print()
        print("Performance:")
        perf_metrics = results["performance_metrics"]
        for key, value in perf_metrics.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, float):
                print(f"  • {formatted_key}: {value:.3f}")
            else:
                print(f"  • {formatted_key}: {value}")

        # Generate reports
        output_dir = args.output_dir or settings.eval_reports_dir
        print(f"\n📝 Generating reports in: {output_dir}")

        report_gen = ReportGenerator(output_dir=output_dir)
        files = report_gen.generate_report(results, output_format="both")

        print("\n✅ Reports generated successfully:")
        print(f"  • JSON: {files['json']}")
        print(f"  • Markdown: {files['markdown']}")

        print("\n" + "=" * 70)
        print("✅ Evaluation completed successfully!")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
