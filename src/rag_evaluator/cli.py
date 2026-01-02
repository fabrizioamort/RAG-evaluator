"""CLI entry point for RAG Evaluator."""

import argparse
import sys
from pathlib import Path

from rag_evaluator.config import settings
from rag_evaluator.evaluation.evaluator import RAGEvaluator
from rag_evaluator.evaluation.report_generator import ReportGenerator
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


def get_rag_implementation(rag_type: str) -> ChromaSemanticRAG:
    """Get RAG implementation instance by type.

    Args:
        rag_type: Type of RAG to instantiate

    Returns:
        RAG implementation instance

    Raises:
        ValueError: If rag_type is not supported
    """
    if rag_type == "vector_semantic":
        return ChromaSemanticRAG()
    else:
        raise ValueError(
            f"RAG type '{rag_type}' not yet implemented. Currently supported: vector_semantic"
        )


def cmd_prepare(args: argparse.Namespace) -> int:
    """Handle prepare command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print(f"Preparing documents from: {args.input_dir}")
    print(f"RAG type: {args.rag_type}")

    try:
        # Check if input directory exists
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"❌ Error: Input directory not found: {args.input_dir}")
            return 1

        # Get RAG implementation
        rag_impl = get_rag_implementation(args.rag_type)

        # Prepare documents
        print(f"\n📄 Preparing documents with {rag_impl.name}...")
        rag_impl.prepare_documents(str(input_dir))

        print("✅ Documents prepared successfully!")
        return 0

    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error preparing documents: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Handle evaluate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print(f"Evaluating: {args.rag_type}")
    print(f"Test set: {args.test_set}")
    print(f"Output directory: {args.output}")

    try:
        # Check if test set exists
        test_set_path = Path(args.test_set)
        if not test_set_path.exists():
            print(f"❌ Error: Test set not found: {args.test_set}")
            print(
                f"\nCreate a test set at {args.test_set} or use --test-set to specify a different path"
            )
            return 1

        # Initialize evaluator
        print("\n🔧 Loading test set...")
        evaluator = RAGEvaluator(test_set_path=str(test_set_path))
        print(f"✅ Loaded {len(evaluator.test_cases)} test cases")

        # Handle 'all' option
        if args.rag_type == "all":
            print("\n⚠️  'all' option not yet fully implemented")
            print("Currently evaluating: vector_semantic")
            rag_types = ["vector_semantic"]
        else:
            rag_types = [args.rag_type]

        # Evaluate each RAG type
        all_results = {}
        for rag_type in rag_types:
            print(f"\n{'=' * 70}")
            print(f"Evaluating: {rag_type}")
            print(f"{'=' * 70}")

            # Get RAG implementation
            rag_impl = get_rag_implementation(rag_type)

            # Run evaluation
            print("\n🚀 Running evaluation...")
            results = evaluator.evaluate(rag_impl, verbose=args.verbose)
            all_results[rag_impl.name] = results

            # Print summary
            print("\n📊 Results:")
            print(f"  Pass Rate: {results['pass_rate']:.1f}%")
            print(f"  Test Cases: {results['test_cases_count']}")
            print(f"  Time: {results['total_evaluation_time']:.2f}s")

        # Generate reports
        print(f"\n📝 Generating reports in: {args.output}")
        report_gen = ReportGenerator(output_dir=args.output)

        if len(all_results) == 1:
            # Single implementation report
            results = next(iter(all_results.values()))
            files = report_gen.generate_report(results, output_format="both")
            print("\n✅ Reports generated:")
            print(f"  • JSON: {files['json']}")
            print(f"  • Markdown: {files['markdown']}")
        else:
            # Comparison report
            files = report_gen.generate_comparison_report(all_results, output_format="both")
            print("\n✅ Comparison reports generated:")
            print(f"  • JSON: {files['json']}")
            print(f"  • Markdown: {files['markdown']}")

        print("\n" + "=" * 70)
        print("✅ Evaluation completed successfully!")
        print("=" * 70)

        return 0

    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_ui(args: argparse.Namespace) -> int:
    """Handle ui command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("Launching Streamlit UI...")
    print("(Full UI implementation coming soon)")
    print("\nFor now, you can run the evaluation script:")
    print("  uv run python scripts/run_evaluation.py --help")
    return 0


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluator - Compare different RAG implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare documents for ChromaDB RAG
  rag-eval prepare --rag-type vector_semantic --input-dir data/raw

  # Evaluate ChromaDB RAG
  rag-eval evaluate --rag-type vector_semantic

  # Evaluate with custom test set
  rag-eval evaluate --rag-type vector_semantic --test-set my_tests.json

  # Verbose evaluation output
  rag-eval evaluate --rag-type vector_semantic --verbose
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare documents for RAG implementations",
        description="Index and prepare documents for a RAG implementation",
    )
    prepare_parser.add_argument(
        "--rag-type",
        choices=["vector_semantic"],
        default="vector_semantic",
        help="RAG implementation type",
    )
    prepare_parser.add_argument(
        "--input-dir",
        default=settings.raw_data_dir,
        help=f"Directory containing source documents (default: {settings.raw_data_dir})",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on RAG implementations",
        description="Evaluate RAG implementation(s) using DeepEval metrics",
    )
    eval_parser.add_argument(
        "--rag-type",
        choices=["vector_semantic", "all"],
        default="vector_semantic",
        help="RAG implementation to evaluate",
    )
    eval_parser.add_argument(
        "--test-set",
        default=settings.eval_test_set_path,
        help=f"Path to test set JSON file (default: {settings.eval_test_set_path})",
    )
    eval_parser.add_argument(
        "--output",
        default=settings.eval_reports_dir,
        help=f"Directory for evaluation reports (default: {settings.eval_reports_dir})",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation progress",
    )

    # Web UI command
    subparsers.add_parser(
        "ui",
        help="Launch Streamlit web interface",
        description="Launch interactive web UI for RAG evaluation (coming soon)",
    )

    args = parser.parse_args()

    # Route to appropriate command handler
    if args.command == "prepare":
        sys.exit(cmd_prepare(args))
    elif args.command == "evaluate":
        sys.exit(cmd_evaluate(args))
    elif args.command == "ui":
        sys.exit(cmd_ui(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
