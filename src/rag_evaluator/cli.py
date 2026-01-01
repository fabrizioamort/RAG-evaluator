"""CLI entry point for RAG Evaluator."""

import argparse


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluator - Compare different RAG implementations"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare documents for RAG implementations"
    )
    prepare_parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing source documents",
    )
    prepare_parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed documents",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on RAG implementations")
    eval_parser.add_argument(
        "--rag-type",
        choices=["vector_semantic", "vector_hybrid", "graph_rag", "filesystem_rag", "all"],
        default="all",
        help="RAG implementation to evaluate",
    )
    eval_parser.add_argument(
        "--output",
        default="reports",
        help="Directory for evaluation reports",
    )

    # Web UI command
    subparsers.add_parser("ui", help="Launch Streamlit web interface")

    args = parser.parse_args()

    if args.command == "prepare":
        print(f"Preparing documents from {args.input_dir} to {args.output_dir}")
        print("(Implementation coming soon)")
    elif args.command == "evaluate":
        print(f"Evaluating {args.rag_type} RAG implementation(s)")
        print(f"Results will be saved to {args.output}")
        print("(Implementation coming soon)")
    elif args.command == "ui":
        print("Launching Streamlit UI...")
        print("(Implementation coming soon)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
