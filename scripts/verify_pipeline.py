#!/usr/bin/env python
"""Interactive pipeline verification script for RAG Evaluator.

Smoke-tests one RAG type at a time using a tiny dataset (mini corpus +
5 questions) with confirmation prompts between every destructive or
expensive step.

Usage:
    uv run python scripts/verify_pipeline.py --rag-type vector_semantic
    uv run python scripts/verify_pipeline.py --rag-type vector_hybrid
    uv run python scripts/verify_pipeline.py --rag-type graph_rag
    uv run python scripts/verify_pipeline.py --rag-type filesystem_rag

Flags:
    --skip-cleanup     Skip cleanup step (DB already clean)
    --skip-prepare     Skip prepare step (already indexed)
    --cleanup-only     Only run cleanup, then exit
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# -- paths (relative to project root) ----------------------------------------
ROOT = Path(__file__).parent.parent
FULL_TEST_SET = ROOT / "data" / "test_set_multihop.json"
MINI_CORPUS_DIR = ROOT / "data" / "raw" / "multihop_verify"
MINI_TEST_SET = ROOT / "data" / "test_set_verify.json"
CHROMA_DIR = ROOT / "data" / "chroma_db"
FILESYSTEM_PREPARED_DIR = ROOT / "data" / "prepared" / "filesystem_rag"

QDRANT_URL = "http://localhost:6333"
NEO4J_URI = "bolt://localhost:7687"

OPENAI_USAGE_URL = "https://platform.openai.com/usage"
PASS_THRESHOLD = 0.60   # 60% = 3 of 5 questions


# -- helpers ------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    """Ask yes/no question. Returns True for 'y', exits on 'n' or Ctrl+C."""
    while True:
        try:
            answer = input(f"\n{prompt} [y/n]: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted by user.")
            sys.exit(0)
        if answer == "y":
            return True
        if answer == "n":
            print("Skipping. Exiting.")
            sys.exit(0)


def pause(message: str) -> None:
    """Pause and wait for Enter."""
    try:
        input(f"\n{message}\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def build_mini_dataset() -> None:
    """Create mini corpus + test set from full MultiHop-RAG data."""
    if MINI_CORPUS_DIR.exists() and MINI_TEST_SET.exists():
        print("  Mini dataset already exists, skipping creation.")
        print(f"  Corpus:   {MINI_CORPUS_DIR}/ ({len(list(MINI_CORPUS_DIR.glob('*.txt')))} files)")
        print(f"  Test set: {MINI_TEST_SET} ({len(json.loads(MINI_TEST_SET.read_text(encoding='utf-8'))['test_cases'])} questions)")
        return

    if not FULL_TEST_SET.exists():
        print(f"  ERROR: Full test set not found: {FULL_TEST_SET}")
        print("  Run first:  uv run python scripts/load_multihop_rag.py --max-questions 100 --exclude-null")
        sys.exit(1)

    print("  Building mini dataset from full MultiHop-RAG data...")
    full = json.loads(FULL_TEST_SET.read_text(encoding="utf-8"))
    all_cases = full["test_cases"]

    # Pick first 5 cases that have resolvable evidence sources
    selected = []
    for case in all_cases:
        sources = [Path(p) for p in case.get("evidence_sources", []) if Path(p).exists()]
        if sources:
            case["_resolved_sources"] = sources
            selected.append(case)
        if len(selected) == 5:
            break

    if len(selected) < 5:
        print(f"  WARNING: Only found {len(selected)} questions with resolvable evidence sources.")
        if not selected:
            print("  No valid questions found. Check that data/raw/multihop_rag/ exists.")
            sys.exit(1)

    # Copy referenced articles to mini corpus dir
    MINI_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    all_sources: set[Path] = set()
    for case in selected:
        all_sources.update(case.pop("_resolved_sources"))

    for src in sorted(all_sources):
        shutil.copy2(src, MINI_CORPUS_DIR / src.name)

    # Write mini test set
    mini = {
        "metadata": {
            "description": "Mini verification test set (5 questions) from MultiHop-RAG",
            "source": str(FULL_TEST_SET),
            "corpus_directory": str(MINI_CORPUS_DIR),
        },
        "test_cases": selected,
    }
    MINI_TEST_SET.write_text(json.dumps(mini, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Created corpus:   {MINI_CORPUS_DIR}/ ({len(all_sources)} articles)")
    print(f"  Created test set: {MINI_TEST_SET} (5 questions)")


def check_qdrant_reachable() -> bool:
    """Return True if Qdrant HTTP API responds."""
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/healthz")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def check_neo4j_reachable() -> bool:
    """Return True if Neo4j Bolt port is open."""
    import socket
    try:
        host = NEO4J_URI.replace("bolt://", "").split(":")[0]
        port = int(NEO4J_URI.split(":")[-1])
        with socket.create_connection((host, port), timeout=5):
            return True
    except Exception:
        return False


def preflight_check(rag_type: str) -> None:
    """Run pre-flight checks for a RAG type. Exits if checks fail."""
    section("PRE-FLIGHT CHECK")

    if rag_type == "vector_semantic":
        print("  ChromaDB runs in-process - no Docker needed.")
        print("  OK")

    elif rag_type == "vector_hybrid":
        print("  Qdrant requires Docker.")
        print("  Start it with: docker-compose up -d qdrant")
        confirm("  Is Qdrant running?")
        print(f"  Checking connection to {QDRANT_URL}...")
        if not check_qdrant_reachable():
            print(f"  ERROR: Qdrant not reachable at {QDRANT_URL}")
            print("  Make sure Docker is running and run: docker-compose up -d qdrant")
            sys.exit(1)
        print("  OK - Qdrant is reachable.")

    elif rag_type == "graph_rag":
        print("  Neo4j requires Docker.")
        print("  Start it with: docker-compose up -d neo4j")
        confirm("  Is Neo4j running?")
        print(f"  Checking connection to {NEO4J_URI}...")
        if not check_neo4j_reachable():
            print(f"  ERROR: Neo4j not reachable at {NEO4J_URI}")
            print("  Make sure Docker is running and run: docker-compose up -d neo4j")
            sys.exit(1)
        print("  OK - Neo4j is reachable.")

    elif rag_type == "filesystem_rag":
        print("  Filesystem RAG runs in-process - no Docker needed.")
        print("  NOTE: This RAG makes LLM calls DURING indexing (not just evaluation).")
        print("  Indexing 10 articles will cost ~$0.05-0.10 in OpenAI calls.")
        print("  OK")


def run_cleanup(rag_type: str) -> None:
    """Clean the target database for the given RAG type."""
    section("STEP 1 - CLEANUP")

    if rag_type == "vector_semantic":
        print(f"  Will DELETE directory: {CHROMA_DIR}")
        print("  This removes all previously indexed documents from ChromaDB.")
        confirm("  Proceed with cleanup?")
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            print(f"  Deleted: {CHROMA_DIR}")
        else:
            print("  Nothing to clean - directory did not exist.")

    elif rag_type == "vector_hybrid":
        print("  Qdrant auto-clears all points at the start of prepare_documents().")
        print("  No manual cleanup needed.")
        print("  OK - skipping cleanup step.")

    elif rag_type == "graph_rag":
        print("  Will run: MATCH (n) DETACH DELETE n  (clears all Neo4j nodes)")
        confirm("  Proceed with cleanup?")
        try:
            # Import here to avoid loading heavy deps at startup
            from rag_evaluator.config import settings
            from rag_evaluator.rag_implementations.graph_rag.indexer import GraphIndexer
            indexer = GraphIndexer(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password,
            )
            indexer.clear_graph()
            print("  Neo4j graph cleared.")
        except Exception as e:
            print(f"  ERROR during Neo4j cleanup: {e}")
            sys.exit(1)

    elif rag_type == "filesystem_rag":
        print(f"  Will DELETE directory: {FILESYSTEM_PREPARED_DIR}")
        print("  FilesystemRAG rebuilds this from scratch on every prepare run.")
        confirm("  Proceed with cleanup?")
        if FILESYSTEM_PREPARED_DIR.exists():
            shutil.rmtree(FILESYSTEM_PREPARED_DIR)
            print(f"  Deleted: {FILESYSTEM_PREPARED_DIR}")
        else:
            print("  Nothing to clean - directory did not exist.")


def run_prepare(rag_type: str) -> None:
    """Index the mini corpus into the target RAG database."""
    section("STEP 2 - PREPARE (index 10 articles)")
    print(f"  Corpus:   {MINI_CORPUS_DIR}/")
    print(f"  RAG type: {rag_type}")
    print("  This will make OpenAI embedding API calls.")
    if rag_type == "filesystem_rag":
        print("  NOTE: Filesystem RAG also calls the LLM during indexing (~$0.05-0.10).")
    confirm("  Proceed with indexing?")

    cmd = [
        "uv", "run", "rag-eval", "prepare",
        "--rag-type", rag_type,
        "--input-dir", str(MINI_CORPUS_DIR),
    ]
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode != 0:
        print(f"\n  ERROR: prepare failed (exit code {result.returncode})")
        print("  Fix the issue above and re-run with --skip-cleanup --skip-prepare")
        sys.exit(1)

    print("\n  Prepare completed successfully.")
    pause(
        f"  CHECK YOUR OPENAI USAGE: {OPENAI_USAGE_URL}\n"
        "  Note down the cost of indexing before continuing."
    )


def run_evaluate(rag_type: str) -> dict:
    """Run the 5-question evaluation and return the results dict."""
    section("STEP 3 - EVALUATE (5 questions)")
    print(f"  Test set: {MINI_TEST_SET}")
    print(f"  RAG type: {rag_type}")
    print("  This will make OpenAI LLM + DeepEval API calls (~$0.02-0.05).")
    confirm("  Proceed with evaluation?")

    report_dir = ROOT / "reports" / "verify"
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "rag-eval", "evaluate",
        "--rag-type", rag_type,
        "--test-set", str(MINI_TEST_SET),
        "--output", str(report_dir),
    ]
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode != 0:
        print(f"\n  ERROR: evaluate failed (exit code {result.returncode})")
        print("  Fix the issue above and re-run with --skip-cleanup --skip-prepare")
        sys.exit(1)

    print("\n  Evaluation completed.")
    pause(
        f"  CHECK YOUR OPENAI USAGE: {OPENAI_USAGE_URL}\n"
        "  Note down the cost of evaluation before continuing."
    )

    # Load the latest report for this RAG type from verify dir
    import glob as _glob
    import os
    type_file_map = {
        "vector_semantic": "chromadb_semantic_search",
        "vector_hybrid": "hybrid_search",
        "graph_rag": "neo4j_graph_rag",
        "filesystem_rag": "filesystem_rag",
    }
    pattern = str(report_dir / f"eval_{type_file_map.get(rag_type, rag_type)}*.json")
    files = _glob.glob(pattern)
    if not files:
        # Fallback: any eval json
        files = _glob.glob(str(report_dir / "eval_*.json"))
    if files:
        latest = max(files, key=os.path.getmtime)
        with open(latest, encoding="utf-8") as f:
            return json.load(f)
    return {}


def print_summary(rag_type: str, results: dict) -> bool:
    """Print pass/fail summary. Returns True if PASSED."""
    section("STEP 4 - SUMMARY")

    if not results:
        print("  Could not load evaluation results.")
        return False

    pass_rate = results.get("pass_rate", 0) / 100
    test_count = results.get("test_cases_count", 0)
    passed_verdict = pass_rate >= PASS_THRESHOLD

    metrics = results.get("metrics_summary", {})

    verdict = "PASS" if passed_verdict else "FAIL"
    print(f"\n  +{'-' * 45}+")
    print(f"  |  {rag_type:<43}|")
    print(f"  |  {test_count} questions{' ' * 35}|")
    print(f"  |  Pass rate: {int(pass_rate * test_count)}/{test_count} ({pass_rate:.0%})  ->  {verdict:<12}|")
    print(f"  +{'-' * 45}+")
    for key, label in [
        ("faithfulness_avg",         "Faithfulness      "),
        ("answer_relevancy_avg",     "Answer Relevancy  "),
        ("contextual_recall_avg",    "Contextual Recall "),
        ("contextual_precision_avg", "Context Precision "),
    ]:
        val = metrics.get(key)
        if val is not None:
            bar = "#" * int(val * 10) + "." * (10 - int(val * 10))
            print(f"  |  {label} {bar}  {val:.2f}  |")
    print(f"  +{'-' * 45}+")

    if passed_verdict:
        print("\n  RESULT: PASS - pipeline is working correctly.")
        print("  You can proceed to the next RAG type or run the full evaluation.")
    else:
        print("\n  RESULT: FAIL - pass rate is below 60%.")
        print("  This may be expected with only 5 questions.")
        print("  Check the report in reports/verify/ for details.")
        print("  If errors occurred (not just low scores), fix them before the full run.")

    return passed_verdict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive pipeline verification for RAG Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rag-type",
        required=True,
        choices=["vector_semantic", "vector_hybrid", "graph_rag", "filesystem_rag"],
        help="RAG type to verify",
    )
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip cleanup step")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare step")
    parser.add_argument("--cleanup-only", action="store_true", help="Only run cleanup then exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG Pipeline Verification")
    print(f"  RAG type: {args.rag_type}")
    print("=" * 60)

    # Phase 1: build mini dataset
    section("MINI DATASET")
    build_mini_dataset()

    # Phase 2: pre-flight
    preflight_check(args.rag_type)

    # Phase 3: cleanup
    if not args.skip_cleanup and not args.skip_prepare:
        run_cleanup(args.rag_type)
    elif args.skip_cleanup:
        print("\n  [--skip-cleanup] Skipping cleanup step.")

    if args.cleanup_only:
        print("\n  [--cleanup-only] Done.")
        return 0

    # Phase 4: prepare
    if not args.skip_prepare:
        run_prepare(args.rag_type)
    else:
        print("\n  [--skip-prepare] Skipping prepare step.")

    # Phase 5: evaluate
    results = run_evaluate(args.rag_type)

    # Phase 6: summary
    passed = print_summary(args.rag_type, results)

    pause("  Review the summary above and take any notes.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
