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

# ── paths (relative to project root) ────────────────────────────────────────
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


# ── helpers ──────────────────────────────────────────────────────────────────

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
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


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
