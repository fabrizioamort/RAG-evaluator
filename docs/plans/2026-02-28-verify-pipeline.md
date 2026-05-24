# Verification Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `scripts/verify_pipeline.py` — an interactive script that smoke-tests one RAG type at a time using a tiny dataset (mini corpus + 5 questions) with confirmation prompts between every destructive or expensive step.

**Architecture:** The script has two phases. Phase 1 creates a mini dataset by picking 5 questions from `test_set_multihop.json` that have known evidence sources, then copying those source articles to `data/raw/multihop_verify/`. Phase 2 runs a 4-step interactive loop (cleanup → prepare → evaluate → summary) for the chosen RAG type, pausing for user confirmation and cost checks at each boundary.

**Tech Stack:** Python stdlib only (no new deps) + existing `rag_evaluator` package internals for cleanup calls + `subprocess` to invoke `rag-eval` CLI commands for prepare and evaluate.

---

## Important Codebase Facts

Before reading any task, understand these behaviours — they drive every decision in this plan:

| RAG type | Cleanup needed before prepare? | How to clean |
|---|---|---|
| `vector_semantic` (ChromaDB) | **YES** — appends on re-index, creates duplicates | Delete `data/chroma_db/` directory |
| `vector_hybrid` (Qdrant) | **NO** — auto-deletes all points before re-index | Just check Docker is up |
| `graph_rag` (Neo4j) | **YES** — appends on re-index, creates duplicates | Call `GraphIndexer.clear_graph()` via Python |
| `filesystem_rag` | **NO** — full rebuild overwrites previous index | Just delete `data/prepared/filesystem_rag/` as safety |

**Neo4j cleanup code path:**
```python
from rag_evaluator.rag_implementations.graph_rag.indexer import GraphIndexer
indexer = GraphIndexer(neo4j_uri, neo4j_username, neo4j_password)
indexer.clear_graph()   # runs MATCH (n) DETACH DELETE n
```

**Chroma persist path:** `./data/chroma_db` (from `settings.chroma_persist_directory`)

**Filesystem prepared path:** `data/prepared/filesystem_rag` (from `FilesystemRAG.__init__` default)

**Qdrant URL:** `http://localhost:6333` (from `settings.qdrant_url`)

**Neo4j URI:** `bolt://localhost:7687` (from `settings.neo4j_uri`)

---

## Task 1: Create the mini dataset builder

**Files:**
- Create: `scripts/verify_pipeline.py` (skeleton + `build_mini_dataset()` function only)

The mini dataset is built from the already-downloaded MultiHop-RAG data. Strategy:
1. Read `data/test_set_multihop.json`
2. Filter to test cases that have at least one non-empty `evidence_sources` path pointing to a file that actually exists on disk
3. Take the first 5 such test cases
4. Collect all unique article `.txt` files they reference
5. Copy those files to `data/raw/multihop_verify/` (create dir if needed)
6. Write those 5 test cases to `data/test_set_verify.json`

If `data/raw/multihop_verify/` and `data/test_set_verify.json` already exist, skip creation and print "Mini dataset already exists, skipping."

**Step 1: Create the file with the mini-dataset builder**

```python
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
```

**Step 2: Run a quick sanity check**

```powershell
cd C:\Users\fabri\projects\RAG-evaluator
uv run python -c "
from scripts.verify_pipeline import build_mini_dataset
build_mini_dataset()
"
```

Expected output:
```
  Building mini dataset from full MultiHop-RAG data...
  Created corpus:   data\raw\multihop_verify\ (N articles)
  Created test set: data\test_set_verify.json (5 questions)
```

If it prints `Mini dataset already exists`, delete `data/raw/multihop_verify/` and `data/test_set_verify.json` first and re-run.

**Step 3: Commit**

```bash
git add scripts/verify_pipeline.py data/test_set_verify.json
git commit -m "feat: add verify_pipeline script skeleton with mini dataset builder"
```

---

## Task 2: Add pre-flight checks

**Files:**
- Modify: `scripts/verify_pipeline.py` — add `preflight_check(rag_type)` function

Pre-flight checks run before any destructive action. For Docker-based RAG types they ask the user to confirm the service is up, then verify connectivity.

**Step 1: Add the pre-flight function after `build_mini_dataset()`**

```python
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
        print("  ChromaDB runs in-process — no Docker needed.")
        print("  OK")

    elif rag_type == "vector_hybrid":
        print("  Qdrant requires Docker.")
        print(f"  Start it with: docker-compose up -d qdrant")
        confirm("  Is Qdrant running?")
        print(f"  Checking connection to {QDRANT_URL}...")
        if not check_qdrant_reachable():
            print(f"  ERROR: Qdrant not reachable at {QDRANT_URL}")
            print("  Make sure Docker is running and run: docker-compose up -d qdrant")
            sys.exit(1)
        print("  OK — Qdrant is reachable.")

    elif rag_type == "graph_rag":
        print("  Neo4j requires Docker.")
        print(f"  Start it with: docker-compose up -d neo4j")
        confirm("  Is Neo4j running?")
        print(f"  Checking connection to {NEO4J_URI}...")
        if not check_neo4j_reachable():
            print(f"  ERROR: Neo4j not reachable at {NEO4J_URI}")
            print("  Make sure Docker is running and run: docker-compose up -d neo4j")
            sys.exit(1)
        print("  OK — Neo4j is reachable.")

    elif rag_type == "filesystem_rag":
        print("  Filesystem RAG runs in-process — no Docker needed.")
        print("  NOTE: This RAG makes LLM calls DURING indexing (not just evaluation).")
        print("  Indexing 10 articles will cost ~$0.05-0.10 in OpenAI calls.")
        print("  OK")
```

**Step 2: Run a quick check**

```powershell
uv run python -c "
import sys; sys.argv = ['verify_pipeline.py']
from scripts.verify_pipeline import preflight_check
preflight_check('vector_semantic')
"
```

Expected:
```
  ChromaDB runs in-process — no Docker needed.
  OK
```

**Step 3: Commit**

```bash
git add scripts/verify_pipeline.py
git commit -m "feat: add pre-flight service checks to verify_pipeline"
```

---

## Task 3: Add cleanup functions

**Files:**
- Modify: `scripts/verify_pipeline.py` — add `run_cleanup(rag_type)` function

**Step 1: Add cleanup function**

```python
def run_cleanup(rag_type: str) -> None:
    """Clean the target database for the given RAG type."""
    section("STEP 1 — CLEANUP")

    if rag_type == "vector_semantic":
        print(f"  Will DELETE directory: {CHROMA_DIR}")
        print("  This removes all previously indexed documents from ChromaDB.")
        confirm("  Proceed with cleanup?")
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            print(f"  Deleted: {CHROMA_DIR}")
        else:
            print("  Nothing to clean — directory did not exist.")

    elif rag_type == "vector_hybrid":
        print("  Qdrant auto-clears all points at the start of prepare_documents().")
        print("  No manual cleanup needed.")
        print("  OK — skipping cleanup step.")

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
            print("  Nothing to clean — directory did not exist.")
```

**Step 2: Test cleanup for ChromaDB**

```powershell
uv run python -c "
from scripts.verify_pipeline import run_cleanup
run_cleanup('vector_semantic')
"
```

Expected: prompt asking to confirm, then prints deletion message.

**Step 3: Commit**

```bash
git add scripts/verify_pipeline.py
git commit -m "feat: add per-RAG cleanup functions to verify_pipeline"
```

---

## Task 4: Add prepare and evaluate runners

**Files:**
- Modify: `scripts/verify_pipeline.py` — add `run_prepare()` and `run_evaluate()` functions

Both functions invoke the existing `rag-eval` CLI via `subprocess` so they reuse all existing logic without duplication.

**Step 1: Add the two runner functions**

```python
def run_prepare(rag_type: str) -> None:
    """Index the mini corpus into the target RAG database."""
    section("STEP 2 — PREPARE (index 10 articles)")
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
    section("STEP 3 — EVALUATE (5 questions)")
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
    import glob, os
    type_file_map = {
        "vector_semantic": "chromadb_semantic_search",
        "vector_hybrid": "hybrid_search",
        "graph_rag": "neo4j_graph_rag",
        "filesystem_rag": "filesystem_rag",
    }
    pattern = str(report_dir / f"eval_{type_file_map.get(rag_type, rag_type)}*.json")
    files = glob.glob(pattern)
    if not files:
        # Fallback: any eval json
        files = glob.glob(str(report_dir / "eval_*.json"))
    if files:
        latest = max(files, key=os.path.getmtime)
        with open(latest, encoding="utf-8") as f:
            return json.load(f)
    return {}
```

**Step 2: Smoke-test the prepare runner (requires prior cleanup)**

```powershell
uv run python -c "
from scripts.verify_pipeline import run_prepare
run_prepare('vector_semantic')
"
```

Expected: confirmation prompt → indexes 10 articles → cost pause → returns.

**Step 3: Commit**

```bash
git add scripts/verify_pipeline.py
git commit -m "feat: add prepare and evaluate runners to verify_pipeline"
```

---

## Task 5: Add summary printer and main entrypoint

**Files:**
- Modify: `scripts/verify_pipeline.py` — add `print_summary()` and `main()`

**Step 1: Add summary and main**

```python
def print_summary(rag_type: str, results: dict) -> bool:
    """Print pass/fail summary. Returns True if PASSED."""
    section("STEP 4 — SUMMARY")

    if not results:
        print("  Could not load evaluation results.")
        return False

    pass_rate = results.get("pass_rate", 0) / 100
    test_count = results.get("test_cases_count", 0)
    passed = int(pass_rate * test_count)
    passed_verdict = pass_rate >= PASS_THRESHOLD

    metrics = results.get("metrics_summary", {})

    print(f"\n  ┌{'─' * 45}┐")
    print(f"  │  {rag_type:<43}│")
    print(f"  │  {test_count} questions                             │")
    verdict = "PASS ✓" if passed_verdict else "FAIL ✗"
    print(f"  │  Pass rate: {passed}/{test_count} ({pass_rate:.0%})  →  {verdict:<12}│")
    print(f"  ├{'─' * 45}┤")
    for key, label in [
        ("faithfulness_avg",        "Faithfulness      "),
        ("answer_relevancy_avg",    "Answer Relevancy  "),
        ("contextual_recall_avg",   "Contextual Recall "),
        ("contextual_precision_avg","Context Precision "),
    ]:
        val = metrics.get(key)
        if val is not None:
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            print(f"  │  {label} {bar}  {val:.2f}  │")
    print(f"  └{'─' * 45}┘")

    if passed_verdict:
        print("\n  RESULT: PASS — pipeline is working correctly.")
        print("  You can proceed to the next RAG type or run the full evaluation.")
    else:
        print("\n  RESULT: FAIL — pass rate is below 60%.")
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
```

**Step 2: Run a full end-to-end test with ChromaDB**

```powershell
uv run python scripts/verify_pipeline.py --rag-type vector_semantic
```

Walk through all prompts. Expected final output: summary table with PASS or FAIL verdict.

**Step 3: Commit**

```bash
git add scripts/verify_pipeline.py
git commit -m "feat: add summary and main entrypoint to verify_pipeline"
```

---

## Task 6: Final integration check — all 4 RAG types

Run the verification for all 4 RAG types in order, one at a time. For each one:

1. `uv run python scripts/verify_pipeline.py --rag-type vector_semantic`
2. `uv run python scripts/verify_pipeline.py --rag-type vector_hybrid`  ← start `docker-compose up -d qdrant` first
3. `uv run python scripts/verify_pipeline.py --rag-type graph_rag`  ← start `docker-compose up -d neo4j` first
4. `uv run python scripts/verify_pipeline.py --rag-type filesystem_rag`

For each run: record OpenAI cost at the two pause points (after prepare and after evaluate).

If any step fails: fix the root cause before running the full pipeline. Common issues and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: Input directory not found` | Mini corpus not created | Delete `data/test_set_verify.json`, re-run script |
| `Error connecting to Qdrant` | Docker not running | `docker-compose up -d qdrant`, wait 10s |
| `ServiceUnavailable: Neo4j` | Docker not running or slow start | `docker-compose up -d neo4j`, wait 30s |
| `OPENAI_API_KEY not set` | Missing .env | Add `OPENAI_API_KEY=sk-...` to `.env` |
| Pass rate 0% | All retrieval misses | Check corpus files exist in multihop_verify/ |
| OOM / process killed | RAM pressure | Close browser tabs, run Docker services one at a time |

**After all 4 pass:**
- Mini verification is complete
- Run the full pipeline:
  ```powershell
  # Full indexing
  uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw/multihop_rag
  uv run rag-eval prepare --rag-type vector_hybrid --input-dir data/raw/multihop_rag
  uv run rag-eval prepare --rag-type graph_rag --input-dir data/raw/multihop_rag
  uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw/multihop_rag

  # Full evaluation
  uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set_multihop.json
  uv run rag-eval evaluate --rag-type vector_hybrid --test-set data/test_set_multihop.json
  uv run rag-eval evaluate --rag-type graph_rag --test-set data/test_set_multihop.json
  uv run rag-eval evaluate --rag-type filesystem_rag --test-set data/test_set_multihop.json
  ```
