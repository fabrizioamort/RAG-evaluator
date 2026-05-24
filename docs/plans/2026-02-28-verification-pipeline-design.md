# Verification Pipeline Design

**Date:** 2026-02-28
**Status:** Approved

## Goal

Provide a safe, interactive way to verify that the full RAG evaluation pipeline (index → evaluate) works correctly on a small dataset before committing to a full run that costs real money.

## Context

- Dataset: MultiHop-RAG (609 news articles, 2,556 multi-hop questions) already downloaded to `data/raw/multihop_rag/` and `data/test_set_multihop.json`
- Hardware: Windows 11, Ryzen 5, 8GB RAM
- Budget concern: avoid wasting tokens if the pipeline crashes mid-run
- RAG systems: vector_semantic (ChromaDB), vector_hybrid (Qdrant), graph_rag (Neo4j), filesystem_rag
- Order of verification: ChromaDB → Qdrant → Neo4j → Filesystem (one at a time)

## Solution

A single script `scripts/verify_pipeline.py` that runs a smoke test for one RAG type at a time using a tiny dataset (10 articles + 5 questions), with interactive confirmation prompts between every destructive or expensive step.

## Mini Dataset

Extracted once from the already-downloaded MultiHop-RAG data:

- **Documents:** `data/raw/multihop_verify/` — 10 article `.txt` files (first 10 from corpus)
- **Test set:** `data/test_set_verify.json` — 5 questions whose evidence sources are contained within those 10 articles

## Per-RAG Cleanup Behavior

| RAG Type | Cleanup Action |
|----------|---------------|
| `vector_semantic` | Delete `data/chroma_db/` directory |
| `vector_hybrid` | No manual cleanup needed (Qdrant auto-clears on prepare) |
| `graph_rag` | Run `MATCH (n) DETACH DELETE n` via Neo4j Bolt |
| `filesystem_rag` | Delete `data/prepared/filesystem_rag/` directory |

## Interactive Flow

For each RAG type the script runs 4 stages with confirmation prompts between each:

```
PRE-FLIGHT   → service reachability check (Docker-based RAGs ask user to confirm service is up)
STEP 1       → CLEANUP: prompt → confirm → delete/clear
STEP 2       → PREPARE: prompt → confirm → index 10 articles → pause for cost check
STEP 3       → EVALUATE: prompt → confirm → run 5 questions → pause for cost check
STEP 4       → SUMMARY: pass/fail table with metrics
```

Every prompt is `[y/n]` or `Press Enter to continue`. The script exits cleanly on `n` or `Ctrl+C` at any point.

## PASS Criteria

60% pass rate (3 of 5 questions). Intentionally lenient — we are testing that the pipeline runs correctly, not that it performs well on a tiny dataset.

## Usage

```powershell
# Run verification for one RAG type
uv run python scripts/verify_pipeline.py --rag-type vector_semantic
uv run python scripts/verify_pipeline.py --rag-type vector_hybrid
uv run python scripts/verify_pipeline.py --rag-type graph_rag
uv run python scripts/verify_pipeline.py --rag-type filesystem_rag

# Optional flags
--skip-cleanup     # Skip cleanup step (DB already clean)
--skip-prepare     # Skip prepare step (already indexed)
--cleanup-only     # Only run cleanup, then exit
```

## What Success Means

If all 4 RAG types pass the verification test, you can be confident that:
- The full indexing of 609 articles will run without crashing
- The full evaluation of 100 questions will complete successfully
- The only remaining risk is cost (which the cost estimates already cover)

## Remaining Risks Even After Passing

- **Memory:** 609 articles is ~30x more data. Monitor RAM during full indexing.
- **Neo4j heap:** May need tuning for the full corpus. Watch for OOM errors.
- **Long-running jobs:** Full indexing may take 30-60 min per RAG type. Keep PC awake.
- **OpenAI rate limits:** Filesystem RAG makes LLM calls during indexing. May hit rate limits on large corpus.
