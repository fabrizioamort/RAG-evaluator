# Legal RAG Bench - Manual UI Walkthrough

Step-by-step guide for running the Legal RAG Bench architecture comparison
through the platform UI, and comparing results against the Isaacus paper.

See `docs/legal-rag-plan.md` for the full plan and `docs/legal-rag-implementation-memory.md`
for implementation status.

## Prerequisites: data is already converted

The converted dataset lives in `data/legal_rag_bench/` (gitignored):

- **Full:** `data/legal_rag_bench/full/raw/` = 4,876 passage files +
  `full/test_set.json` (100 questions, each with `relevant_passage_id`).
- **Subset:** `data/legal_rag_bench/subset/raw/` = 50 files +
  `subset/test_set.json` (10 questions).

Content mode is `clean` (only passage text indexed - the correct default for
paper calibration). To regenerate:

```powershell
uv run python scripts/convert_legal_rag_bench.py --content-mode clean
```

## 0. Start the stack

```powershell
# Infra: Postgres (backend) + Qdrant (needed for vector_hybrid). Neo4j not needed.
docker-compose up -d postgres qdrant

# Backend (new terminal)
cd platform/backend
uv run python dev_server.py

# Frontend (new terminal)
cd platform/frontend
npm run dev
```

Open the UI at the URL Vite prints (usually `http://localhost:5173`).

## Phase 0 - Smoke test first (use the 50-passage SUBSET)

Goal: prove the whole path works before spending money/time on 100 Q x 3
systems. Do this with **one system** (Chroma / `vector_semantic`).

1. **Create a project** (e.g. "Legal RAG Bench").
2. **Import the Knowledge Base** from the subset corpus folder:
   `data/legal_rag_bench/subset/raw` (50 `.txt` files).
   - Verify **document count = 50**.
3. **Import the Test Set** from `data/legal_rag_bench/subset/test_set.json`
   (10 questions).
   - Verify a question shows its `relevant_passage_id` (in metadata). This is
     the gold field that hit@5 depends on.
4. **Create a RAG Config** (`vector_semantic`) with the benchmark-critical
   settings (plan section 7.2):
   - embedding_model `text-embedding-3-large`, embedding_dimension `3072`
   - chunk_size `8000`, chunk_overlap `0` (one passage = one chunk)
   - llm_model `gpt-4o-mini`

   Note: **temperature** and **top_k** are *not* set here.
   - Temperature is hardcoded to `0.0` for both generation and the judge
     (`evaluation_runner.py`, `legal_rag_bench_judge.py`) - already the
     benchmark value, nothing to configure.
   - Vector `top_k` is a per-evaluation query override, set in the Start
     Evaluation wizard (step 6), not on the RAG config. This lets the same
     index be queried at different k without rebuilding.
5. **Build the index** from that config + KB. On the index detail (plan
   section 7.3) verify:
   - **chunk count = 50** (must equal document count - if not, chunking split
     passages; stop and fix before trusting hit@5)
   - status = ready, embedding model = text-embedding-3-large.
6. **Create an evaluation** via the Start Evaluation wizard. The settings are
   spread across the wizard steps:
   - **Test Set / Index steps:** select the Legal RAG Bench test set and the
     ready index.
   - **Query step:** set **Top K = 5**; pick the **Generation Provider / RAG
     Model** (`gpt-4o-mini`) and, on the same screen, the **Judge Provider /
     Judge Model** (the judge selector is the second model picker - easy to
     miss).
   - **Metrics step:** under the **Legal RAG Bench** group, enable
     **Legal RAG Bench: Retrieval** and **Legal RAG Bench: Binary Judge**.
     These are opt-in (off by default). The DeepEval metrics above them are
     optional secondary diagnostics.
7. **Run it.** Open a result row and confirm the **Legal RAG Bench panel**
   shows:
   - retrieved context + retrieval trace are visible,
   - a real **Hit@5** badge, gold passage id + rank, Correct/Grounded,
     taxonomy chip.

**Critical Phase-0 gate:** if *every* question shows Hit@5 = miss, the
retrieved-source-id to gold-passage-id matching isn't lining up (corpus
filenames are `passage_0001__<id>.txt`, gold is the bare `<id>`). Verify on a
couple of questions you know should hit before scaling up.

### Phase 0 exit criteria

- index builds from UI;
- evaluation runs from UI;
- retrieved context and trace are visible;
- `hit@5` (or `gold_accessed`) is computed;
- result export works.

## Phase 1 - The real comparison (full 100 Q, 4,876 passages, 3 systems)

Hold the KB and Test Set **fixed** across all three systems (same KB version,
same test set, same generator, same judge, same `top_k=5`).

1. **Import full KB** from `data/legal_rag_bench/full/raw` -> verify
   **document count = 4,876**.
2. **Import full Test Set** from `data/legal_rag_bench/full/test_set.json`
   (100 questions).
3. **Build three indexes from the same KB version**, identical
   embedding/generator settings:
   - `vector_semantic` (Chroma) - chunk_size 8000 / overlap 0 ->
     **verify chunk count = 4,876**
   - `vector_hybrid` (Qdrant) - same -> **verify chunk count = 4,876**
     (needs the Qdrant container)
   - `filesystem_rag` - agentic; set and **record** its retrieval budget
     (max tool calls / file reads)
4. **Run three evaluations**, one per index, all with: same test set,
   `top_k=5` (vector), generator `gpt-4o-mini`, the same judge model/provider,
   Legal RAG Bench judge enabled.
5. **Create a Comparison:** baseline = Chroma, compared = Qdrant hybrid +
   Filesystem. Open the **"Legal RAG Bench" tab** -> side-by-side Hit@5 /
   Gold accessed / Correct / Grounded + taxonomy.
6. **Export** from the comparison header: **Markdown** (full report +
   manifests), **Headline CSV**, **Taxonomy CSV**, **JSONL** (per-question
   reproducibility). These are your article tables.

## How to compare against the paper (honestly)

Use the comparability rules from the plan (section 12). The safe claims:

- **Chroma calibration anchor:** the paper reports **hit@5 ~= 52.0%** for
  `text-embedding-3-large`. Compare your Chroma hit@5 to a **broad range**
  around that. If it's in the ballpark, you can say *"calibrated Chroma against
  the paper's Text Embedding 3 Large row"* - do **not** claim "we reproduced
  the paper".
- **Filesystem RAG:** report **`gold_accessed`**, never hit@5 (it ignores
  top_k). This is built into the metric/export already.
- **Correctness & groundedness:** call them **indicative only**. Your judge
  (e.g. gpt-4o) is not the paper's GPT-5.2 binary judge, so absolute numbers
  aren't comparable - but Chroma-vs-Qdrant-vs-Filesystem deltas *within your
  run* are fair game since the judge/generator/embedding are held constant.
- Headline numbers use **clean** content mode (the default in the converted
  data).

### Paper reference numbers (Text Embedding 3 Large, k=5)

| Metric | Paper value | Use as |
|---|---|---|
| Retrieval accuracy (hit@5) | 52.0% | Primary calibration anchor (broad range) |
| Correctness | 76.5% | Indicative only (different judge) |
| Groundedness | 91.5% | Indicative only (different judge) |

## Key checkpoints (do not skip)

- **Document count = 4,876** after KB import (50 for subset).
- **Vector chunk count = 4,876** after each Chroma/Qdrant build (50 for
  subset). If chunk count != document count, chunking split passages.
- **Phase-0 Hit@5 sanity check** on a few known-hit questions before running
  the full judge pass.

If any of these fail, stop there rather than running the full judge pass.
