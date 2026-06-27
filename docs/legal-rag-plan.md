# Legal RAG Bench Article Plan - UI-First Workflow

Status: updated after codebase, paper, CLI, and platform UI review.

Goal: produce a credible article showing what this codebase can do by running
Legal RAG Bench across three RAG architectures, using the **web UI as the main
experiment surface**:

1. `vector_semantic` - Chroma semantic vector search.
2. `vector_hybrid` - Qdrant dense + sparse hybrid search.
3. `filesystem_rag` - agentic filesystem search.

The article should not claim to reproduce the paper end-to-end. The correct
framing is:

> We use Legal RAG Bench as a controlled legal QA benchmark to compare three RAG
> architectures through this platform UI. The Isaacus paper is used as a
> calibration reference, especially for judge-free retrieval accuracy.

## 1. References

- Blog: https://isaacus.com/blog/legal-rag-bench
- Paper: https://arxiv.org/abs/2603.01710
- Paper PDF: https://arxiv.org/pdf/2603.01710
- Official repo: https://github.com/isaacus-dev/legal-rag-bench
- Dataset: https://huggingface.co/datasets/isaacus/legal-rag-bench

## 2. UI vs CLI Architecture

The platform UI does **not** shell out to `rag-eval prepare` or
`rag-eval evaluate`.

There are two orchestration paths:

| Surface | Orchestration | Core RAG code |
|---|---|---|
| CLI | `src/rag_evaluator/cli.py` | Same RAG implementation classes |
| UI | React -> FastAPI -> backend services | Same RAG implementation classes |

The UI workflow uses persistent platform entities:

1. Project.
2. Knowledge Base.
3. RAG Config.
4. Knowledge Base Index.
5. Test Set.
6. Evaluation.
7. Comparison / result detail / retrieval trace.

Backend services involved:

- `platform/backend/app/services/index_build_service.py`
  creates and builds immutable indexes.
- `platform/backend/app/services/rag_adapter.py`
  instantiates core RAG classes from platform config snapshots.
- `platform/backend/app/services/evaluation_runner.py`
  runs evaluations against ready indexes and stores results.

Therefore, for the article, the UI should be the primary workflow. CLI/scripts
should only support dataset conversion, optional bulk import/export, local smoke
checks, and backend service development.

## 3. Paper Baseline

Legal RAG Bench contains:

- 4,876 passages from the Victorian Criminal Charge Book.
- 100 expert-written legal questions.
- One gold supporting passage id per question.
- Long-form reference answers.

Original Isaacus harness:

- Indexing: FAISS over the dataset `text` field.
- No extra chunking: one Legal RAG Bench passage is one retrieved document.
- Retrieval depth: `k=5`.
- Retrieval accuracy: gold passage id appears in the top 5 retrieved documents.
- Embedding models: Kanon 2 Embedder, OpenAI Text Embedding 3 Large, Gemini
  Embedding 001.
- Generators: GPT-5.2, Gemini 3.1 Pro.
- Judge: GPT-5.2 with high reasoning, returning binary `correct` and
  `grounded` verdicts.

Important calibration number:

- OpenAI Text Embedding 3 Large reached 52.0% retrieval accuracy at `k=5`.
- Its average paper scores were 76.5% correctness and 91.5% groundedness.

Use the 52.0% retrieval accuracy as a sanity check for our `vector_semantic`
calibration run. Do not present it as a guaranteed target, because our stack
uses Chroma, different prompt plumbing, and potentially different indexed text.

## 4. Core Thesis For The Article

The paper's main claim is that retrieval quality sets the ceiling for legal RAG.
Our article should extend that idea from "which embedding model wins?" to:

> How much does the retrieval architecture itself change outcomes when the
> dataset, embedding model, generator, and judge are held fixed?

The UI angle strengthens the article:

> This is not just a benchmark script. It is a reusable evaluation workflow:
> upload corpus, configure RAG, build isolated indexes, launch evaluations,
> inspect traces, and compare systems.

## 5. What Is Comparable

| Dimension | Paper | Our UI Run | Comparability |
|---|---|---|---|
| Dataset | Legal RAG Bench full set | Same 4,876 passages + 100 QA imported into platform | Direct |
| Retrieval unit | One passage | Must force one passage = one chunk for vector indexes | Direct if enforced |
| Retrieval depth | `k=5` | Query override `top_k=5` for Chroma and Qdrant | Direct |
| Filesystem retrieval depth | Not applicable | Agent decides files to read | Not direct |
| Embedding model | Text Embedding 3 Large for calibration row | RAG Config uses Text Embedding 3 Large | Direct if Qdrant dimension is fixed |
| Generator | GPT-5.2 / Gemini 3.1 Pro | Evaluation query/generation model from UI config | Comparable only when disclosed |
| Judge | GPT-5.2 high reasoning binary judge | Platform judge model / benchmark judge metric | Indicative unless same judge/prompt |
| Correctness | Binary entailment | Binary benchmark judge preferred; DeepEval optional | Indicative |
| Groundedness | Binary support in retrieved context | Binary benchmark judge preferred; faithfulness optional | Indicative |
| Retrieval accuracy | Gold id in top 5 | `hit@5` for vector systems | Primary anchor |

## 6. Review Findings Incorporated

### Finding 1: UI and CLI are independent orchestration paths

The UI can run the same broad workflow as the CLI, but it does not call the CLI.
It goes through FastAPI and backend services.

Implication:

- Do not build the article workflow around `rag-eval evaluate`.
- Add missing Legal RAG Bench logic to backend services and expose it in the UI.
- Keep scripts only for dataset conversion and backend smoke testing.

### Finding 2: Current UI evaluations do not yet produce paper-style metrics

The platform currently stores DeepEval-style metrics and retrieval artifacts.
For Legal RAG Bench, it also needs:

- `relevant_passage_id` on test cases.
- extracted `retrieved_passage_ids`.
- vector `hit@5`.
- filesystem `gold_accessed`.
- binary `correct` and `grounded` judge verdicts.
- paper-style taxonomy.
- summary/export tables for the article.

### Finding 3: Hybrid Qdrant currently assumes 1536-dimensional vectors

`vector_hybrid` creates the Qdrant dense vector config with a hardcoded 1536
dimension, which matches `text-embedding-3-small`, not
`text-embedding-3-large`.

Required fix:

- derive vector dimension from the selected embedding model, or
- add a config value such as `embedding_dimension=3072` for
  `text-embedding-3-large`.

The UI must expose or store this in the RAG Config / index build snapshot.

### Finding 4: Chroma and Hybrid must be forced to one passage per chunk

The paper indexed each benchmark passage as one document. Our vector systems
use text splitters by default.

Required UI-configurable build parameters:

- `chunk_size`: large enough to keep every passage intact, e.g. `8000` chars.
- `chunk_overlap`: `0`.
- Verify post-indexing that indexed chunk count equals 4,876 for the full run.

If the index `chunk_count` is not 4,876, stop and debug before running judge
metrics.

### Finding 5: Filesystem RAG cannot be reported as `hit@5`

`filesystem_rag` ignores `top_k`; the agent decides what to inspect. Therefore,
do not label its retrieval metric as `hit@5`.

Use filesystem-specific retrieval metrics:

- `gold_accessed`: whether the gold passage file was read or included in
  `context_sources`.
- `files_read_count`: retrieval budget actually consumed.
- `gold_access_rank`: optional, if the trace can recover ordering.
- `answer_correct` and `answer_grounded`: same judge as the vector systems.

In UI tables, label vector metrics as `hit@5` and filesystem metrics as
`gold_accessed`.

### Finding 6: The converter currently adds metadata into text

The current converter writes files with headers such as `Passage ID:` and
`Title:`. That improves traceability, but the official harness indexes the
dataset text and keeps ids in metadata.

Decision:

- For calibration against the paper, add a `clean` content mode that indexes
  only the original passage text, with passage id recoverable from filename or
  metadata.
- For demos and qualitative UI screenshots, an `annotated` mode is fine.

Recommended default for headline numbers: `clean`.

### Finding 7: DeepEval is useful, but not the same judge as the paper

DeepEval metrics can be included, but the article should not treat them as a
direct substitute for the paper's binary GPT-5.2 judge.

Preferred approach:

1. Add a benchmark-specific binary judge metric to the backend evaluation
   runner.
2. Return JSON: `{"correct": bool, "grounded": bool, "reasoning": str}`.
3. Store the raw judge output as an artifact.
4. Keep DeepEval as optional secondary diagnostics in the UI.

Primary article metrics should be:

- retrieval: `hit@5` or `gold_accessed`.
- correctness: binary benchmark judge.
- groundedness: binary benchmark judge.
- taxonomy: derived from the three binary signals.

## 7. Required Product/Backend Work

### 7.1 Dataset conversion and import

Keep a script such as:

```text
scripts/convert_legal_rag_bench.py
```

Responsibilities:

- Convert Hugging Face/local JSONL data into platform-importable corpus files.
- Support `--content-mode clean|annotated`.
- Produce a platform test set import file with:
  - `question`,
  - `expected_answer`,
  - `ground_truth_context`,
  - `relevant_passage_id`,
  - legal benchmark metadata.

UI objective:

- The user should import the converted corpus as a Knowledge Base.
- The user should import the converted QA as a Test Set.

If the current UI import format cannot preserve `relevant_passage_id`, extend
the test set schema/import path before running the final benchmark.

### 7.2 RAG Config UI must expose benchmark-critical build settings

The UI RAG Config form must allow or preserve:

```yaml
embedding_model: text-embedding-3-large
embedding_dimension: 3072
chunk_size: 8000
chunk_overlap: 0
llm_model: gpt-4o-mini
temperature: 0
vector_top_k_default: 5
filesystem_max_tool_calls: fixed_and_reported
filesystem_max_file_reads: fixed_and_reported
```

Some values may live in `parameters`, some in top-level RAG config fields.
What matters is that they are frozen into the `KnowledgeBaseIndex`
`config_snapshot`, because UI evaluations run against ready indexes.

### 7.3 Index build checks in UI

After creating each index from the UI:

- Show document count.
- Show chunk count.
- Show embedding model.
- Show build config snapshot.
- Show storage isolation id / physical id.

For the full Legal RAG run, the UI should make it easy to verify:

- document count = 4,876.
- vector chunk count = 4,876.
- index status = ready.
- build config uses `text-embedding-3-large`.
- Chroma and Qdrant indexes were built from the same Knowledge Base version.

### 7.4 Evaluation UI must support benchmark metrics

Extend evaluation creation/results to support a Legal RAG Bench mode.

Evaluation creation should allow:

- selecting the ready index,
- selecting the Legal RAG Bench test set,
- setting `top_k=5` for vector systems,
- setting generator model/provider,
- setting judge model/provider,
- selecting `legal_rag_binary_judge`,
- optionally selecting DeepEval diagnostics.

Per-result rows should store and display:

```json
{
  "system": "vector_semantic",
  "question_id": "legal_001",
  "question": "...",
  "reference_answer": "...",
  "relevant_passage_id": "...",
  "answer": "...",
  "retrieved_sources": ["..."],
  "retrieved_passage_ids": ["..."],
  "hit_at_5": true,
  "gold_accessed": true,
  "judge": {
    "correct": true,
    "grounded": true,
    "reasoning": "..."
  },
  "taxonomy": "success",
  "latency_ms": 1234,
  "token_usage": {}
}
```

For vector systems, `hit_at_5` is primary. For filesystem, `gold_accessed` is
primary and `hit_at_5` should be null.

### 7.5 Add taxonomy derivation

Use the paper-inspired hierarchy:

| Condition | Label |
|---|---|
| `correct = true` and `grounded = true` | `success` |
| `grounded = false` | `hallucination_or_ungrounded` |
| `grounded = true`, `correct = false`, retrieved/gold access = false | `retrieval_error` |
| `grounded = true`, `correct = false`, retrieved/gold access = true | `reasoning_error` |

For filesystem, use `gold_accessed` instead of `hit_at_5`.

### 7.6 Add comparison/export support

The UI already has comparison concepts. Extend them for Legal RAG Bench:

- compare Chroma, Qdrant hybrid, and filesystem evaluations side by side;
- show retrieval score, correctness, groundedness, taxonomy, latency, cost;
- export article-ready CSV/Markdown;
- export per-question JSONL for reproducibility;
- include run manifests and exact config snapshots in the export.

## 8. UI Workflow For The Article

### Step 1: Prepare data with script

Use CLI only for conversion:

```powershell
uv run python scripts/convert_legal_rag_bench.py --content-mode clean
```

Output should be importable into the platform as:

- a Knowledge Base corpus containing 4,876 passage files;
- a Test Set containing 100 QA rows and `relevant_passage_id`.

### Step 2: Start the platform

```powershell
docker-compose up -d postgres qdrant

cd platform/backend
uv run python dev_server.py

cd platform/frontend
npm run dev
```

Use the browser UI for the remaining steps.

### Step 3: Create project

Create a project, for example:

```text
Legal RAG Bench Architecture Comparison
```

### Step 4: Import Knowledge Base

Create a Knowledge Base from:

```text
data/legal_rag_bench/full/raw
```

Expected:

- 4,876 documents.
- no accidental duplicate uploads.
- stable KB version.

### Step 5: Import Test Set

Import:

```text
data/legal_rag_bench/full/test_set.json
```

Expected:

- 100 test cases.
- each test case keeps `relevant_passage_id`.
- each test case keeps `ground_truth_context`.

If the UI cannot import this format yet, add a compatible import path before
continuing.

### Step 6: Create three RAG configs

Create one RAG Config per architecture.

Shared values:

```yaml
embedding_model: text-embedding-3-large
embedding_dimension: 3072
llm_model: gpt-4o-mini
temperature: 0
chunk_size: 8000
chunk_overlap: 0
```

Configs:

1. `LegalBench - Chroma semantic`
   - `rag_type`: `vector_semantic`
   - vector `top_k` default/query override: `5`

2. `LegalBench - Qdrant hybrid`
   - `rag_type`: `vector_hybrid`
   - vector `top_k` default/query override: `5`
   - Qdrant URL configured.
   - dense vector dimension must match Text Embedding 3 Large.

3. `LegalBench - Filesystem`
   - `rag_type`: `filesystem_rag`
   - fixed `max_tool_calls`.
   - fixed `max_file_reads`.
   - report these as retrieval budget.

### Step 7: Build three indexes from the UI

From the Knowledge Base page, create indexes for each RAG Config.

After each build, verify:

- status = ready;
- document count = 4,876;
- vector chunk count = 4,876 for Chroma/Qdrant;
- config snapshot is correct;
- embedding model is Text Embedding 3 Large;
- index physical id is unique.

### Step 8: Run calibration evaluation

Start with Chroma only.

Evaluation settings:

- index: `LegalBench - Chroma semantic`;
- test set: Legal RAG Bench full;
- query override: `top_k=5`;
- judge: selected Phase 1 judge;
- metrics:
  - `legal_rag_binary_judge`,
  - `legal_rag_retrieval`,
  - optional DeepEval diagnostics.

Check:

- Chroma `hit@5` is in the same broad range as the paper's 52.0% Text
  Embedding 3 Large retrieval accuracy.
- If wildly off, debug before running Qdrant/filesystem.

### Step 9: Run full architecture evaluations

Run one evaluation per ready index:

1. Chroma semantic.
2. Qdrant hybrid.
3. Filesystem RAG.

Keep fixed:

- same Knowledge Base version;
- same Test Set;
- same generator;
- same judge;
- same vector `top_k=5`;
- same filesystem retrieval budget.

### Step 10: Compare in UI

Create a UI comparison across the three completed evaluations.

Article-ready outputs:

- headline metrics table;
- taxonomy breakdown;
- per-question disagreement table;
- retrieval trace examples;
- run manifest/config snapshot export;
- CSV/Markdown export.

## 9. Experiment Phases

### Phase 0: UI smoke

Purpose: prove the UI path works end to end.

Scope:

- subset only, if subset import exists;
- one project;
- one KB;
- one test set;
- one Chroma config;
- one Chroma index;
- one evaluation;
- cheap generator and judge.

Exit criteria:

- index builds from UI;
- evaluation runs from UI;
- retrieved context and trace are visible;
- `hit@5` or equivalent benchmark retrieval metric is computed;
- result export works.

### Phase 1: cheap UI architecture comparison

Purpose: get directionally useful findings for the article draft.

Scope:

- full 100 questions;
- full 4,876 passages;
- systems: `vector_semantic`, `vector_hybrid`, `filesystem_rag`;
- embedding: `text-embedding-3-large`;
- generator: `gpt-4o-mini` or another explicitly named cheap model;
- judge: `gpt-4o` or another explicitly named cheap judge;
- vector retrieval: `k=5`;
- filesystem budget: fixed and reported.

Report:

- `hit@5` for Chroma and Qdrant.
- `gold_accessed` for filesystem.
- correctness and groundedness as indicative.
- taxonomy breakdown.
- latency and rough cost.

Do not compare Phase 1 correctness/groundedness directly to the paper's GPT-5.2
judge numbers.

### Phase 2: premium UI run

Purpose: produce final publishable numbers if Phase 1 is interesting.

Scope:

- same Knowledge Base version;
- same Test Set;
- same three ready indexes or rebuilt indexes with identical configs;
- stronger generator and judge;
- ideally closest available model family to the paper's GPT-5.2 judge setup.

Report Phase 1 vs Phase 2 deltas. If the cheap judge changes the conclusion,
that becomes an article finding.

## 10. Backend Implementation Notes

### Keep benchmark logic close to platform evaluation

Prefer adding reusable backend services over a standalone CLI-only runner.

Suggested service/module split:

```text
platform/backend/app/services/legal_rag_bench_metrics.py
platform/backend/app/services/legal_rag_bench_judge.py
platform/backend/app/services/evaluation_exporter.py
```

Responsibilities:

- extract passage ids from retrieved sources/context;
- compute `hit@5` / `gold_accessed`;
- run binary judge;
- derive taxonomy;
- aggregate summaries;
- export CSV/Markdown/JSONL.

The CLI can import the same service later if needed, but the UI is the primary
consumer for this article.

### Store benchmark fields without breaking generic evaluations

Add benchmark-specific data in flexible places where possible:

- raw metrics artifact,
- retrieved context artifact,
- retrieval trace artifact,
- evaluation result metadata,
- summary metrics JSON.

If first-class columns are useful later, add migrations after the schema is
stable.

### Preserve reproducibility

Every UI evaluation used in the article must expose:

- run manifest;
- build config snapshot;
- effective query config;
- judge model/provider;
- generation model/provider;
- KB version;
- Test Set id/version;
- exact timestamp.

## 11. Article Structure

### Short LinkedIn version

1. Hook: Legal RAG performance is mostly retrieval-limited.
2. Setup: same benchmark, same UI workflow, three architectures.
3. Main table: Chroma vs Qdrant Hybrid vs Filesystem.
4. One screenshot of UI comparison or trace.
5. One surprising example where architecture changed the answer.
6. Caveat: not a paper replication; paper used different model/judge setup.
7. Link to full write-up and repo/codebase.

### Full site article

Suggested title:

> Legal RAG Bench in a real evaluation UI: Chroma, Qdrant Hybrid, and agentic
> file search compared

Sections:

1. Why Legal RAG Bench is a good stress test.
2. What the original paper found.
3. Why I used a UI workflow instead of one-off scripts.
4. What this codebase adds: interchangeable RAG configs, isolated indexes,
   traces, comparisons, manifests.
5. Experimental setup and comparability table.
6. Retrieval results first.
7. Correctness, groundedness, and taxonomy.
8. Qualitative examples from retrieval traces.
9. Cost, latency, and operational tradeoffs.
10. What I would improve next.

## 12. Reporting Tables

Headline table:

| System | Retrieval mode | Retrieval metric | Retrieval score | Correct | Grounded | RAG accuracy | Avg latency | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Chroma semantic | dense vector | hit@5 | TBD | TBD | TBD | TBD | TBD | calibration anchor |
| Qdrant hybrid | dense + sparse | hit@5 | TBD | TBD | TBD | TBD | TBD | architecture extension |
| Filesystem RAG | agentic file search | gold_accessed | TBD | TBD | TBD | TBD | TBD | not k-comparable |

Taxonomy table:

| System | Success | Hallucination/ungrounded | Retrieval error | Reasoning error |
|---|---:|---:|---:|---:|
| Chroma semantic | TBD | TBD | TBD | TBD |
| Qdrant hybrid | TBD | TBD | TBD | TBD |
| Filesystem RAG | TBD | TBD | TBD | TBD |

UI artifact table:

| Artifact | Purpose |
|---|---|
| Evaluation manifest | Reproducibility |
| Retrieval trace | Qualitative examples |
| Per-question JSONL | Auditability |
| Comparison CSV/Markdown | Article tables |
| Config snapshot | Shows fixed build/query settings |

Comparability caveat table:

| Claim | Safe? | Wording |
|---|---|---|
| "We reproduced the paper" | No | Avoid |
| "We calibrated Chroma against the paper's Text Embedding 3 Large row" | Yes, if hit@5 is close | Use with caveat |
| "Hybrid beats/loses to Chroma on this benchmark" | Yes, if same embedding/generator/judge | Use |
| "Filesystem hit@5 is X" | No | Use `gold_accessed` instead |
| "DeepEval correctness equals paper correctness" | No | Call it indicative |
| "The UI can run reproducible RAG experiments" | Yes, after benchmark fields/export are added | Use |

## 13. Final Checklist

- [ ] Add converter `--content-mode clean|annotated`.
- [ ] Ensure UI can import 4,876 Legal RAG passage files as a Knowledge Base.
- [ ] Ensure UI can import 100 Legal RAG QA cases with `relevant_passage_id`.
- [ ] Expose/preserve `text-embedding-3-large` in RAG Config UI.
- [ ] Fix Qdrant vector dimension for `text-embedding-3-large`.
- [ ] Expose/preserve `chunk_size=8000` and `chunk_overlap=0` in RAG Config UI.
- [ ] Expose filesystem retrieval budget settings in RAG Config UI.
- [ ] Verify vector indexes show chunk count = 4,876.
- [ ] Add Legal RAG retrieval metric: `hit@5` for vector systems.
- [ ] Add Legal RAG filesystem metric: `gold_accessed`.
- [ ] Add binary Legal RAG judge metric.
- [ ] Add taxonomy derivation.
- [ ] Display benchmark fields in Evaluation Results UI.
- [x] Add comparison support for benchmark metrics.
- [x] Add CSV/Markdown/JSONL export for article artifacts.
- [ ] Run Phase 0 UI smoke.
- [ ] Run Chroma calibration and compare broad range to paper's 52.0% hit@5.
- [ ] Run Phase 1 full UI comparison.
- [ ] Decide whether Phase 2 premium UI run is worth the cost.
- [ ] Publish with explicit caveats and exact model/date/config metadata.
