# Implementation Plan: Google Vertex AI Search RAG (`google_vertex_search`)

**Status:** Proposed
**Date:** 2026-06-15
**Branch:** `claude/google-vector-search-rag-ywrudk`

## 1. Goal

Add a new RAG type that uses **Google Vertex AI Search** (a managed Data Store) as
the retrieval backend, fully integrated with the existing RAG Evaluator framework:

- **Indexing:** create a Vertex AI Search Data Store, import the project's documents,
  and let Google parse/chunk/embed them.
- **Retrieve + Evaluate:** query the Data Store for chunks and run the standard
  DeepEval evaluation pipeline.
- **Bring-your-own Data Store:** support pointing the RAG at an *existing* Data Store
  (eval-only) without re-indexing.
- **Frontend:** surface the new type in the platform web UI.

---

## 2. Critical Background — Disambiguating Google's Naming

This is the trap called out in the request. Google ships several overlapping products
with confusingly similar names. **This plan targets the first one only.**

| Product | Storage concept | Indexing | Retrieval | Python lib |
|---|---|---|---|---|
| **Vertex AI Search** *(formerly Discovery Engine / Agent Builder / Gen App Builder / Search & Conversation)* | **Data Store** ✅ | Import docs → Google parses, chunks, embeds automatically | `SearchService.search` (managed; can return chunks or grounded answers) | `google-cloud-discoveryengine` |
| Vertex AI **Vector Search** *(formerly Matching Engine)* | Index / Index Endpoint (v2.0: "Collections") | You generate & manage embeddings yourself | `find_neighbors` (raw ANN) | `google-cloud-aiplatform` |
| Vertex AI **RAG Engine** | `RagCorpus` | Managed, mid-level | `retrieval_query` | `vertexai` (preview) |
| Google **Cloud Datastore / Firestore in Datastore mode** | NoSQL "kind"/entities | N/A | N/A | `google-cloud-datastore` — **unrelated**, ignore |

**Decision (confirmed with requester):** Target **Vertex AI Search (Data Stores)**.
The "datastore" terminology and the "add an existing datastore for evaluation"
requirement map exactly to this product. Note the irony: the product literally named
"Vector Search" is *not* the one we want.

### Confirmed design decisions

1. **Product:** Vertex AI Search (Data Stores) via `google-cloud-discoveryengine`.
2. **Generation:** *Configurable*, defaulting to the **existing framework LLM pipeline**
   (`BaseRAG.generate` using the configured OpenAI/compatible model) so retrieval
   comparisons stay apples-to-apples across RAG types. Optional flag to instead use
   Google's native grounded Answer API.
3. **Auth:** Support **both** Application Default Credentials (ADC) and an explicit
   service-account JSON key path.

---

## 3. How Vertex AI Search Works (the bits that matter here)

- A **Data Store** is the container of documents. For our use case it is an
  **unstructured** data store with **document chunking enabled** (set via
  `documentProcessingConfig` at creation time — required to retrieve chunks).
- **Importing documents** (`DocumentService.import_documents`) accepts:
  - **GCS** (`gcs_source`) — the standard path for unstructured files (PDF/HTML/TXT/…).
  - **BigQuery** (`bigquery_source`).
  - **Inline** documents (`inline_source`) — useful for small/structured payloads.
  Import is a **long-running operation (LRO)**; Google parses, chunks, and embeds
  asynchronously.
- **Retrieval** uses `SearchService.search` against a **serving config**. To get chunks
  back, set `ContentSearchSpec.search_result_mode = CHUNKS` (requires chunking enabled
  on the data store). `num_previous_chunks` / `num_next_chunks` (max 3 each) expand
  context around each hit.
- A data store can be searched directly via its **default serving config**; an
  "Engine"/"App" is not strictly required for the search API path we use.

### Key implication for indexing

Vertex AI Search imports unstructured docs from **GCS**, not from the local disk. Our
framework loads documents from `data/raw`. Therefore the indexing flow must **stage
local files into a GCS bucket**, then call `import_documents` with `gcs_source`. This
plan adds a configurable staging bucket. (Inline import is the fallback for small text
payloads but does not support native PDF/HTML parsing.)

---

## 4. Architecture Fit

The framework already gives us clean seams (verified in code):

- `BaseRAG` (`src/rag_evaluator/common/base_rag.py`) — abstract
  `prepare_documents`, `query`, `get_metrics`; overridable `retrieve`, `generate`,
  `load_index`, `prepare_documents_resumable`.
- Registry (`src/rag_evaluator/rag_implementations/registry.py`) — three dicts to
  extend: `_RAG_CLASS_PATHS`, `RAG_TYPES`, `RAG_TYPE_PARAMETERS`.
- CLI (`src/rag_evaluator/cli.py`) — `--rag-type` choices auto-derive from `RAG_TYPES`.
- Config (`src/rag_evaluator/config.py`) — pydantic `Settings` loaded from root `.env`.
- Evaluation (`src/rag_evaluator/evaluation/evaluator.py`) — RAG-agnostic; calls
  `query()` and expects `{"answer", "context", "metadata"}`.

We implement `retrieve()` (Google search) + `generate()` (configurable) and let the
base `query()` compose them, matching the pattern used by `vector_semantic`.

---

## 5. New Module Layout

```
src/rag_evaluator/rag_implementations/google_vertex_search/
├── __init__.py
├── google_vertex_rag.py      # GoogleVertexSearchRAG(BaseRAG)
├── client.py                 # auth + DiscoveryEngine client factory (ADC / SA key)
└── gcs_staging.py            # stage local docs -> GCS for import
```

**RAG type key:** `google_vertex_search`
**Display name:** "Google Vertex AI Search"
**Class:** `GoogleVertexSearchRAG`

---

## 6. Implementation Detail — `GoogleVertexSearchRAG`

### 6.1 Construction / auth (`client.py`)

- Resolve credentials in this order:
  1. Explicit service-account JSON path (`GOOGLE_VERTEX_SA_KEY_PATH` /
     `google.oauth2.service_account.Credentials.from_service_account_file`).
  2. Otherwise ADC (`GOOGLE_APPLICATION_CREDENTIALS` env or `gcloud auth
     application-default login`).
- Build clients with the correct **regional endpoint** (e.g. `global`, `us`, `eu`):
  `DataStoreServiceClient`, `DocumentServiceClient`, `SearchServiceClient` (and
  optionally `ConversationalSearchServiceClient` for grounded answers).
- Fail fast with a clear error if `project_id` / `location` are unset.

### 6.2 `prepare_documents(documents_path)` — indexing

1. **Resolve mode:**
   - If `data_store_id` is provided **and** `reuse_existing_data_store=True`, skip
     creation/import (eval-only). Validate the data store exists.
   - Else create/get a data store (deterministic id from `data_store_id` or generated).
2. **Create data store** (if needed) via `DataStoreServiceClient.create_data_store`:
   - `industry_vertical=GENERIC`, `solution_types=[SOLUTION_TYPE_SEARCH]`,
     `content_config=CONTENT_REQUIRED` (unstructured).
   - `document_processing_config` with **chunking enabled** (layout-aware; OCR parser
     for PDFs recommended).
3. **Stage docs to GCS** (`gcs_staging.py`): reuse the framework's document discovery
   (`discover_source_documents` / `create_loader`) to enumerate files, upload to
   `gs://<staging_bucket>/<data_store_id>/...`.
4. **Import** via `DocumentService.import_documents` with `gcs_source` and
   `reconciliation_mode=INCREMENTAL`; **poll the LRO** to completion, surfacing
   progress through `self._report_progress(...)`.
5. **Checkpointing:** override `prepare_documents_resumable` to persist
   `{data_store_id, gcs_prefix, imported_uris, import_operation_name}` so a re-run can
   resume polling / skip already-uploaded files.
6. **Metrics:** record document/chunk counts and import duration.

> Note: Google charges for storage + import; embeddings are managed (no OpenAI
> embedding tokens consumed during indexing for this RAG type).

### 6.3 `retrieve(question, top_k)` — retrieval

1. Build `SearchRequest` against the data store's default serving config:
   - `query=question`, `page_size=top_k`.
   - `ContentSearchSpec(search_result_mode=CHUNKS,
     chunk_spec=ChunkSpec(num_previous_chunks=N, num_next_chunks=N))`.
2. Map each returned chunk → `RetrievedChunk` (content, score/relevance, rank, source
   URI/title from `document_metadata`).
3. Build a `RetrievalTrace(strategy="vector")` with a `vertex_search` step (query,
   data store id, latency, result count) for parity with other RAG types.
4. Return `RetrievedContext(chunks, chunk_details, trace, retrieval_time)`.

Edge cases: empty results → empty context (let DeepEval score it); chunking disabled on
a BYO data store → fall back to `DOCUMENTS` mode and use snippets/derived answers, with
a logged warning.

### 6.4 `generate(question, context)` — generation (configurable)

- **Default (`generation_mode="framework"`):** reuse the standard generation path
  (configured LLM via the base class / `llm_client`), feeding the retrieved chunks as
  context — identical to `vector_semantic`. Tracks prompt/completion tokens.
- **Optional (`generation_mode="google_grounded"`):** call Vertex AI Search's grounded
  Answer API (`ConversationalSearchService` / `:answer`) so retrieval **and** generation
  are Google-native. Token accounting is best-effort (Google-side).
- `query()` = `retrieve()` + `generate()` (inherited composition), returning
  `{"answer", "context", "metadata"}`.

### 6.5 `get_metrics()`

Return retrieval latency, result counts, token usage (framework generation), data store
id, and whether the data store was reused vs created.

---

## 7. Configuration

### 7.1 `config.py` additions (`Settings`)

```python
# Google Vertex AI Search (Discovery Engine)
google_vertex_project_id: str = Field(default="")
google_vertex_location: str = Field(default="global")          # global | us | eu
google_vertex_sa_key_path: str | None = Field(default=None)    # optional SA JSON; else ADC
google_vertex_data_store_id: str = Field(default="")           # set to reuse existing
google_vertex_staging_bucket: str = Field(default="")          # gs bucket for import
google_vertex_generation_mode: str = Field(default="framework")  # framework | google_grounded
```

### 7.2 `.env.example` additions

```env
GOOGLE_VERTEX_PROJECT_ID=your-gcp-project
GOOGLE_VERTEX_LOCATION=global
# Auth: leave SA key empty to use Application Default Credentials (ADC)
GOOGLE_VERTEX_SA_KEY_PATH=
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/adc.json   # alternative ADC path
GOOGLE_VERTEX_DATA_STORE_ID=        # set to reuse an existing data store (eval-only)
GOOGLE_VERTEX_STAGING_BUCKET=your-staging-bucket
GOOGLE_VERTEX_GENERATION_MODE=framework
```

### 7.3 Registry parameter schema (`RAG_TYPE_PARAMETERS["google_vertex_search"]`)

| Param | Phase | Notes |
|---|---|---|
| `data_store_id` | build | platform_managed; if set + `reuse_existing` → eval-only |
| `reuse_existing_data_store` | build | bool; skip create/import |
| `location` | build | data store region |
| `staging_bucket` | build | GCS bucket for import (when indexing) |
| `num_previous_chunks` | query | 0–3 context expansion |
| `num_next_chunks` | query | 0–3 context expansion |
| `generation_mode` | query | `framework` (default) \| `google_grounded` |

`top_k` continues to come from the standard query path.

---

## 8. Dependencies

Add an **optional extra** (GCP libs are heavy; keep core install lean — matches the
`uv sync --all-extras` story):

```toml
[project.optional-dependencies]
google-vertex = [
    "google-cloud-discoveryengine>=0.13",
    "google-cloud-storage>=2.16",      # GCS staging for import
    "google-auth>=2.30",
]
```

Use **lazy imports** inside the implementation (like the DeepEval pattern) and raise a
friendly "install `uv sync --extra google-vertex`" error if missing, so the registry can
still be imported without the libs installed.

---

## 9. Registry + CLI Wiring

- `_RAG_CLASS_PATHS["google_vertex_search"] = "...google_vertex_search.google_vertex_rag.GoogleVertexSearchRAG"`
- `RAG_TYPES["google_vertex_search"] = {"name": "Google Vertex AI Search", "description": "Managed Google Vertex AI Search data store (Discovery Engine) with automatic parsing, chunking, and embedding"}`
- `RAG_TYPE_PARAMETERS["google_vertex_search"] = {...}` (table in §7.3).
- CLI `--rag-type` choices update automatically. Add epilog examples:

```powershell
# Index into a new Vertex AI Search data store
uv run rag-eval prepare --rag-type google_vertex_search --input-dir data/raw

# Evaluate (new or existing data store via .env / config)
uv run rag-eval evaluate --rag-type google_vertex_search
```

---

## 10. Platform Backend (FastAPI)

The backend exposes RAG types from the same registry, so the type appears in
`GET /api/v1/rag-types` automatically. Work needed:

- Confirm parameter schema flows through `GET /rag-types/{type}/parameters`.
- Ensure RAG config validation accepts the new build/query params.
- Surface GCP config (project, location, auth, staging bucket, data store id) —
  prefer environment variables for secrets; data-store id can be a per-config field.
- "Existing data store" path: a RAG config with `reuse_existing_data_store=true` +
  `data_store_id` should be indexable as a no-op "ready" index (or skip the index step
  entirely) so evaluations can run immediately.
- No Alembic migration expected unless we add columns to models (the existing
  parameters JSON should suffice — confirm during implementation).

---

## 11. Frontend (Web UI)

> **Correction:** the request mentions "Next.js", but the platform frontend is
> **React + Vite + Tailwind** (`platform/frontend`, React Router v6, TanStack Query,
> Axios). There is no Next.js app. Plan targets the actual React/Vite frontend. (The
> Streamlit/legacy CLI UI is intentionally left untouched per the request.)

The UI is **backend-driven**: RAG types and their parameter schemas come from
`GET /api/v1/rag-types`, so dropdowns/forms auto-populate. Only hardcoded
cosmetics and the "existing data store" affordance need edits:

1. **`src/components/rag-configs/RAGConfigList.tsx`** — add icon case for
   `google_vertex_search` (e.g. a `Cloud` icon).
2. **`src/components/playground/ResultCard.tsx`** and
   **`src/components/playground/IndexSelector.tsx`** — add color + label for
   `google_vertex_search` in the (currently duplicated) `ragTypeColors` /
   `ragTypeLabels` maps. *Recommend* extracting these to a shared
   `src/lib/rag-types.ts` while here.
3. **`src/components/rag-configs/RAGConfigDialog.tsx`** — parameter fields render from
   the backend schema automatically; verify the new params display sensibly. Add a
   small "Use existing data store" toggle that maps to
   `reuse_existing_data_store` + `data_store_id` if richer UX is desired.
4. **`src/api/client.ts`** — no required change (types are string-keyed); optionally add
   the new key to any TS literal unions/comments.
5. Run `npm run lint` (zero-warning policy) after edits.

---

## 12. Gotchas / Risks

- **GCS dependency for import:** unstructured import requires staging to GCS; needs a
  bucket + `storage.objects.create` permission. Documented as a prerequisite.
- **Async import LRO:** indexing is not instantaneous; the resumable checkpoint must
  store the operation name so re-runs poll rather than re-import.
- **Chunking must be enabled at creation** to use `CHUNKS` retrieval mode — cannot be
  toggled later. BYO data stores without chunking fall back to `DOCUMENTS` mode.
- **Region/endpoint correctness:** the client endpoint must match the data store
  location (`global`/`us`/`eu`), a common source of `NOT_FOUND` errors.
- **IAM:** roles `discoveryengine.editor` (index) / `discoveryengine.viewer` (query)
  plus storage access for the staging bucket.
- **Cost:** managed storage/query billing; no local embedding token cost on this path.
- **Quotas:** import and search QPS quotas can throttle large eval runs.

---

## 13. Testing

- **Unit (mocked clients):** auth resolution (ADC vs SA key), request construction
  (create_data_store, import_documents, search with CHUNKS), response→`RetrievedContext`
  mapping, eval-only/reuse branch. No network calls in CI.
- **`get_rag_class("google_vertex_search")`** resolves and the registry stays importable
  without the optional extra installed.
- **Integration (manual / gated):** against a real GCP project — index a small corpus,
  retrieve, evaluate; plus a "reuse existing data store" run.
- **Quality gates:** `uv run ruff check .`, `uv run mypy src/rag_evaluator`,
  `uv run pytest`; backend `pytest`/`ruff`; frontend `npm run lint`.

---

## 14. Step-by-Step Task Checklist

1. [ ] Add `google-vertex` optional extra to `pyproject.toml`.
2. [ ] `config.py` + `.env.example`: GCP settings.
3. [ ] `client.py`: auth (ADC + SA key) and DiscoveryEngine client factory.
4. [ ] `gcs_staging.py`: stage local docs → GCS.
5. [ ] `google_vertex_rag.py`: `GoogleVertexSearchRAG` (prepare / resumable / retrieve /
       generate / get_metrics), with reuse-existing branch.
6. [ ] Register in `registry.py` (3 dicts) + CLI epilog examples.
7. [ ] Backend: verify type exposure, params, and eval-only/no-op index path.
8. [ ] Frontend: icon/color/label + optional "existing data store" toggle; lint.
9. [ ] Tests (mocked unit + registry) and docs.
10. [ ] Docs: update `docs/cli.md`, `docs/rag_strategies.md`, `README.md` RAG list.

---

## 15. Open Questions / Future

- Should "existing data store" be a first-class UI flow (browse the project's data
  stores via the API) or just an id field? (Plan assumes id field + toggle for v1.)
- Support **structured**/BigQuery data stores later? (v1 = unstructured only.)
- Optionally expose an "Engine/App" path for advanced serving configs and grounded
  answers with citations.

---

## 16. References

- [The GCP RAG Spectrum: Vertex AI Search, RAG Engine, and Vector Search](https://medium.com/google-cloud/the-gcp-rag-spectrum-vertex-ai-search-rag-engine-and-vector-search-which-one-should-you-use-f56d50720d5a)
- [Vertex AI Search vs Vertex AI Vector Search](https://medium.com/@saichandra2520/vertex-ai-search-vs-vertex-ai-vector-search-whats-the-actual-difference-5038213b88ac)
- [Vertex AI APIs for building search and RAG experiences](https://docs.cloud.google.com/generative-ai-app-builder/docs/builder-apis)
- [Create a search data store](https://docs.cloud.google.com/generative-ai-app-builder/docs/create-data-store-es)
- [Parse and chunk documents](https://cloud.google.com/generative-ai-app-builder/docs/parse-chunk-documents)
- [ContentSearchSpec (chunk mode)](https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/ContentSearchSpec)
- [DataStoreServiceClient (Python)](https://docs.cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.services.data_store_service.DataStoreServiceClient)
- [DocumentServiceClient (Python)](https://docs.cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1beta.services.document_service.DocumentServiceClient)
- [RAG and grounding on Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/rag-and-grounding-on-vertex-ai)
</content>
</invoke>
