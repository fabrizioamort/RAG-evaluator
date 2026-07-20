# RLM-RAG: indexing and retrieval internals

RLM-RAG is an RLM-inspired, agentic retrieval strategy for exploring a prepared document
filesystem with generated Python. A main **orchestrator model** writes exploration code;
a smaller **worker model** can prepare summaries/topics and answer delegated sub-tasks.
For small corpora, the implementation bypasses the code loop and uses a direct-context
fallback.

This document describes the current implementation, including important gaps between
configured controls and runtime behavior. For the cross-strategy overview, see
[RAG Strategies](../../../../docs/rag_strategies.md#rlm-rag).

> **Security warning:** generated Python execution is not a hardened sandbox. The
> current `full` mode provides process termination for executed snippets but does not
> provide a complete filesystem exploration environment or wire in the documented
> prompt-injection wrapper. Do not use either mode to execute model-generated code in a
> security boundary that contains untrusted data or credentials. See
> [Security status and limitations](#6-security-status-and-limitations).

## 1. Architectural summary

RLM-RAG has one preparation path and two query paths:

```text
.txt/.md/.pdf/.docx sources (top-level files only)
        |
        v
text extraction -> summary -> topics -> catalog/section/topic indexes
        |
        +---------------- document count <= threshold ----------------+
        |                                                             |
        v                                                             v
RLM agent mode                                               Simple-context mode
orchestrator writes Python                                  first top_k catalog docs
        |                                                             |
        v                                                             v
REPL + fs.* tools + optional worker calls                    one generation request
        |                                                             |
        +------------------------- answer ------------------------------+
```

No embeddings, vector store, or ranked lexical index are built. In agent mode, retrieval
is an iterative program written by the orchestrator. In simple mode, selection is catalog
order rather than relevance ranking.

Main implementation files:

- [`rlm_rag.py`](rlm_rag.py): configuration, preparation/loading, mode routing, public
  query API, and trace adaptation.
- [`preparation.py`](preparation.py): source processing, generated artifacts, manifest,
  and simple-context fallback.
- [`agent.py`](agent.py): budgets, filesystem tools, in-process REPL, and agent loop.
- [`llm_client.py`](llm_client.py): orchestrator/worker calls, retries, circuit breaker,
  and response cache.
- [`security.py`](security.py): experimental process REPL and currently unwired injection
  guard.
- [`prompts.py`](prompts.py): generated-code protocol and exploration guidance.

## 2. Prepared filesystem contract

The default programmatic location is `<RAGConfig.storage_path>/rlm_rag`; with default
`RAGConfig` this resolves to `data/indexes/rlm_rag`. The web platform assigns an isolated
path under `storage/indexes/<physical_id>/rlm_rag`.

Preparation creates:

```text
prepared_path/
├── manifest.json
├── _meta/
│   ├── catalog.json
│   └── section_index.json
├── _index/
│   └── topics/
│       └── _topic_map.json
├── _summaries/
│   └── <doc_id>_summary.md
└── documents/
    └── <doc_id>.md
```

Unlike Filesystem RAG, the current RLM-RAG preparation does not build entity, temporal,
question-seed, or BM25 indexes, and it does not preserve original files.

## 3. Indexing logic

### 3.1 Source discovery and identifiers

`DocumentProcessor.prepare()` inspects only direct children of the input directory with
`input_dir.glob("*")`; it is **not recursive**. Supported extensions are:

- `.txt`
- `.md`
- `.pdf`
- `.docx`

The document ID is the filename stem, and the output is always
`documents/<stem>.md`. Filenames with the same stem but different extensions therefore
collide in the prepared directory and should be avoided.

Source loading is format-specific:

- TXT/Markdown: UTF-8 text is read directly.
- PDF: `pypdf.PdfReader` extracts page text and joins pages with blank lines.
- DOCX: `python-docx` extracts non-empty paragraphs and joins them with blank lines.

Loading failures are logged, counted, and skipped. PDF/DOCX extraction does not perform
layout reconstruction or convert source styles into Markdown headings. The `.md` output
extension means “prepared text file”; it does not guarantee normalized Markdown.

### 3.2 Document summary

When `use_llm_summaries=True` (default), the worker model receives at most the first
8,000 characters and is asked for a retrieval-oriented Markdown summary under 300 words
covering purpose, concepts, entities, and document type. Provider-reported tokens are
recorded as preparation metrics.

If summaries are disabled or the LLM call fails, the fallback contains:

- up to ten lines that already begin with `#`;
- a preview of the first 500 characters.

The summary is written to `_summaries/<doc_id>_summary.md`.

### 3.3 Section index

The processor scans prepared text line by line. Every line beginning with `#` starts a
section. For each section, `_meta/section_index.json` records:

- heading title;
- heading depth;
- start line;
- end line.

The stored section offsets are currently **zero-based**, while runtime `fs.read_file()`
and `fs.read_document()` line arguments are documented and implemented as 1-indexed.
Callers using section metadata directly must account for this mismatch. Documents with
no Markdown headings have an empty section list.

### 3.4 Topic extraction

When `use_llm_topics=True` (default), the worker model receives the first 4,000 characters
and is asked to return a JSON array of up to `max_topics_per_doc` lowercase topics. Code
fences are stripped before JSON parsing.

If topic generation is disabled, malformed, or fails, the fallback:

1. extracts lowercase alphabetic words of at least four characters;
2. removes a small English stop-word list;
3. returns the most frequent terms.

Topics are normalized to lowercase for `_index/topics/_topic_map.json`, which maps each
topic to document IDs. The per-document catalog retains the analyzer's returned strings.

### 3.5 Catalog

`_meta/catalog.json` contains one object per successfully processed source:

- `id`, inferred display `title`, document path, and summary path;
- topics;
- section, line, word, and character counts.

The catalog is the primary discovery artifact used by both query modes. It does not
contain chunks or retrieval scores.

### 3.6 Chunking configuration: current behavior

`RLMConfig` exposes `chunk_size` and `chunk_overlap`, and both values participate in
manifest invalidation. The current `DocumentProcessor`, however, writes one file per
source and does **not** chunk document content. Consequently:

- changing either value forces re-preparation;
- neither value currently changes the generated documents, summaries, catalog, sections,
  topics, or query behavior;
- `total_chunks` in metrics is effectively one per catalog document because catalog
  entries do not contain `chunk_count`.

These parameters should be treated as reserved/incomplete until actual chunk creation is
implemented.

### 3.7 Manifest and cache invalidation

`manifest.json` stores:

- creation/update timestamps;
- source document count;
- for each supported top-level source: suffix, byte size, and the first 16 hex characters
  of its SHA-256 digest;
- a hash of preparation-affecting configuration.

The configuration hash includes `chunk_size`, `chunk_overlap`, `use_llm_summaries`,
`use_llm_topics`, `max_topics_per_doc`, and `worker_model`.

On `prepare_documents(force=False)`, preparation is skipped only when the source set,
content hashes, and configuration hash all match. `force=True` bypasses this check.

Preparation creates/overwrites current artifacts but does not clear the output directory
first. If sources are removed or processing fails, old files can remain under
`documents/` or `_summaries/`. They disappear from the newly written catalog/topic map,
but agent-mode `fs.grep()` scans physical Markdown files and can still find stale files.
Use an isolated platform index or remove the prepared directory before a clean rebuild
when source deletion matters.

The resumable wrapper is not incremental: it runs normal preparation, then marks a source
complete when its document and summary files exist.

## 4. Mode selection

After preparation or `load_index()`, `_load_catalog_and_route()` compares the catalog's
document count with `small_corpus_threshold` (default `10`):

```text
doc_count <= small_corpus_threshold -> simple_context
doc_count >  small_corpus_threshold -> rlm_agent
```

This is a document-count threshold, not a token or character threshold. Ten very large
documents still select simple mode, while eleven tiny documents select agent mode.

Changing `small_corpus_threshold` is query-time configuration in the platform and can
change the retrieval algorithm used against the same prepared index.

## 5. Query and retrieval logic

### 5.1 Simple-context mode

Simple mode loads every catalog-listed document into memory at initialization. For each
question it then:

1. selects the **first** `top_k` documents in catalog order;
2. truncates each selected document to 3,000 characters;
3. concatenates them into one prompt;
4. asks the orchestrator model for an answer with `[doc_id]` citations.

There is no query-dependent ranking, topic filtering, lexical search, or agent loop in
this mode. `top_k` is meaningful here and defaults to five. Because the catalog is built
from sorted source filenames, renaming files can change selected context.

Returned context contains the first 500 characters of each selected full document, while
the model may have seen up to 3,000. Metadata reports `retrieval_time=0.0`, generation
time, selected source IDs, and measured token usage.

### 5.2 Agent-mode initialization

Agent mode loads three JSON artifacts into memory:

- catalog;
- section index;
- topic map.

It constructs a corpus overview for the system prompt using:

- document and topic counts;
- the ten topics linked to the most documents;
- metadata for at most the first 20 catalog documents.

The prompt instructs the orchestrator to emit fenced `python` blocks and eventually set
three variables:

```python
final_answer = "..."
confidence = "HIGH"  # or MEDIUM/LOW
sources_used = ["doc_a", "doc_b"]
```

### 5.3 Orchestrator/REPL loop

Each loop turn consumes one `max_repl_steps` unit (default `15`):

1. Send the complete conversation to the orchestrator model.
2. Extract fenced blocks labeled with lowercase `python`; the extractor also requires
   newlines immediately before and after the code body.
3. Execute every extracted block in the selected REPL.
4. Return output/errors, newly created variable names, and low-budget warnings to the
   orchestrator.
5. Stop when `final_answer` exists or the turn budget is exhausted.

One orchestrator response can contain multiple code blocks; all are executed inside the
same loop turn. If no code is emitted, the agent asks for Python and continues. If the
budget expires without `final_answer`, one final orchestrator call synthesizes a
best-effort answer from the accumulated conversation.

Queries on one `RLMFilesystemRAG` instance are serialized with a lock because the budget,
REPL namespace, and tracking state are mutable. The streaming path is separate and
should not be assumed to provide the same serialization guarantees.

### 5.4 In-process REPL (`lite`)

`SimpleREPL` preserves its namespace across turns, allowing generated code to build lists,
dictionaries, intermediate results, and a final answer incrementally. It exposes:

- `fs`: read-only helper object for the prepared filesystem;
- `call_sub_llm`: worker-model delegation;
- `budget`: a snapshot of remaining configured resources;
- `show()` and captured `print()`;
- common types/functions plus preloaded `re`, `json`, and `math`.

The last standalone expression is evaluated again and displayed automatically. Output is
capped at 10,000 characters per execution; auto-displayed collections are capped more
aggressively.

Although the prompt says imports, file writes, network access, `eval`, and `exec` are not
allowed, this is primarily an instruction to the model, not a complete runtime sandbox.
Generated code is compiled and passed to Python `exec` in the application process. Treat
`lite` as trusted-development functionality only.

### 5.5 Filesystem exploration API

In lite agent mode the orchestrator can use:

| API | Behavior |
| --- | --- |
| `fs.get_catalog()` | Returns catalog document metadata. |
| `fs.get_topics()` | Returns topic-to-document mappings. |
| `fs.get_sections(doc_id)` | Returns stored heading boundaries. |
| `fs.list_dir(path)` | Lists non-hidden entries and sizes. |
| `fs.read_summary(doc_id)` | Reads a generated summary and consumes one file-read unit. |
| `fs.read_document(doc_id, ...)` | Reads `documents/<doc_id>.md`, optionally by 1-indexed line range or headings only. |
| `fs.read_file(path, ...)` | General prepared-file read with the same options. |
| `fs.grep(pattern, path, max_results)` | Case-insensitive regex search through Markdown files. |

Helper paths are resolved below `prepared_path`, preventing `..` escape through these
methods. In `full` configuration, helper-path validation would additionally whitelist
`_meta`, `_index`, `_summaries`, and `documents`; direct Python operations are a separate
security concern.

Read behavior:

- successful reads consume `max_file_reads` (default `12`);
- content is truncated at `max_read_bytes` characters (despite the parameter name being
  “bytes”);
- `headers_only=True` returns all lines beginning with `#`;
- read ranges are inclusive and 1-indexed;
- missing document names return suggestions when possible;
- retrieved-context tracking stores at most the first 1,500 characters of each read.

`max_read_lines` exists in configuration and the platform schema but is not currently
enforced by `read_file`. `fs.grep()` does not consume the file-read budget and returns at
most the requested number of line hits, each truncated to 200 characters.

### 5.6 Worker-model sub-calls

Generated code can call:

```python
call_sub_llm(prompt, context=None, mode="analysis")
```

Modes choose a short system instruction for analysis, summarization, or fact extraction.
The worker response is capped at 2,000 completion tokens and uses the shared retry,
circuit-breaker, response-cache, and token-tracking infrastructure.

Current budget caveats:

- `max_sub_calls` is shown in budget status, but `LLMClient.call()` does not currently
  consult or increment `BudgetManager`; it is therefore not an effective per-query limit.
- `max_tokens` is also displayed by the budget manager, but orchestrator/worker token use
  is not recorded there. Provider token accounting is still available in result metadata.
- `max_recursion_depth` is enforced inside `LLMClient.call()`, although the exposed worker
  call is normally synchronous rather than a recursively executing agent.

### 5.7 Final answer, sources, and confidence

If generated code sets `sources_used`, those IDs are returned. Otherwise, the agent falls
back to distinct sources from actual reads, or from grep hits if nothing was read.

Retrieved context prefers successful file-read snippets; if none exist, conversation
messages are returned as a fallback. The standard trace contains:

- every code block, output preview, success/error status, execution duration, and newly
  created variables;
- files accessed through helper reads;
- retrieved chunks paired with document IDs and synthetic descending scores.

The scores in the trace are positional placeholders (`1.0`, `0.95`, ...), not retrieval
similarity scores.

`confidence` is set by generated code or defaults to `LOW`. The prompt describes a
source-count policy, but the runtime does not independently verify or downgrade the
model's confidence value. `min_sources_for_high_confidence` therefore affects prompt
instructions, not deterministic validation.

In agent mode, retrieval and generation timing are combined in `retrieval_time`, while
`generation_time` is reported as `0.0`. `top_k` is ignored.

### 5.8 Reliability infrastructure

The shared RLM LLM client provides:

- exponential-backoff retries for errors whose text suggests rate limiting, timeout,
  connection failure, overload, or capacity problems;
- a circuit breaker that opens after repeated failures and retries after a timeout;
- an optional process-local response cache keyed by model and serialized messages;
- compatibility retries between `max_tokens` and `max_completion_tokens`, and omission
  of unsupported temperature parameters;
- OpenAI-compatible and Vertex AI provider routing.

The response cache defaults to 100 entries with a five-minute TTL. Cached calls report
zero additional tokens.

## 6. Security status and limitations

### 6.1 `lite` mode

`lite` runs generated code in the backend/CLI Python process. It has low startup overhead
and persistent variables, but a timeout value is not enforced around `SimpleREPL.execute`.
A slow or malicious snippet can block the process, and generated code must not be treated
as safely confined by the documented namespace.

Use it only when all of the following are acceptable:

- the model and corpus are trusted for local experimentation;
- the process has minimal credentials and filesystem permissions;
- generated-code execution is an accepted risk.

### 6.2 Current `full` mode

`full` selects `ProcessREPL`, which executes each code block in a new subprocess and can
terminate it after `repl_timeout`. This provides process separation and a hard per-block
time limit, but the current implementation is incomplete for RLM retrieval:

- the subprocess namespace does not expose `fs`, `call_sub_llm`, or `budget`;
- parent-side accumulated variables are not injected into the next subprocess;
- helper-based file reads and retrieval tracking therefore do not work as they do in
  lite mode;
- `InjectionGuard` and `SecureFilesystemTools` are defined but are not connected when the
  agent is initialized;
- the generated-code environment is not a hardened operating-system sandbox.

As a result, `full` should currently be considered experimental rather than a production
security mode. It may prevent normal agent exploration from succeeding, and it should
not be represented as complete prompt-injection protection.

### 6.3 Deployment guidance

For any RLM-RAG experiment:

- run the worker process/container with least privilege;
- do not mount secrets or unrelated host directories;
- disable unnecessary network access at the container/platform level;
- use disposable storage and credentials;
- inspect traces and generated code;
- prefer a non-code-executing RAG strategy for adversarial or multi-tenant corpora.

Application-level path checks and prompt instructions are not substitutes for OS/container
sandboxing.

## 7. Configuration

### Platform-exposed parameters

| Parameter | Phase | Default | Current effect |
| --- | --- | --- | --- |
| `prepared_path` | Build | managed | Physical prepared-directory location. |
| `worker_model` | Build | `gpt-5-nano` | Builds summaries/topics and is also used for agent sub-calls. |
| `chunk_size` | Build | `1000` | Included in manifest hash; no content chunking currently occurs. |
| `chunk_overlap` | Build | `200` | Included in manifest hash; no content chunking currently occurs. |
| `use_llm_summaries` | Build | `true` | Enables worker-generated document summaries. |
| `use_llm_topics` | Build | `true` | Enables worker-generated topic labels. |
| `max_topics_per_doc` | Build | `5` | Limits generated/fallback topics. |
| `orchestrator_model` | Query | RAG config model | Writes code and final answers. |
| `security_mode` | Query | `lite` | Selects in-process or experimental subprocess REPL. |
| `max_repl_steps` | Query | `15` | Limits orchestrator exploration turns. |
| `repl_timeout` | Query | `5.0` | Enforced by `ProcessREPL`; not enforced by `SimpleREPL`. |
| `max_file_reads` | Query | `12` | Enforced for successful helper reads in lite mode. |
| `max_read_bytes` | Query | `50000` | Character truncation limit per helper read. |
| `max_read_lines` | Query | `1000` | Present but not currently enforced. |
| `max_sub_calls` | Query | `8` | Present in budget status but not currently enforced. |
| `max_recursion_depth` | Query | `2` | Limits nested entry into worker-call logic. |
| `small_corpus_threshold` | Query | `10` | Chooses simple-context vs agent mode. |
| `top_k` | Query/top level | `5` | Used only by simple-context mode. |

The platform classifies `worker_model` as build-time because it changes summaries and
topics, even though agent-mode sub-calls also use it at query time. It cannot be changed
as a ready-index override.

When the top-level `llm_model` is overridden and `orchestrator_model` is not explicitly
overridden, the platform uses the new generation model as the effective orchestrator.

### Additional programmatic parameters

`RLMConfig` also exposes:

- `max_tokens`;
- circuit breaker thresholds/timeouts;
- retry count/base delay;
- response-cache enablement, size, and TTL;
- `min_sources_for_high_confidence`;
- logging level and orchestrator reasoning effort;
- provider, base URL, and API key.

These are not all exposed by the web parameter registry. Several controls have the
runtime caveats documented above.

## 8. Comparison with Filesystem RAG

| Concern | Filesystem RAG | RLM-RAG |
| --- | --- | --- |
| Deterministic index | BM25 sections plus Markdown navigation indexes | Catalog, section metadata, and topic map only |
| Query control | Native model tool calls | Model-generated Python |
| Small-corpus path | Same agent path | Direct first-`top_k` context fallback |
| Source formats | Recursive TXT/PDF/DOCX | Top-level TXT/MD/PDF/DOCX |
| Preparation enrichment | Summary, topics, entities, dates, questions | Summary and topics |
| Retrieval ranking | BM25 prefetch, then agent navigation | None built in; generated code chooses search/filter logic |
| External worker calls | No separate worker tier | Optional worker calls from generated code |
| Main risk | Variable agent path/latency | Generated-code execution plus variable agent path/latency |

Choose Filesystem RAG when you want an inspectable lexical index and native constrained
tools. Experiment with RLM-RAG when programmatic corpus operations are the capability
under test and the generated-code risk is acceptable.

## 9. Usage

CLI defaults:

```powershell
uv run rag-eval prepare --rag-type rlm_rag --input-dir data/raw
uv run rag-eval evaluate --rag-type rlm_rag
```

Programmatic configuration:

```python
from rag_evaluator.rag_implementations.rlm_rag import RLMConfig, RLMFilesystemRAG

rag = RLMFilesystemRAG(
    prepared_path="data/prepared/rlm_rag",
    rlm_config=RLMConfig(
        security_mode="lite",
        orchestrator_model="gpt-5-mini",
        worker_model="gpt-5-nano",
        use_llm_summaries=True,
        use_llm_topics=True,
        small_corpus_threshold=10,
    ),
)
rag.prepare_documents("data/raw")
result = rag.query_with_trace("Compare the recurring themes across the corpus.")
print(result["answer"])
print(result["retrieval_trace"])
```

To query an existing prepared index, instantiate with the same `prepared_path` and call
`load_index()` before querying.
