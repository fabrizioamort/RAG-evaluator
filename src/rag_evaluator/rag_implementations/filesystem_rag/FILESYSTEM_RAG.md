# Filesystem RAG: indexing and retrieval internals

Filesystem RAG is an agentic retrieval strategy that compiles source documents into an
inspectable directory of Markdown documents, summaries, navigation indexes, and a local
BM25 passage index. At query time, a language-model agent combines deterministic lexical
prefetch with filesystem tools and iterative source reading.

This document describes the implementation as it exists in this repository. For the
cross-strategy comparison and user-facing parameter summary, see
[RAG Strategies](../../../../docs/rag_strategies.md#filesystem-rag).

## 1. Architectural summary

Filesystem RAG has two distinct phases:

```text
.txt/.pdf/.docx sources
        |
        v
load -> Markdown conversion -> per-document analysis
        |                         |
        |                         +-> summary, topics, entities,
        |                             dates, question seeds, key sections
        v
prepared filesystem -> navigation indexes + BM25 section passages
        |
        v
query router -> deterministic prefetch -> ReAct agent -> targeted reads -> answer
```

It does **not** create embeddings or use a vector database. The durable index is the
prepared directory itself. Retrieval and answer generation are interleaved in one agent
run; there is no fixed top-k context boundary.

Main implementation files:

- [`filesystem_rag.py`](filesystem_rag.py): `BaseRAG` integration, preparation, querying,
  metrics, and retrieval-trace conversion.
- [`preparation/pipeline.py`](preparation/pipeline.py): eight-stage preparation workflow.
- [`preparation/document_processor.py`](preparation/document_processor.py): loading and
  Markdown conversion.
- [`preparation/analyzer.py`](preparation/analyzer.py): heuristic and LLM analysis.
- [`preparation/index_builder.py`](preparation/index_builder.py): document artifacts and
  navigation indexes.
- [`passage_index.py`](passage_index.py): persistent BM25 passage index and runtime scorer.
- [`agent/prefetch.py`](agent/prefetch.py): deterministic candidate generation.
- [`agent/agent.py`](agent/agent.py): tool-calling ReAct loop.
- [`agent/tools.py`](agent/tools.py): read-only filesystem and search tools.

## 2. Prepared filesystem contract

The default CLI path is `data/prepared/filesystem_rag`. The web platform instead assigns
an isolated path under `storage/indexes/<physical_id>/filesystem_rag`.

A successful preparation produces:

```text
prepared_path/
├── manifest/checkpoint files       # May be added by the calling platform
├── _meta/
│   ├── corpus_overview.md           # Corpus-level description and statistics
│   ├── navigation_guide.md          # Search guidance embedded in the agent prompt
│   └── statistics.json              # Counts, formats, topics, entities, analysis methods
├── _index/
│   ├── passages/
│   │   └── bm25.json                # Serializable section-level lexical index
│   ├── topics/
│   │   ├── _topic_map.md
│   │   └── <topic_slug>.md
│   ├── entities/
│   │   ├── _entity_registry.md
│   │   └── <entity_type_slug>.md
│   ├── temporal/
│   │   └── timeline.md
│   └── questions/
│       └── question_seeds.md
├── _summaries/
│   └── <doc_id>_summary.md
├── documents/
│   ├── <doc_id>.md                  # Converted source text
│   └── <doc_id>.meta.json           # Provenance and analysis metadata
└── _original/
    └── <original files>             # Copied by the standard FilesystemRAG pipeline
```

The Markdown files are not rewritten with visible line-number prefixes. Line offsets are
stored in metadata and returned by tools, and `read_file` accepts inclusive, 1-indexed
line ranges.

## 3. Indexing logic

### 3.1 Source discovery and identifiers

`load_documents()` recursively scans the input directory and sorts matching paths. The
supported source extensions are:

- `.txt`
- `.pdf`
- `.docx`

Markdown (`.md`) is not currently accepted by this preparation pipeline.

Each source is loaded through the shared document-loader abstraction. Failed files are
logged and skipped rather than aborting source discovery.

Most files receive sequential identifiers such as `doc_001`. A filename containing a
Legal RAG Bench-style passage identifier such as `1_5-c6-s1` is normalized to
`1.5-c6-s1`; this keeps prepared filenames and retrieval traces recognizable. Duplicate
identifiers receive `_2`, `_3`, and so on.

### 3.2 Conversion to Markdown

Conversion is format-specific:

- **TXT:** detects all-caps headings, underlined headings, and numbered sections, then
  converts detected headings to Markdown.
- **PDF:** starts from extracted text, adds a title, and heuristically promotes short
  capitalized paragraphs to headings.
- **DOCX:** starts from extracted paragraphs, adds a title, recognizes simple table-like
  rows, and heuristically promotes short paragraphs to headings.

The processor then records title, word/character/line counts, modification date, and
section information. Section boundaries are derived from Markdown headings when the
format-specific converter did not already provide them.

A current TXT-specific caveat is that directly detected sections contain start lines but
not computed end lines. The BM25 builder consequently extends those sections to the end
of the document, and the summary can display an unset end value. TXT start offsets are
also detected before an inferred title is inserted. Treat these line ranges as navigation
hints and verify them against `read_file` output.

This is structural normalization, not semantic parsing: PDF layout, tables, columns, and
DOCX styles are only as accurate as the shared loader's extracted text and the heading
heuristics.

### 3.3 Hybrid per-document analysis

Every converted document is analyzed into a `DocumentAnalysis` containing:

- summary;
- topics and topic scores;
- entities grouped by type;
- temporal markers;
- candidate questions the document can answer;
- key sections;
- related topics;
- the analysis method actually used.

The default decision is based on `word_threshold` (default `1000`):

```text
word_count < word_threshold  -> heuristic analysis
word_count >= word_threshold -> LLM analysis
```

`force_analysis_method="heuristic"` or `"llm"` can override this programmatically.

#### Heuristic path

The heuristic analyzer uses:

- keyword-density scores over predefined general, technical, business, science, and
  legal vocabularies;
- term-frequency keywords after stop-word removal;
- regex entity extraction;
- regex date extraction with surrounding text;
- title/topic/section templates for question seeds;
- extractive summary heuristics.

It is inexpensive and deterministic but less domain-aware than the LLM path.

#### LLM path

The LLM analyzer asks for a JSON object containing the fields above. Input is truncated
to 40,000 characters. The prompt emphasizes decisive rules, qualifications, named
procedures, legal references, and realistic evaluation questions. Invalid JSON or any
provider failure falls back to heuristic analysis for that document.

The selected RAG generation model is also used for Filesystem RAG analysis unless the
caller constructs the preparation pipeline differently.

### 3.4 Per-document artifacts

For each successfully analyzed source, the indexer atomically writes:

1. `documents/<doc_id>.md`: converted content.
2. `documents/<doc_id>.meta.json`: original path/format, counts, title, topics, entities,
   sections, question seeds, summary path, and analysis method.
3. `_summaries/<doc_id>_summary.md`: overview, key points, section line ranges, entities,
   and example questions.

The metadata's `original_file` field is also used after querying to map prepared document
IDs back to original source paths in evaluation output.

### 3.5 Navigation indexes

The Markdown indexes are browsing aids for the agent, not independent ranking engines.

#### Topic index

Up to five corpus-adaptive labels are taken from `analysis.topics`, with scored topic
categories as a fallback. The first label is primary and the others secondary.
`_topic_map.md` maps labels to document IDs; one detail file per topic adds summaries,
sections, entities, question seeds, and links to other topics.

#### Entity index

All analyzer-provided entity types are preserved. `_entity_registry.md` shows frequent
entities and document IDs; one file per entity type contains the complete mapping.

#### Question-seed index

Generated questions are classified with string heuristics into factual, how-to,
comparison, analysis, or other groups. Each entry maps a question to a document ID.
During querying, this file also feeds deterministic overlap-based prefetch hints.

#### Timeline

Dates and associated text are collected, string-sorted, grouped by the first four date
characters, and linked to source document IDs. Date normalization is intentionally
lightweight, so mixed date formats may not be chronologically ordered.

### 3.6 BM25 passage index

The primary deterministic search artifact is `_index/passages/bm25.json`.

#### Passage creation

A passage normally corresponds to one detected Markdown section (subject to the TXT
boundary caveat above). It records:

- document and passage IDs;
- title and section title;
- source and summary paths;
- inclusive start/end lines;
- token count and term frequencies;
- a short preview.

The indexed search text concatenates the document title, section title, section text,
and document-level analysis text (summary, question seeds, topics, and entities). This
improves lexical recall but means a term found only in analysis metadata can make every
section of the document eligible. Sections with fewer than five distinct normalized
terms are skipped; if all are skipped, one whole-document fallback passage is created.

#### Tokenization and scoring

The local tokenizer keeps alphanumeric words with simple hyphen/apostrophe support,
lowercases them, and applies a light English suffix stemmer. It stores corpus document
frequencies and uses BM25 with:

- `k1 = 1.5`
- `b = 0.75`

At runtime, all positive-scoring sections are ranked, but only the best section from each
document is returned. The result includes matched terms, a focused snippet, and a
`read_hint` containing the exact source and line range.

### 3.7 Corpus synthesis and validation

Preparation creates a corpus overview, navigation guide, statistics, and copies original
files. Corpus overview synthesis is heuristic by default; `use_llm_synthesis=True` is a
programmatic option and considers at most the first 20 summaries.

The final validation checks required directories/files and verifies that each processed
document has Markdown, metadata, and summary artifacts. Validation results are reported
in preparation metrics.

The core preparation pipeline is a full-corpus build. Its resumable wrapper registers
per-document checkpoints but invokes the same full preparation pass and then verifies
artifacts; it is not incremental per-document indexing.

## 4. Query and retrieval logic

### 4.1 Agent initialization and session cache

`load_index()` initializes a `FilesystemRAGAgent` from an existing prepared path without
re-running preparation. Agent startup warms these files:

- corpus overview;
- navigation guide;
- topic map;
- entity registry.

Only the corpus overview and navigation guide are automatically embedded in every system
prompt. The topic and entity indexes remain available through tools.

### 4.2 Query routing

A regex/heuristic `QueryRouter` labels the question as:

- `known_item`: specific name, definition, location, procedure, or direct lookup;
- `exploratory`: summary, comparison, explanation, broad topic, benefits, or challenges.

The route changes the navigation hint in the system prompt; it does not select a separate
retrieval backend or force a fixed tool sequence. Ambiguous questions default toward
exploratory behavior.

### 4.3 Deterministic prefetch

Before the first LLM turn, the agent creates up to eight candidates:

1. BM25 search with the original question.
2. A second BM25 search with a deterministic legal-vocabulary reformulation when one is
   produced by the built-in rules.
3. Question-seed hints whose normalized terms overlap the question.

Candidates are merged by source, receive a small boost when found by multiple paths, and
are diversified so one Legal RAG Bench section family contributes at most two results.
Candidate snippets and read hints are injected into the initial prompt as **hypotheses**,
not accepted evidence. This prefetch is deterministic for a fixed index and question.

The legal reformulation table is benchmark-oriented. It can improve recall for the legal
corpus but should not be interpreted as a general query-expansion model.

### 4.4 ReAct tool loop

The LLM receives OpenAI-compatible function definitions and iterates until it returns a
plain final answer or reaches a limit.

| Tool | Behavior |
| --- | --- |
| `search_passages(query, top_k)` | Runs the local BM25 index and returns one best section per document with snippets/read hints. |
| `grep_search(...)` | Regex search over files; supports ranked matches, context lines, truncation metadata, and an all-terms mode. A failed plain multi-word search may retry automatically in all-terms mode. |
| `read_file(...)` | Reads complete files, headers, or inclusive line ranges. This is the principal evidence channel. |
| `list_directory(path)` | Lists files/directories and file sizes. |
| `find_files(pattern, path)` | Recursively finds names matching a glob. |
| `get_file_info(path)` | Returns byte size, line count, modification date, and extension without loading content. |

All tool paths are resolved below `prepared_path`; `..` traversal outside that root is
rejected. Common binary extensions and NUL-containing files are not returned.

### 4.5 Progressive disclosure and read limits

The prompt instructs the agent to search first, read summaries, inspect headings, and
then read targeted source ranges.

Important implementation limits:

- A full `read_file` is refused above 100,000 bytes; the agent must use headers, grep, or
  line ranges.
- `headers_only=True` extracts headings only when the file has more than 500 lines;
  shorter files are returned in full.
- A read result sent back to the LLM is normally capped at 10,000 characters.
- Evidence retained in response context is capped at 20,000 characters per read.
- Old tool-result messages are compacted after two iterations to control prompt growth.
- Repeating an identical tool call returns a compact cache reference rather than running
  it again. It still consumes a tool-call slot, but a cached `read_file` does not consume
  another file-read slot.

### 4.6 Evidence collection and answer safeguards

Only `read_file` results become retrieved context. Search snippets guide navigation but
are not automatically reported as retrieved evidence. Reads under
`_index/questions/` and `_index/passages/` are explicitly excluded from evidence context;
source document reads are preferred for downstream evaluation.

The loop also includes:

- a one-time warning after roughly 60% of the iteration/tool budget;
- a one-time evidence nudge if the model tries to finish after reading fewer than two
  distinct `documents/` files;
- a recovery attempt when raw tool-call markup leaks into answer text;
- one plain-completion retry for empty, refusal-like, predominantly CJK, or tool-markup
  answers;
- forced best-effort synthesis when iteration, tool-call, or file-read limits are hit.

The default limits are 10 model iterations, 20 tool calls, and 10 uncached file reads.
Agent behavior remains model-dependent, so the same question can follow different paths
and produce different latency or evidence sets.

### 4.7 Result and trace semantics

`query()` executes one combined retrieval/generation pass and returns:

- final answer;
- evidence chunks collected from file reads;
- original/reportable source paths;
- route, iterations, tool calls, prefetched candidates, reasoning trace, retries, timing,
  and measured model-token usage.

`query_with_trace()` also uses exactly one agent pass so the answer and trace stay aligned.
Its standard retrieval trace includes query routing, lexical prefetch, agent steps, file
reads, and retrieved chunks.

`top_k` is ignored by Filesystem RAG's public `query()` and `retrieve()` methods. The
agent decides how much evidence to gather within its budgets. Individual calls to the
internal `search_passages` tool still have their own `top_k`.

The generic `retrieve()` method also runs the full agent (including answer generation),
and a later `generate()` performs a separate context-only generation call. Prefer
`query_with_trace()` when evaluating the native interleaved behavior.

## 5. Configuration

### Platform-exposed parameters

| Parameter | Phase | Default | Effect |
| --- | --- | --- | --- |
| `prepared_path` | Build | managed | Physical prepared-directory location; isolated per platform index. |
| `word_threshold` | Build | `1000` | Chooses heuristic vs LLM document analysis. |
| `llm_model` | Query/top level | `gpt-4o-mini` | Model used by the navigation agent; during a new build it is also used for LLM analysis. |
| `max_iterations` | Query | `10` | Maximum orchestrator turns. |
| `max_tool_calls` | Query | `20` | Maximum tool calls, including duplicate cached calls. |
| `max_file_reads` | Query | `10` | Maximum uncached `read_file` operations. |

Changing build-time parameters requires a new platform index. Query-time values can be
overridden for a ready index.

### Programmatic preparation controls

The Python constructor additionally accepts:

- `force_analysis_method`: `"heuristic"`, `"llm"`, or `None`;
- `use_llm_synthesis`: whether to use the LLM for the corpus overview.

These controls are not currently part of the web platform's public Filesystem RAG
parameter schema.

## 6. Operational characteristics and trade-offs

### Strengths

- Entire index is local, portable, human-readable, and easy to inspect.
- BM25 gives deterministic lexical candidates without an external service.
- Topic/entity/question/timeline views support broad navigation and debugging.
- Targeted file reads can collect evidence beyond a static top-k result set.
- Retrieval traces expose the actual navigation path and sources read.

### Costs and limitations

- Preparation can make one LLM analysis call per document at or above the threshold.
- Query latency and token use include multiple orchestrator turns and tool results.
- No vector-semantic retrieval is used; synonyms depend on analysis metadata,
  reformulation, or model-driven navigation.
- Heuristic extraction is English-oriented and several legal search rules are
  corpus-specific.
- Source conversion is heuristic and may lose complex layout.
- The agent is less deterministic than fixed-ranker strategies.
- The prepared filesystem should be treated as generated data that can contain source
  content and LLM-derived metadata.

## 7. Usage

CLI:

```powershell
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw
uv run rag-eval evaluate --rag-type filesystem_rag
```

Programmatic:

```python
from rag_evaluator.rag_implementations.filesystem_rag import FilesystemRAG

rag = FilesystemRAG(
    prepared_path="data/prepared/filesystem_rag",
    word_threshold=1000,
    max_iterations=10,
    max_tool_calls=20,
    max_file_reads=10,
)
rag.prepare_documents("data/raw")
result = rag.query_with_trace("Which documents define the relevant procedure?")
print(result["answer"])
print(result["retrieval_trace"])
```

To query an already prepared index, instantiate with the same `prepared_path` and call
`load_index()` before `query()`.
