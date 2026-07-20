# OKF RAG Implementation Plan

  Date: 2026-07-09
  Status: Draft for future implementation
  Target RAG type key: `okf_rag`

## Summary

  Add a new `okf_rag` strategy that uses the Open Knowledge Format, OKF, as a compiled, portable, human-readable knowledge
  layer between raw documents and query-time generation.

  This should not be a cosmetic fork of `filesystem_rag`. The value of `okf_rag` is that it turns a corpus into an OKF-
  conformant bundle of concept documents with YAML frontmatter, cross-links, citations, indexes, logs, and optional search/
  graph sidecars.

  The existing Filesystem RAG already proves that agentic filesystem navigation fits this repository. `okf_rag` should reuse
  that lesson but tighten the prepared artifact around a public format.

  Primary references:

- Karpathy, LLM Wiki: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- Google Cloud OKF announcement:
  <https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing>
- OKF v0.1 draft spec: <https://raw.githubusercontent.com/GoogleCloudPlatform/knowledge-catalog/main/okf/SPEC.md>

## Core idea

  `okf_rag` should compile raw documents into a durable markdown knowledge bundle.

  The bundle should be:

- readable by humans;
- parseable by agents;
- versionable in git;
- portable outside this project;
- navigable through indexes, links, backlinks, and metadata;
- grounded in immutable raw sources.

  The intended pipeline is:

  ```text
  raw documents
     ↓
  OKF producer / enricher
     ↓
  OKF bundle:
    index.md
    log.md
    sources/*.md
    concepts/*.md
    entities/*.md
    topics/*.md
     ↓
  OKF query agent:
    search catalog
    read concept pages
    traverse links/backlinks
    inspect source evidence
    answer with citations

## Why this belongs in this codebase

  The repository already compares multiple RAG strategies behind a common BaseRAG interface:

  - vector_semantic
  - vector_hybrid
  - graph_rag
  - filesystem_rag
  - rlm_rag
  - google_vertex_search

  okf_rag covers a distinct design point:

  > retrieval over a persistent, compiled, interoperable knowledge bundle.

  It is closest to filesystem_rag, but the prepared artifact has stronger semantics. Filesystem RAG prepares documents into
  summaries and indexes. OKF RAG prepares documents into concept pages with typed metadata, links, source evidence, and
  conformance rules.

  This should be especially useful for:

  - legal corpora;
  - technical documentation;
  - data catalogs;
  - research collections;
  - internal knowledge bases;
  - multi-hop and synthesis-heavy evaluation sets.

  ## Non-goals

  The first version should not try to solve everything.

  Out of scope for v1:

  - automatic web crawling;
  - query-time mutation of the OKF bundle;
  - full incremental wiki maintenance;
  - typed semantic graph edges;
  - embeddings or hybrid vector search;
  - custom frontend graph visualization;
  - replacing existing vector/hybrid/graph RAG systems.

  The first version should prove that OKF is a useful prepared artifact and retrieval substrate.

  ## Recommended design decision

  Implement okf_rag as a new RAG type, not as a mode inside filesystem_rag.

  Reasons:

  - It has different build artifacts.
  - It needs OKF conformance checks.
  - It has different retrieval traces.
  - It should be evaluated against Filesystem RAG directly.
  - It needs separate build/query parameters in the registry.
  - It should eventually support externally supplied OKF bundles.

  Reuse lower-level ideas from Filesystem RAG, but keep the high-level pipeline separate.

  ## Proposed prepared artifact layout

  Default output path:

  data/prepared/okf_rag/
  ├── index.md
  ├── log.md
  ├── sources/
  │   ├── index.md
  │   └── <source_id>.md
  ├── concepts/
  │   ├── index.md
  │   └── <concept_slug>.md
  ├── entities/
  │   ├── index.md
  │   └── <entity_slug>.md
  ├── topics/
  │   ├── index.md
  │   └── <topic_slug>.md
  ├── questions/
  │   ├── index.md
  │   └── <question_slug>.md
  ├── references/
  │   ├── index.md
  │   └── <reference_slug>.md
  └── .rag_eval/
      ├── manifest.json
      ├── source_map.json
      ├── concept_catalog.json
      ├── link_graph.json
      ├── backlinks.json
      ├── lexical_index.json
      ├── conformance_report.json
      └── build_metrics.json

  Rules:

  - OKF-facing files are markdown.
  - Runtime helper artifacts live under .rag_eval/.
  - index.md and log.md are reserved OKF files.
  - Every other .md file must contain parseable YAML frontmatter.
  - Every concept document must have a non-empty type.
  - Sidecar JSON files should be rebuildable from the markdown bundle.

  ## OKF document convention

  Example concept page:

  ---
  type: Concept
  title: Example concept
  description: One-sentence summary used for routing and snippets.
  resource:
  tags: [example, topic]
  timestamp: 2026-07-09T00:00:00Z
  okf_version: "0.1"
  rag_evaluator:
    source_ids: ["source_001"]
    confidence: medium
    lifecycle: draft
  ---

  # Summary

  Short factual summary.

  # Details

  Structured notes, tables, bullets, examples, and relationships.

  # Relationships

  - Related to [Other concept](/concepts/other_concept.md).

  # Source Evidence

  - [source_001:42-57](/sources/source_001.md)

  # Citations

  [1] [Original source file](/sources/source_001.md)

  Recommended type values:

  - Source
  - Concept
  - Entity
  - Topic
  - Question
  - Reference
  - Claim
  - Metric
  - API Endpoint
  - Schema
  - Legal Authority
  - Case
  - Contract Clause

  These should not be hard-coded as the only allowed values. OKF consumers must tolerate unknown types.

  ## Source representation

  The sources/ directory should contain one markdown concept per raw source document, at least for v1.

  Example:

  ---
  type: Source
  title: source_001.pdf
  description: Source document converted from raw upload.
  resource: file:///original/path/source_001.pdf
  tags: [source]
  timestamp: 2026-07-09T00:00:00Z
  rag_evaluator:
    original_path: data/raw/source_001.pdf
    sha256: "<hash>"
    loader: "pdf"
  ---

  # Extracted Text

  Line-numbered or section-numbered extracted source text.

  Source pages must preserve enough stable location information to support citations. That can be line numbers, page numbers,
  section IDs, or chunk IDs.

  For v1, prefer one source page per source document. Later versions can split large documents into section-level source
  concepts.

  ## Proposed package layout

  Add:

  src/rag_evaluator/rag_implementations/okf_rag/
  ├── __init__.py
  ├── okf_rag.py
  ├── models.py
  ├── okf_parser.py
  ├── okf_writer.py
  ├── conformance.py
  ├── catalog.py
  ├── graph.py
  ├── search.py
  ├── preparation.py
  ├── prompts.py
  └── agent.py

  ### okf_rag.py

  Defines:

  class OKFRAG(BaseRAG):
      ...

  Responsibilities:

  - implement prepare_documents;
  - implement load_index;
  - implement retrieve;
  - implement generate;
  - implement query;
  - implement query_with_trace;
  - implement get_metrics.

  The lifecycle should match the project pattern:

  - build paths call prepare_documents;
  - query paths call load_index;
  - query-time overrides must not mutate stored artifacts.

  ### models.py

  Define internal data models:

  - OKFDocument
  - OKFFrontmatter
  - OKFLink
  - OKFCitation
  - OKFBuildMetrics
  - OKFConformanceIssue
  - OKFRetrievalCandidate
  - OKFRetrievalPlan

  Use dataclasses unless Pydantic is already clearly preferred in this layer.

  ### okf_parser.py

  Responsibilities:

  - parse YAML frontmatter;
  - extract markdown links;
  - extract citations;
  - resolve absolute bundle-relative links;
  - resolve relative links;
  - return tolerant parse results with warnings.

  Consumers should not reject unknown fields or unknown type values.

  ### okf_writer.py

  Responsibilities:

  - write concept markdown files;
  - normalize slugs and paths;
  - preserve unknown frontmatter keys when updating documents;
  - generate index.md;
  - append to log.md.

  Writes should be deterministic where possible. Stable file ordering matters because OKF bundles should produce useful git
  diffs.

  ### conformance.py

  Validate hard OKF requirements:

  - every non-reserved .md file has parseable frontmatter;
  - every non-reserved .md file has non-empty type;
  - reserved index.md and log.md follow expected conventions when present.

  Also produce soft quality warnings:

  - missing title;
  - missing description;
  - broken internal links;
  - concepts with no source evidence;
  - claims without citations;
  - orphan pages;
  - duplicate titles/slugs;
  - stale timestamps.

  Output:

  .rag_eval/conformance_report.json

  Hard conformance errors may fail preparation when strict_conformance=True. Soft quality issues should warn but not fail.

  ### catalog.py

  Build:

  .rag_eval/concept_catalog.json

  The catalog should include:

  - path;
  - type;
  - title;
  - description;
  - tags;
  - timestamp;
  - source IDs;
  - headings;
  - short body preview.

  The agent should use this catalog before reading full page bodies.

  ### graph.py

  Build:

  .rag_eval/link_graph.json
  .rag_eval/backlinks.json

  Responsibilities:

  - extract directed edges from markdown links;
  - compute backlinks;
  - detect broken links;
  - detect orphans;
  - expose neighbor lookup.

  Do not add typed graph semantics in v1. OKF v0.1 treats links as relationships whose meaning is conveyed by surrounding
  prose.

  ### search.py

  Implement local lexical concept search.

  Search over:

  - title;
  - description;
  - tags;
  - headings;
  - body preview;
  - optionally source text.

  V1 should use deterministic local lexical ranking. Avoid embeddings initially so the OKF-specific behavior is isolated.

  Later versions can add SQLite FTS5 or hybrid vector/BM25 search.

  ### preparation.py

  Responsibilities:

  1. Discover raw documents.
  2. Convert them to markdown/source pages.
  3. Create Source OKF documents.
  4. Extract candidate concepts/entities/topics.
  5. Create concept/entity/topic pages.
  6. Add source evidence references.
  7. Generate indexes.
  8. Append build log entries.
  9. Build catalog/search/graph sidecars.
  10. Run conformance checks.

  Use deterministic extraction first where possible:

  - filenames;
  - headings;
  - keyword frequency;
  - dates;
  - simple metadata.

  Use LLM enrichment only where judgment is useful:

  - summarization;
  - concept extraction;
  - entity/topic extraction;
  - relationship prose;
  - contradiction detection;
  - tag selection.

  ### prompts.py

  Centralize prompts for:

  - source summarization;
  - concept extraction;
  - entity extraction;
  - topic extraction;
  - concept merge/update;
  - contradiction detection;
  - query planning;
  - answer synthesis;
  - linting.

  Prompt rules:

  - require citations to source IDs or source line ranges;
  - disallow unsupported claims;
  - preserve uncertainty;
  - prefer structured markdown;
  - return JSON for machine-parsed outputs;
  - do not invent links to concepts that do not exist.

  ### agent.py

  Implement an OKF query agent with a small toolset:

  - read_index(path)
  - search_concepts(query, filters=None, limit=10)
  - read_concept(path)
  - get_neighbors(path)
  - get_backlinks(path)
  - read_source_evidence(source_ref)
  - validate_citation(source_ref, claim)

  Recommended query flow:

  1. Search the concept catalog.
  2. Select candidate concept pages.
  3. Expand through links/backlinks for multi-hop questions.
  4. Read only the most relevant concept bodies.
  5. Read source evidence for important claims.
  6. Synthesize answer with citations.

  Avoid loading the entire OKF bundle into context.

  ## Registry integration

  Update:

  src/rag_evaluator/rag_implementations/registry.py

  Add class path:

  "okf_rag": "rag_evaluator.rag_implementations.okf_rag.okf_rag.OKFRAG"

  Add metadata:

  "okf_rag": {
      "name": "OKF RAG",
      "description": "Agentic retrieval over an Open Knowledge Format markdown knowledge bundle",
  }

  Suggested parameter schema:

  "okf_rag": {
      "properties": {
          "prepared_path": {
              "type": "string",
              "phase": "build",
              "description": "Prepared OKF bundle output path",
              "platform_managed": True,
          },
          "okf_version": {
              "type": "string",
              "phase": "build",
              "default": "0.1",
              "description": "Target OKF specification version",
          },
          "concept_granularity": {
              "type": "string",
              "phase": "build",
              "default": "balanced",
              "enum": ["coarse", "balanced", "fine"],
              "description": "Controls how aggressively source material is split into concepts",
          },
          "use_llm_enrichment": {
              "type": "boolean",
              "phase": "build",
              "default": True,
              "description": "Use LLM calls to summarize, extract concepts, and write relationship prose",
          },
          "max_concepts_per_source": {
              "type": "integer",
              "phase": "build",
              "default": 12,
              "minimum": 1,
              "description": "Upper bound on generated concept pages per source document",
          },
          "include_source_pages": {
              "type": "boolean",
              "phase": "build",
              "default": True,
              "description": "Store source text as OKF Source documents",
          },
          "strict_conformance": {
              "type": "boolean",
              "phase": "build",
              "default": True,
              "description": "Fail preparation on hard OKF conformance errors",
          },
          "max_iterations": {
              "type": "integer",
              "phase": "query",
              "default": 8,
              "minimum": 1,
              "description": "Maximum agent planning iterations per query",
          },
          "max_concepts_read": {
              "type": "integer",
              "phase": "query",
              "default": 12,
              "minimum": 1,
              "description": "Maximum OKF concept pages read per query",
          },
          "max_source_reads": {
              "type": "integer",
              "phase": "query",
              "default": 6,
              "minimum": 0,
              "description": "Maximum source evidence reads per query",
          },
          "expand_backlinks": {
              "type": "boolean",
              "phase": "query",
              "default": True,
              "description": "Allow query agent to expand retrieval through backlinks",
          },
          "search_mode": {
              "type": "string",
              "phase": "query",
              "default": "lexical_graph",
              "enum": ["lexical", "graph", "lexical_graph"],
              "description": "Concept retrieval mode",
          },
      },
  }

  ## BaseRAG behavior

  Suggested constructor:

  def __init__(
      self,
      llm_model: str = "gpt-4o-mini",
      prepared_path: str = "data/prepared/okf_rag",
      okf_version: str = "0.1",
      concept_granularity: str = "balanced",
      use_llm_enrichment: bool = True,
      max_concepts_per_source: int = 12,
      include_source_pages: bool = True,
      strict_conformance: bool = True,
      max_iterations: int = 8,
      max_concepts_read: int = 12,
      max_source_reads: int = 6,
      expand_backlinks: bool = True,
      search_mode: str = "lexical_graph",
      config: RAGConfig | None = None,
  ) -> None:

  ### prepare_documents

  Expected behavior:

  - build or replace the OKF bundle;
  - generate source/concept/entity/topic/question docs;
  - generate OKF indexes and logs;
  - build sidecar catalog/search/graph files;
  - run conformance checks;
  - initialize runtime handles.

  Recommendation for v1: full rebuild only. Add incremental ingest later.

  ### load_index

  Expected behavior:

  - read existing OKF bundle;
  - load catalog and graph sidecars;
  - rebuild sidecars if missing or stale;
  - do not mutate OKF markdown unless explicitly configured.

  ### retrieve

  Expected behavior:

  - perform catalog search;
  - expand through graph/backlinks if enabled;
  - read selected concept pages;
  - read source evidence when needed;
  - return RetrievedContext.

  Each RetrievedChunk.metadata should include:

  - okf_path;
  - okf_type;
  - title;
  - tags;
  - source_refs;
  - retrieval_reason;
  - via_link;
  - via_backlink.

  Retrieval trace steps should include:

  - catalog_search;
  - concept_read;
  - graph_expansion;
  - backlink_expansion;
  - source_evidence_read;
  - citation_validation.

  ### generate

  Expected behavior:

  - generate answer from retrieved OKF context;
  - cite concept/source references;
  - prefer source evidence over synthesized concept prose when they conflict;
  - state uncertainty when context is insufficient.

  ### query

  Expected behavior:

  - reset token counters;
  - call retrieve;
  - call generate;
  - return standard result shape:

  {
      "answer": "...",
      "context": [...],
      "metadata": {...},
  }

  Metadata should include:

  - retrieval_time;
  - generation_time;
  - concepts_read;
  - sources_read;
  - search_mode;
  - graph_expansions;
  - broken_links_encountered;
  - citation_validation_failures;
  - token_usage.

  ### query_with_trace

  Override this method so OKF-specific retrieval steps appear in the trace.

  ## Backend/platform integration

  The registry-driven backend should mostly pick this up automatically, but verify:

  - GET /api/v1/rag-types includes okf_rag;
  - GET /api/v1/rag-types/okf_rag/parameters returns phase metadata;
  - index creation stores build-time parameters;
  - evaluation rejects build-time overrides for existing OKF indexes;
  - query overrides allow only query-phase OKF parameters plus generation settings.

  Ready-index check for okf_rag should require:

  - root index.md;
  - .rag_eval/manifest.json;
  - .rag_eval/conformance_report.json;
  - no hard conformance failures.

  ## Frontend integration

  Likely automatic if the frontend consumes /rag-types.

  Manual checks:

  - RAG config dialog renders okf_rag;
  - enum fields render as selects;
  - booleans render as toggles;
  - platform-managed prepared_path is hidden or disabled;
  - build/query phase grouping is correct.

  Custom graph visualization is optional and should not block v1.

  ## Evaluation strategy

  Do not evaluate OKF RAG only on simple factual lookup. It may not beat vector/hybrid search there.

  Evaluate on:

  - multi-hop questions;
  - cross-document synthesis;
  - entity relationship questions;
  - timeline/process questions;
  - comparison questions;
  - source-grounded citation questions.

  Compare against:

  - vector_semantic;
  - vector_hybrid;
  - graph_rag;
  - filesystem_rag;
  - rlm_rag.

  Metrics:

  - answer correctness;
  - faithfulness;
  - context precision;
  - context recall;
  - citation accuracy;
  - retrieval trace interpretability;
  - query latency;
  - build cost;
  - token usage.

  OKF-specific diagnostics:

  - number of concepts generated;
  - average source refs per concept;
  - orphan concept count;
  - broken link count;
  - percentage of answer claims backed by source evidence;
  - compression ratio from raw source count to concept count.

  ## Testing plan

  Add tests under:

  tests/rag_implementations/test_okf_rag/

  Unit tests:

  - parse valid frontmatter;
  - warn/fail on invalid frontmatter;
  - identify reserved files;
  - extract absolute bundle-relative links;
  - extract relative links;
  - detect broken links;
  - build backlinks;
  - generate index files;
  - generate log entries;
  - build concept catalog;
  - lexical search returns expected concepts;
  - conformance passes on minimal valid bundle;
  - conformance fails on concept without type.

  Integration tests:

  - prepare a tiny raw corpus into OKF;
  - load an existing OKF bundle;
  - retrieve relevant concepts;
  - query returns standard RAG result shape;
  - query trace contains OKF-specific steps;
  - build-time overrides are rejected for ready indexes;
  - query-time overrides are accepted.

  Use a small fixture corpus:

  tests/fixtures/okf_rag/raw/
  ├── company_policy.md
  ├── incident_runbook.md
  └── product_faq.md

  If possible, keep a deterministic no-LLM build mode for stable tests.

  ## Implementation phases

  ### Phase 1: Read-only OKF consumer

  Goal: query an existing OKF bundle.

  Tasks:

  1. Create okf_rag package.
  2. Implement parser.
  3. Implement catalog builder.
  4. Implement graph/backlink builder.
  5. Implement lexical search.
  6. Implement OKFRAG.load_index.
  7. Implement retrieve, generate, and query.
  8. Register okf_rag.
  9. Add tests with a hand-written OKF bundle.

  This proves the query model before tackling LLM-based OKF generation.

  ### Phase 2: Deterministic OKF producer

  Goal: convert raw documents into a minimal OKF bundle without LLM enrichment.

  Tasks:

  1. Create source pages from raw files.
  2. Generate simple concept/topic pages from headings, filenames, and metadata.
  3. Generate indexes.
  4. Generate logs.
  5. Generate sidecars.
  6. Run conformance.
  7. Add integration tests for prepare_documents.

  Expected quality: basic but useful.

  ### Phase 3: LLM enrichment

  Goal: generate better concepts, entities, topics, and relationship prose.

  Tasks:

  1. Add source summarization prompt.
  2. Add concept extraction prompt.
  3. Add entity/topic extraction prompts.
  4. Add concept page writer.
  5. Add source evidence mapping.
  6. Add contradiction notes where detected.
  7. Add token usage accounting.

  Mitigations against hallucination:

  - require source references;
  - cap concepts per source;
  - preserve raw source pages;
  - warn on uncited claims;
  - keep generated concepts marked with lifecycle/confidence metadata.

  ### Phase 4: Lint workflow

  Goal: keep OKF bundles healthy.

  Tasks:

  1. Detect broken links.
  2. Detect orphan pages.
  3. Detect duplicate concepts.
  4. Detect missing citations.
  5. Detect stale overview/index pages.
  6. Produce a lint report.
  7. Optionally add safe repair mode for metadata/index fixes.

  Do not mutate bundles during evaluation queries.

  ### Phase 5: Incremental ingest

  Goal: add new sources to an existing OKF bundle without full rebuild.

  Tasks:

  1. Detect source hashes.
  2. Add new source pages.
  3. Identify affected concepts.
  4. Update affected concept pages.
  5. Append log entries.
  6. Rebuild sidecars.
  7. Run lint/conformance.

  This is where OKF RAG becomes genuinely “compounding” like the LLM Wiki idea.

  ## Security considerations

  Rules:

  - restrict all reads to prepared_path;
  - normalize and validate paths;
  - reject path traversal;
  - treat markdown as untrusted text;
  - do not execute code from the bundle;
  - do not follow external URLs at query time;
  - do not mutate the bundle during evaluation queries.

  If future web enrichment is added:

  - require explicit configuration;
  - restrict allowed domains;
  - cap fetched pages;
  - store fetched pages as Reference concepts;
  - log every fetch.

  ## Performance considerations

  Potential bottlenecks:

  - LLM enrichment during build;
  - reading too many concept pages;
  - searching large bundles;
  - rebuilding sidecars repeatedly.

  Mitigations:

  - build sidecars once;
  - load catalog and graph into memory during load_index;
  - search the compact catalog before reading page bodies;
  - cap concept/source reads;
  - cache parsed markdown/frontmatter;
  - rebuild sidecars only when bundle files changed.

  ## Open questions

  1. Should okf_rag support externally supplied OKF bundles?
      - Recommendation: yes, starting with Phase 1.

  2. Should query-time interaction update the OKF bundle?
      - Recommendation: no for evaluations. Possibly later in an explicit maintenance mode.

  3. Should OKF RAG use embeddings?
      - Recommendation: not initially.

  4. Should source pages store full text or references to external raw files?
      - Recommendation: store enough extracted text for portable evaluation.

  5. Should links be typed?
      - Recommendation: not in v1. Keep OKF v0.1 compatibility.

  6. Should code be shared with Filesystem RAG?
      - Recommendation: share low-level safe file/search utilities only if clean.

  ## Definition of done for first useful version

  The first useful version is complete when:

  - okf_rag is registered as a RAG type;
  - a hand-written OKF bundle can be loaded and queried;
  - raw documents can be prepared into a minimal OKF bundle;
  - the bundle passes OKF conformance checks;
  - query returns standard BaseRAG result shape;
  - query trace includes catalog search, concept reads, graph expansion, and source evidence reads;
  - backend exposes the RAG type and parameter schema;
  - tests cover parsing, conformance, graph, search, prepare, load, retrieve, and query shape;
  - docs explain when to choose OKF RAG vs Filesystem RAG.

  ## Documentation updates

  Update:

  docs/rag_strategies.md
  docs/api.md
  docs/custom_rag_integration.md
  README.md

  Suggested docs section:

  ## OKF RAG

  Type key: `okf_rag`

  OKF RAG compiles raw documents into an Open Knowledge Format bundle: markdown
  concept documents with YAML frontmatter, cross-links, citations, indexes, and
  logs. Query-time retrieval searches and traverses the concept graph before
  reading source evidence and generating an answer.

  Document limitations:

  - experimental;
  - build may use LLM calls;
  - best suited for multi-hop and synthesis tasks;
  - query is read-only by default;
  - OKF v0.1 is intentionally minimal.

  ## Recommended implementation order

  1. Implement OKF parser, conformance, catalog, and graph utilities.
  2. Add a hand-written OKF fixture and tests.
  3. Implement read-only OKFRAG.load_index and retrieve.
  4. Implement generate and query.
  5. Register okf_rag.
  6. Add deterministic preparation from raw documents.
  7. Add LLM enrichment.
  8. Add conformance/lint reporting.
  9. Add backend readiness checks if needed.
  10. Add docs and evaluation examples.

  This order gives a working OKF consumer first, then adds production quality to OKF generation.
