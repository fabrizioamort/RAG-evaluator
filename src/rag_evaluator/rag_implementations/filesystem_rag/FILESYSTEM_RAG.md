# Filesystem RAG - Implementation Details

Filesystem RAG is an agentic approach to retrieval that treats the filesystem as a structured knowledge base. Instead of relying on vector similarity, it uses an LLM-guided agent to navigate, search, and read documents.

## 1. Filesystem Structure (Stage 1: Preparation)

The `prepare` command creates a structured directory in `data/prepared/filesystem_rag/`:

```
data/prepared/filesystem_rag/
├── _meta/                  # Entry point (What is this corpus?)
│   ├── corpus_overview.md  # High-level description
│   ├── navigation_guide.md # How the agent should use the indexes
│   └── statistics.json     # Corpus metadata
├── _index/                 # Discovery layer (Where should I look?)
│   ├── topics/             # Documents grouped by subject
│   ├── entities/           # Registry of people, organizations, concepts
│   ├── temporal/           # Timeline of events
│   └── questions/          # Question seeds to document mapping
├── _summaries/             # Abstract layer (What's in this doc?)
│   └── doc_XXX_summary.md  # LLM-generated summaries + key sections
└── documents/              # Content layer (Give me the details)
    ├── doc_XXX.md          # Clean Markdown with line numbers
    └── doc_XXX.meta.json   # Detailed document metadata
```

## 2. Agent Logic (Stage 2: Query)

The agent uses a **ReAct (Reason-Act)** loop to answer questions.

### Query Routing
The `QueryRouter` analyzes the incoming question to set a search strategy:
- **KNOWN_ITEM**: Direct lookups (e.g., "Find the API key configuration"). Uses `grep` or question indexes first.
- **EXPLORATORY**: Broad questions (e.g., "Summarize the RAG strategies"). Uses topic maps and summaries first.

### Session Cache
To minimize latency and API calls, core metadata files (`corpus_overview`, `topic_map`, `navigation_guide`) are loaded once per session and provided in the system prompt.

### Progressive Disclosure
To handle large documents without exceeding context windows:
1. Agent reads the `doc_summary`.
2. Agent uses `read_file(headers_only=True)` to see the structure.
3. Agent uses `read_file(start_line=X, end_line=Y)` to read specific sections.

## 3. Toolset

| Tool | Purpose |
|------|---------|
| `list_directory` | Browsing the `_index` or `_summaries` folders. |
| `read_file` | Reading content with line-range and header support. |
| `grep_search` | Rapidly finding specific terms across all Markdown files. |
| `find_files` | Locating documents by ID or pattern. |
| `get_file_info` | Checking file size and line counts. |

## 4. Configuration

Configurable via `.env` or constructor:
- `FILESYSTEM_PREPARED_PATH`: Root for indexed documents.
- `FILESYSTEM_MAX_ITERATIONS`: Limit on agent reasoning steps (default: 10).
- `FILESYSTEM_MAX_TOOL_CALLS`: Limit on tool usage per query (default: 20).
- `FILESYSTEM_MAX_FILE_READS`: Limit on full document reads (default: 10).
- `FILESYSTEM_WORD_THRESHOLD`: threshold for heuristic vs LLM analysis in preparation.
