# Phase 4: Filesystem RAG - Implementation Plan

**Version:** 1.0
**Date:** 2026-01-06
**Status:** Ready for Implementation
**Estimated Duration:** 3 weeks (Weeks 9-11)

---

## Executive Summary

This document provides a comprehensive implementation plan for Phase 4: Filesystem RAG. Unlike traditional RAG approaches that use vector similarity search, Filesystem RAG employs an **LLM-guided agent** that navigates a prepared filesystem structure to find and retrieve relevant information.

### Core Insight

> **Filesystem RAG = Different indexing strategy, not "no indexing"**

The preparation phase creates **navigable file-based indexes** (markdown + JSON) instead of vector embeddings. The agent then explores these indexes and documents using file operations, treating the entire prepared filesystem as a queryable knowledge base.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Index Format | Markdown + JSON | Human-readable, agent-parseable, version-controllable |
| Index Creation | Preparation-time batch | Enables fast query-time navigation |
| LLM Usage in Prep | Hybrid (LLM + heuristics) | Balance cost vs. quality |
| Agent Framework | Custom ReAct loop | Maximum control, minimal dependencies, portfolio value |
| Query Routing | Dual-mode (known-item vs exploratory) | Optimize for different query types |
| Caching Strategy | Session-level index cache | Reduce latency across queries |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Filesystem Structure](#2-filesystem-structure)
3. [Phase 4.1: Preparation Pipeline](#3-phase-41-preparation-pipeline)
4. [Phase 4.2: Agent Implementation](#4-phase-42-agent-implementation)
5. [Phase 4.3: Integration & Testing](#5-phase-43-integration--testing)
6. [Phase 4.4: Evaluation](#6-phase-44-evaluation)
7. [Module Structure](#7-module-structure)
8. [Task Checklist](#8-task-checklist)
9. [Success Criteria](#9-success-criteria)
10. [Appendix: Code Templates](#10-appendix-code-templates)

---

## 1. Architecture Overview

### 1.1 Two-Stage Architecture

The system operates in two distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FILESYSTEM RAG SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: PREPARATION (One-time, at document ingestion)                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Raw Documents → Markdown Conversion → LLM Analysis → Index Build  │ │
│  │                                                                     │ │
│  │  Output: Prepared filesystem with indexes, summaries, metadata     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  STAGE 2: QUERY (Per question, at runtime)                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Question → Query Router → Agent Navigation → Answer Synthesis     │ │
│  │                                                                     │ │
│  │  Agent uses: list_directory, read_file, grep_search, find_files    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Context Engineering Principles

The design follows a **4-layer memory architecture**:

| Layer | Purpose | Filesystem Mapping |
|-------|---------|-------------------|
| Working Memory | Current session context | Active query + files being explored |
| Priority Index | Lightweight pointers | `_index/` directory with topic/entity maps |
| Long-term Storage | Structured entity files | `documents/` with full content + metadata |
| Synthesis Layer | Meta-cognitive overview | `_meta/` with corpus overview and navigation guide |

### 1.3 Design Principles

1. **Progressive Disclosure**: Load metadata first, full content only when needed
2. **Pointer-Based References**: Indexes contain references, not duplicated content
3. **Cognitive Function Labels**: Each index answers a specific type of question
4. **Dual-Mode Navigation**: Different strategies for known-item vs exploratory queries
5. **Session Caching**: Core indexes loaded once per session for low latency
6. **Human Maintainability**: All artifacts are readable markdown/JSON files

---

## 2. Filesystem Structure

### 2.1 Complete Directory Layout

```
data/prepared/filesystem_rag/
│
├── _meta/                              # LAYER 0: Entry Point
│   │                                   # Answers: "What am I working with?"
│   ├── corpus_overview.md              # High-level corpus description
│   ├── navigation_guide.md             # How to use the index structure
│   └── statistics.json                 # Corpus statistics (counts, sizes, dates)
│
├── _index/                             # LAYER 1: Discovery Indexes
│   │                                   # Answers: "Where should I look?"
│   ├── topics/
│   │   ├── _topic_map.md               # Master topic list with doc references
│   │   ├── technical.md                # Technical topic cluster
│   │   ├── business.md                 # Business topic cluster
│   │   ├── science.md                  # Science topic cluster
│   │   └── general.md                  # General knowledge cluster
│   │
│   ├── entities/
│   │   ├── _entity_registry.md         # Master entity list
│   │   ├── people.md                   # People mentioned across docs
│   │   ├── concepts.md                 # Key concepts and definitions
│   │   ├── organizations.md            # Companies, institutions
│   │   └── products.md                 # Products, tools, technologies
│   │
│   ├── temporal/
│   │   └── timeline.md                 # Chronological events/dates
│   │
│   └── questions/
│       └── question_seeds.md           # What questions each doc can answer
│
├── _summaries/                         # LAYER 2: Document Summaries
│   │                                   # Answers: "What does this document contain?"
│   ├── doc_001_summary.md              # Concise summary + key points
│   ├── doc_002_summary.md
│   └── ...
│
├── documents/                          # LAYER 3: Full Content
│   │                                   # Answers: "Give me the details"
│   ├── doc_001.md                      # Full document (converted to markdown)
│   ├── doc_001.meta.json               # Structured metadata
│   ├── doc_002.md
│   ├── doc_002.meta.json
│   └── ...
│
└── _original/                          # LAYER 4: Source Preservation
    ├── doc_001.pdf                     # Original file (reference only)
    └── ...
```

### 2.2 File Format Specifications

#### 2.2.1 corpus_overview.md

```markdown
# Corpus Overview

## Description
This corpus contains [N] documents covering [domains] related to [subject].

## Scope
- Primary topics: [topic1], [topic2], [topic3]
- Time range: [start] - [end]
- Document types: [types]

## Quick Navigation
1. For topic-based search: Start with `_index/topics/_topic_map.md`
2. For entity lookup: Check `_index/entities/_entity_registry.md`
3. For temporal queries: See `_index/temporal/timeline.md`
4. For question matching: See `_index/questions/question_seeds.md`

## Key Statistics
- Total documents: [N]
- Total words: ~[N]
- Primary language: [lang]
- Last updated: [date]
```

#### 2.2.2 navigation_guide.md

```markdown
# Navigation Guide

## Index Structure
- `_index/topics/` - Documents organized by subject matter
- `_index/entities/` - People, concepts, organizations mentioned
- `_index/temporal/` - Timeline of events and dates
- `_index/questions/` - Questions each document can answer

## Recommended Navigation Flow
1. Read `_meta/corpus_overview.md` to understand scope
2. Based on query type:
   - Topical query → `_index/topics/_topic_map.md`
   - Entity query → `_index/entities/_entity_registry.md`
   - Temporal query → `_index/temporal/timeline.md`
   - Direct question → `_index/questions/question_seeds.md`
3. Drill down to specific topic/entity files
4. Read document summaries before full documents
5. Read specific sections of full documents as needed

## File Naming Convention
- `doc_XXX.md` - Converted document content
- `doc_XXX.meta.json` - Structured metadata
- `doc_XXX_summary.md` - Human-readable summary
```

#### 2.2.3 _topic_map.md

```markdown
# Topic Map

## Technical (N documents)
Primary: doc_007, doc_012, doc_023
Secondary: doc_003, doc_015, doc_031
→ Details: [technical.md](topics/technical.md)

## Business (N documents)
Primary: doc_002, doc_008, doc_019
Secondary: doc_005, doc_022
→ Details: [business.md](topics/business.md)

## Science (N documents)
Primary: doc_004, doc_011, doc_028
→ Details: [science.md](topics/science.md)

## General Knowledge (N documents)
Primary: doc_001, doc_006, doc_014
→ Details: [general.md](topics/general.md)
```

#### 2.2.4 Topic Index (e.g., technical.md)

```markdown
# Technical Documents

## RAG Systems
- **doc_007.md** [PRIMARY]
  - Summary: Comprehensive overview of RAG architecture
  - Key sections: Architecture, Chunking Strategies, Evaluation
  - Entities: ChromaDB, LangChain, OpenAI
  - Can answer: "How does RAG work?", "What are RAG challenges?"

- **doc_012.md** [SECONDARY]
  - Summary: RAG performance optimization techniques
  - Key sections: Caching, Latency, Scaling
  - Related: Extends concepts from doc_007

## API Design
- **doc_003.md** [PRIMARY]
  - Summary: REST API design principles
  - Key sections: Authentication, Endpoints, Error Handling

## See Also
- Entities: [concepts.md](../entities/concepts.md#vector-databases)
- Related topics: [science.md](science.md#machine-learning)
```

#### 2.2.5 question_seeds.md

```markdown
# Question Seeds

## Factual Lookups
- "What is RAG?" → doc_007 (section 1)
- "Who created transformers?" → doc_011 (section 2.1)
- "What is ChromaDB?" → doc_007 (section 3.2), doc_023 (section 1)

## How-To Questions
- "How to implement RAG?" → doc_007 (section 4), doc_012 (full)
- "How to optimize embeddings?" → doc_012 (section 2)

## Comparison Questions
- "RAG vs fine-tuning?" → doc_007 (section 5), doc_028 (section 3)
- "ChromaDB vs Pinecone?" → doc_023 (section 4)

## Analysis Questions
- "What are RAG challenges?" → doc_007 (section 6), doc_012 (section 1)
```

#### 2.2.6 Document Summary (doc_XXX_summary.md)

```markdown
# Summary: [Document Title]

**Source:** doc_XXX.md | **Words:** N | **Reading time:** N min

## Overview
[2-3 paragraph summary of key points]

## Key Points
1. [Key point 1]
2. [Key point 2]
3. [Key point 3]

## Main Sections
1. [Section 1 Title] (lines 1-45)
2. [Section 2 Title] (lines 46-120)
3. [Section 3 Title] (lines 121-200)

## Key Entities
- Technologies: [tech1], [tech2]
- Concepts: [concept1], [concept2]
- People: [person1], [person2]

## Related Documents
- doc_XXX.md (relationship description)
- doc_YYY.md (relationship description)

## Questions This Document Answers
- [Question 1]?
- [Question 2]?
```

#### 2.2.7 Document Metadata (doc_XXX.meta.json)

```json
{
  "id": "doc_007",
  "original_file": "rag_introduction.pdf",
  "original_format": "pdf",
  "title": "Introduction to RAG Systems",
  "word_count": 3200,
  "char_count": 18500,
  "line_count": 420,
  "created_date": null,
  "modified_date": "2024-11-15",
  "language": "en",
  "topics": ["rag", "retrieval", "nlp", "vector-search"],
  "topic_scores": {
    "technical": 0.9,
    "science": 0.3
  },
  "entities": {
    "technologies": ["ChromaDB", "LangChain", "OpenAI"],
    "concepts": ["embeddings", "chunking", "semantic-search"],
    "people": [],
    "organizations": ["Anthropic", "OpenAI"]
  },
  "sections": [
    {"title": "Introduction to RAG", "start_line": 1, "end_line": 45},
    {"title": "Architecture Overview", "start_line": 46, "end_line": 120},
    {"title": "Vector Databases", "start_line": 121, "end_line": 200}
  ],
  "summary_path": "_summaries/doc_007_summary.md",
  "related_docs": ["doc_012", "doc_023"],
  "question_seeds": [
    "What is RAG?",
    "How does RAG work?",
    "What are RAG challenges?"
  ]
}
```

---

## 3. Phase 4.1: Preparation Pipeline

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREPARATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: DOCUMENT PROCESSING                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Load Raw   │───▶│  Convert to  │───▶│   Extract    │              │
│  │  Documents   │    │   Markdown   │    │   Metadata   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│        │                    │                    │                       │
│        ▼                    ▼                    ▼                       │
│   PDF,DOCX,TXT       Clean markdown       Basic metadata                │
│                      with structure       (format, size)                │
│                                                                          │
│  STEP 2: LLM-ASSISTED ANALYSIS (Per Document)                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Hybrid Approach:                                                   │ │
│  │  • Simple docs (< 1000 words): Heuristic extraction                │ │
│  │  • Complex docs (≥ 1000 words): LLM analysis                       │ │
│  │                                                                      │ │
│  │  Extract: summary, entities, topics, question seeds, sections      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 3: INDEX SYNTHESIS (After All Documents)                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  1. Cluster topics → Generate topic hierarchy                      │ │
│  │  2. Aggregate entities → Build entity registry                     │ │
│  │  3. Compile timeline from temporal markers                         │ │
│  │  4. Merge question seeds → Create question index                   │ │
│  │  5. Generate corpus overview                                       │ │
│  │  6. Create navigation guide                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 4: VALIDATION                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Verify all documents have summaries                             │ │
│  │  • Check index consistency (no broken references)                  │ │
│  │  • Validate JSON metadata files                                    │ │
│  │  • Generate statistics.json                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Step 1: Document Processing

#### Task 1.1: Load Raw Documents

**Input:** `data/raw/` directory with PDF, DOCX, TXT files
**Output:** List of `RawDocument` objects

```python
@dataclass
class RawDocument:
    id: str                    # e.g., "doc_001"
    original_path: str         # e.g., "data/raw/report.pdf"
    original_format: str       # e.g., "pdf"
    raw_content: str           # Extracted text
    file_size: int             # Bytes
    modified_date: str | None  # ISO date string
```

**Implementation Notes:**

- Reuse existing document loaders from `src/rag_evaluator/common/document_loaders.py`
- Assign sequential IDs: `doc_001`, `doc_002`, etc.
- Handle encoding issues gracefully
- Log warnings for unreadable files

#### Task 1.2: Convert to Markdown

**Input:** `RawDocument` objects
**Output:** Clean markdown files in `documents/` directory

**Conversion Rules:**

1. Preserve heading hierarchy (# ## ###)
2. Preserve lists (-, *, 1.)
3. Preserve code blocks (```)
4. Preserve tables (| | |)
5. Clean excessive whitespace
6. Add line numbers as comments for reference

**Implementation Notes:**

- For TXT: Detect structure heuristically (blank lines = paragraphs, ALL CAPS = headers)
- For PDF: Use existing PDF loader, enhance with structure detection
- For DOCX: Use existing DOCX loader, preserve styles as markdown

#### Task 1.3: Extract Basic Metadata

**Input:** Converted markdown + original file info
**Output:** Initial `doc_XXX.meta.json` files

**Extracted Fields:**

- `id`, `original_file`, `original_format`
- `word_count`, `char_count`, `line_count`
- `modified_date` (from file system if available)
- `language` (detect using simple heuristics or library)

### 3.3 Step 2: LLM-Assisted Analysis

#### Task 2.1: Implement Hybrid Analysis Strategy

**Decision Logic:**

```python
def analyze_document(doc: ProcessedDocument) -> DocumentAnalysis:
    if doc.word_count < 1000:
        return heuristic_analysis(doc)
    else:
        return llm_analysis(doc)
```

#### Task 2.2: Heuristic Analysis (for simple docs)

**Techniques:**

- **Topics:** TF-IDF keywords, header text extraction
- **Entities:** Regex patterns for common entity types (emails, URLs, capitalized phrases)
- **Summary:** First paragraph + sentences containing key terms
- **Questions:** Template-based: "What is [title]?", "How does [main topic] work?"
- **Sections:** Parse markdown headers

**Cost:** Free (no API calls)

#### Task 2.3: LLM Analysis (for complex docs)

**Prompt Template:**

```
Analyze the following document and extract structured information.

DOCUMENT TITLE: {title}
DOCUMENT CONTENT:
{content}

Provide a JSON response with the following structure:
{
  "summary": "2-3 paragraph summary of the document's key points and purpose",
  "topics": ["topic1", "topic2", "topic3"],
  "topic_scores": {"technical": 0.0-1.0, "business": 0.0-1.0, "science": 0.0-1.0, "general": 0.0-1.0},
  "entities": {
    "people": ["Name1", "Name2"],
    "concepts": ["concept1", "concept2"],
    "organizations": ["org1", "org2"],
    "products": ["product1", "product2"]
  },
  "temporal_markers": [
    {"date": "YYYY-MM", "event": "description"}
  ],
  "question_seeds": [
    "Question this document can answer 1?",
    "Question this document can answer 2?",
    "Question this document can answer 3?"
  ],
  "key_sections": [
    {"title": "Section Name", "summary": "Brief description", "start_marker": "first few words"}
  ],
  "related_topics": ["related1", "related2"]
}

Respond ONLY with valid JSON.
```

**Model:** GPT-4o-mini (cost-effective, good enough for extraction)
**Temperature:** 0 (deterministic results)
**Estimated Cost:** ~$0.01-0.02 per document

#### Task 2.4: Generate Document Summary

**Input:** Document content + analysis results
**Output:** `_summaries/doc_XXX_summary.md`

**For heuristic analysis:** Template-based summary from extracted info
**For LLM analysis:** Use the summary from LLM response, format as markdown

#### Task 2.5: Update Document Metadata

**Input:** Analysis results
**Output:** Complete `doc_XXX.meta.json`

Add to existing metadata:

- `topics`, `topic_scores`
- `entities`
- `sections` (with line numbers)
- `question_seeds`
- `summary_path`

### 3.4 Step 3: Index Synthesis

#### Task 3.1: Cluster Topics

**Algorithm:**

1. Collect all `topic_scores` from document metadata
2. Assign each document to topic(s) where score > 0.5
3. Rank documents within topic by score (PRIMARY if > 0.7, SECONDARY otherwise)
4. Handle documents with no strong topic (assign to "general")

**Output:** Topic assignments for each document

#### Task 3.2: Generate Topic Map and Topic Indexes

**_topic_map.md:**

- List all topics with document counts
- Show primary documents for each topic
- Link to detailed topic files

**topics/[topic].md:**

- List all documents in topic
- Include document summaries, key sections, entities
- Add cross-references to related topics

#### Task 3.3: Build Entity Registry

**Algorithm:**

1. Collect all entities from document metadata
2. Group by entity type (people, concepts, organizations, products)
3. For each entity, list all documents where it appears
4. Sort entities by frequency

**Output:**

- `_entity_registry.md` (master list)
- `entities/[type].md` (detailed per-type indexes)

#### Task 3.4: Generate Timeline Index

**Algorithm:**

1. Collect all `temporal_markers` from documents
2. Sort by date
3. Group by year/month
4. Link to source documents

**Output:** `temporal/timeline.md`

#### Task 3.5: Compile Question Seeds Index

**Algorithm:**

1. Collect all `question_seeds` from documents
2. Categorize by type (factual, how-to, comparison, analysis)
3. Link each question to source document(s) and section(s)

**Output:** `questions/question_seeds.md`

#### Task 3.6: Generate Corpus Overview

**Input:** All document metadata, topic assignments, entity counts
**Output:** `_meta/corpus_overview.md`

**Content:**

- Description (synthesized from document summaries)
- Scope (topics, time range, document types)
- Quick navigation guide
- Key statistics

**Option:** Use LLM to synthesize a coherent description from document summaries
**Estimated Cost:** ~$0.10-0.20 (one-time)

#### Task 3.7: Create Navigation Guide

**Output:** `_meta/navigation_guide.md`

**Content:** (Use template from section 2.2.2)

### 3.5 Step 4: Validation

#### Task 4.1: Verify Document Completeness

**Checks:**

- Every document has corresponding summary file
- Every document has valid metadata JSON
- All markdown files are valid UTF-8

#### Task 4.2: Check Index Consistency

**Checks:**

- All document references in indexes exist
- No orphaned documents (not in any index)
- Section references match actual document content

#### Task 4.3: Generate Statistics

**Output:** `_meta/statistics.json`

```json
{
  "generated_at": "2026-01-06T12:00:00Z",
  "total_documents": 45,
  "total_words": 125000,
  "total_characters": 750000,
  "documents_by_format": {
    "pdf": 20,
    "docx": 15,
    "txt": 10
  },
  "documents_by_topic": {
    "technical": 15,
    "business": 12,
    "science": 10,
    "general": 8
  },
  "total_entities": {
    "people": 23,
    "concepts": 67,
    "organizations": 31,
    "products": 45
  },
  "preparation_cost": 0.75,
  "preparation_time_seconds": 120
}
```

---

## 4. Phase 4.2: Agent Implementation

### 4.1 Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FILESYSTEM RAG AGENT                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      SESSION CACHE                               │    │
│  │  Loaded at session start:                                        │    │
│  │  • corpus_overview.md                                            │    │
│  │  • _topic_map.md                                                 │    │
│  │  • _entity_registry.md                                           │    │
│  │  • navigation_guide.md                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      QUERY ROUTER                                │    │
│  │  Analyzes query to determine search mode:                        │    │
│  │  • KNOWN_ITEM: Specific entity/term lookup                       │    │
│  │  • EXPLORATORY: Broad topic exploration                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      REACT AGENT LOOP                            │    │
│  │  While not done and iterations < max:                            │    │
│  │    1. LLM decides next action (tool call or answer)              │    │
│  │    2. Execute tool if needed                                     │    │
│  │    3. Add result to context                                      │    │
│  │    4. Check if ready to answer                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         TOOLS                                    │    │
│  │  • list_directory(path)                                          │    │
│  │  • read_file(path, start_line?, end_line?, headers_only?)        │    │
│  │  • grep_search(pattern, path, file_pattern?)                     │    │
│  │  • find_files(pattern, path)                                     │    │
│  │  • get_file_info(path)                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Tool Specifications

#### Tool 1: list_directory

```python
def list_directory(self, path: str) -> list[dict]:
    """
    List contents of a directory with metadata.

    Args:
        path: Relative path from prepared root (e.g., "_index/topics")

    Returns:
        List of entries, each with:
        - name: str (file or directory name)
        - type: str ("file" or "directory")
        - size: int (bytes, for files only)

    Example:
        list_directory("_index")
        → [
            {"name": "topics", "type": "directory"},
            {"name": "entities", "type": "directory"},
            {"name": "temporal", "type": "directory"},
            {"name": "questions", "type": "directory"}
          ]
    """
```

#### Tool 2: read_file

```python
def read_file(
    self,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    headers_only: bool = False
) -> dict:
    """
    Read file contents with progressive disclosure support.

    Args:
        path: Relative path to file
        start_line: Optional start line (1-indexed)
        end_line: Optional end line (inclusive)
        headers_only: If True and file > 500 lines, return only headers

    Returns:
        - content: str (file content or headers)
        - total_lines: int
        - is_partial: bool (True if start_line/end_line used or headers_only)
        - headers: list[dict] (if headers_only, list of {line, level, text})

    Progressive Disclosure:
        If headers_only=True and file > 500 lines:
        - Returns list of markdown headers with line numbers
        - Agent can then request specific sections

    Example:
        read_file("documents/doc_007.md", headers_only=True)
        → {
            "content": "# Introduction to RAG\n## Architecture\n...",
            "total_lines": 420,
            "is_partial": True,
            "headers": [
              {"line": 1, "level": 1, "text": "Introduction to RAG"},
              {"line": 46, "level": 2, "text": "Architecture"},
              ...
            ]
          }
    """
```

#### Tool 3: grep_search

```python
def grep_search(
    self,
    pattern: str,
    path: str = ".",
    file_pattern: str = "*.md",
    max_results: int = 20
) -> list[dict]:
    """
    Search for pattern in files.

    Args:
        pattern: Regex pattern to search (case-insensitive)
        path: Directory to search in
        file_pattern: Glob pattern for files to search
        max_results: Maximum number of results to return

    Returns:
        List of matches, each with:
        - file: str (relative path)
        - line_number: int
        - content: str (matching line)
        - context: str (surrounding lines)

    Example:
        grep_search("RAG challenges", "_summaries/")
        → [
            {
              "file": "_summaries/doc_007_summary.md",
              "line_number": 45,
              "content": "## RAG Challenges and Solutions",
              "context": "...previous line...\n## RAG Challenges and Solutions\n...next line..."
            }
          ]
    """
```

#### Tool 4: find_files

```python
def find_files(
    self,
    pattern: str,
    path: str = "."
) -> list[str]:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "doc_*.md", "**/summary*.md")
        path: Directory to search in

    Returns:
        List of matching file paths (relative)

    Example:
        find_files("doc_00*.md", "documents/")
        → ["documents/doc_001.md", "documents/doc_002.md", ...]
    """
```

#### Tool 5: get_file_info

```python
def get_file_info(self, path: str) -> dict:
    """
    Get metadata about a file without reading content.

    Args:
        path: Path to file

    Returns:
        - size: int (bytes)
        - lines: int (line count)
        - modified: str (ISO date)
        - type: str (extension)

    Example:
        get_file_info("documents/doc_007.md")
        → {"size": 18500, "lines": 420, "modified": "2026-01-06", "type": "md"}
    """
```

### 4.3 Session Cache Implementation

```python
class SessionCache:
    """
    Load core indexes at session start for low-latency access.
    """

    def __init__(self, prepared_path: str):
        self.prepared_path = prepared_path
        self._cache: dict[str, str] = {}
        self._loaded = False

    def warm(self) -> None:
        """Load core indexes into cache."""
        if self._loaded:
            return

        core_files = [
            "_meta/corpus_overview.md",
            "_meta/navigation_guide.md",
            "_index/topics/_topic_map.md",
            "_index/entities/_entity_registry.md",
        ]

        for file in core_files:
            path = os.path.join(self.prepared_path, file)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._cache[file] = f.read()

        self._loaded = True

    def get(self, file: str) -> str | None:
        """Get cached file content."""
        return self._cache.get(file)

    def get_initial_context(self) -> str:
        """Get concatenated core context for agent system prompt."""
        parts = []
        for file in ["_meta/corpus_overview.md", "_meta/navigation_guide.md"]:
            if content := self._cache.get(file):
                parts.append(f"=== {file} ===\n{content}")
        return "\n\n".join(parts)
```

### 4.4 Query Router Implementation

```python
class QueryRouter:
    """
    Route queries to appropriate search strategy.
    """

    KNOWN_ITEM_PATTERNS = [
        r"where is .+ defined",
        r"find .+ in",
        r"what does .+ say about",
        r"look up",
        r"search for",
    ]

    EXPLORATORY_PATTERNS = [
        r"what are",
        r"how does",
        r"explain",
        r"summarize",
        r"compare",
        r"overview of",
    ]

    def route(self, query: str) -> str:
        """
        Determine search mode based on query.

        Returns:
            "known_item" - Direct grep + targeted reads
            "exploratory" - Navigate indexes first
        """
        query_lower = query.lower()

        # Check for known-item indicators
        for pattern in self.KNOWN_ITEM_PATTERNS:
            if re.search(pattern, query_lower):
                return "known_item"

        # Check for exploratory indicators
        for pattern in self.EXPLORATORY_PATTERNS:
            if re.search(pattern, query_lower):
                return "exploratory"

        # Default to exploratory (safer)
        return "exploratory"

    def get_strategy_hint(self, mode: str) -> str:
        """Get navigation hint for agent based on mode."""
        if mode == "known_item":
            return (
                "This appears to be a known-item search. "
                "Consider using grep_search directly on the query terms, "
                "or check the question_seeds.md index first."
            )
        else:
            return (
                "This appears to be an exploratory query. "
                "Start by consulting the topic map or entity registry "
                "to identify relevant documents before reading."
            )
```

### 4.5 Agent System Prompt

```python
SYSTEM_PROMPT = """You are a Filesystem RAG agent. Your task is to answer questions by navigating a prepared document filesystem.

## Available Tools
- list_directory(path): List files and folders in a directory
- read_file(path, start_line?, end_line?, headers_only?): Read file contents
- grep_search(pattern, path?, file_pattern?): Search for text patterns
- find_files(pattern, path?): Find files by name pattern
- get_file_info(path): Get file metadata without reading content

## Filesystem Structure
```

_meta/           → Corpus overview and navigation guide
_index/          → Topic, entity, temporal, and question indexes
  topics/        → Documents organized by subject
  entities/      → People, concepts, organizations mentioned
  temporal/      → Timeline of events
  questions/     → Query-to-document mapping
_summaries/      → Concise document summaries
documents/       → Full document content with .meta.json metadata

```

## Navigation Strategy
1. Use the cached corpus overview to understand the corpus scope
2. Based on query type:
   - For specific lookups: Check question_seeds.md or use grep_search
   - For topic exploration: Navigate topic indexes first
   - For entity queries: Check entity registry
3. Read summaries before full documents
4. Use headers_only=True for large files to get structure first
5. Read specific line ranges when you know what section you need

## Constraints
- Maximum 20 tool calls per query
- Maximum 10 file reads per query
- Prefer summaries over full documents when sufficient
- Always cite which documents you used

## Response Format
After gathering information, provide:
1. A clear answer to the question
2. Sources: List the documents used (e.g., "doc_007.md, section 3")
3. Confidence: High/Medium/Low based on evidence quality

{strategy_hint}

## Initial Context
{initial_context}
"""
```

### 4.6 ReAct Agent Loop

```python
class FilesystemRAGAgent:
    """LLM-guided filesystem navigation agent."""

    def __init__(
        self,
        llm: Any,  # OpenAI client or similar
        tools: FilesystemRAGTools,
        cache: SessionCache,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        max_file_reads: int = 10
    ):
        self.llm = llm
        self.tools = tools
        self.cache = cache
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_file_reads = max_file_reads
        self.router = QueryRouter()

    def query(self, question: str) -> dict[str, Any]:
        """
        Answer a question by navigating the filesystem.

        Returns:
            {
                "answer": str,
                "context": list[str],
                "metadata": {
                    "files_read": list[str],
                    "tool_calls": int,
                    "reasoning_trace": list[dict],
                    "search_mode": str,
                    "iterations": int
                }
            }
        """
        # Warm cache if needed
        self.cache.warm()

        # Route query
        search_mode = self.router.route(question)
        strategy_hint = self.router.get_strategy_hint(search_mode)

        # Build system prompt
        system_prompt = SYSTEM_PROMPT.format(
            strategy_hint=strategy_hint,
            initial_context=self.cache.get_initial_context()
        )

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        # Tracking
        reasoning_trace = []
        files_read = []
        context_chunks = []
        tool_call_count = 0
        file_read_count = 0

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self._call_llm(messages)

            if response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    # Check limits
                    if tool_call_count >= self.max_tool_calls:
                        messages.append({
                            "role": "system",
                            "content": "Tool call limit reached. Please provide your answer."
                        })
                        break

                    if tool_call.name == "read_file" and file_read_count >= self.max_file_reads:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "File read limit reached. Use information already gathered."
                        })
                        continue

                    # Execute tool
                    result = self._execute_tool(tool_call)
                    tool_call_count += 1

                    if tool_call.name == "read_file":
                        file_read_count += 1
                        files_read.append(tool_call.args.get("path", ""))
                        context_chunks.append(result)

                    # Track reasoning
                    reasoning_trace.append({
                        "iteration": iteration,
                        "tool": tool_call.name,
                        "args": tool_call.args,
                        "result_preview": result[:500] if isinstance(result, str) else str(result)[:500]
                    })

                    # Add result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result if isinstance(result, str) else json.dumps(result)
                    })
            else:
                # LLM provided final answer
                return {
                    "answer": response.content,
                    "context": context_chunks,
                    "metadata": {
                        "files_read": files_read,
                        "tool_calls": tool_call_count,
                        "reasoning_trace": reasoning_trace,
                        "search_mode": search_mode,
                        "iterations": iteration + 1
                    }
                }

        # Max iterations reached - force answer
        return self._synthesize_partial_answer(messages, context_chunks, reasoning_trace, search_mode)

    def _call_llm(self, messages: list[dict]) -> Any:
        """Call LLM with tool definitions."""
        # Implementation depends on LLM client
        pass

    def _execute_tool(self, tool_call: Any) -> str | dict:
        """Execute a tool and return result."""
        tool_name = tool_call.name
        args = tool_call.args

        tool_map = {
            "list_directory": self.tools.list_directory,
            "read_file": self.tools.read_file,
            "grep_search": self.tools.grep_search,
            "find_files": self.tools.find_files,
            "get_file_info": self.tools.get_file_info,
        }

        if tool_name in tool_map:
            try:
                return tool_map[tool_name](**args)
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    def _synthesize_partial_answer(
        self,
        messages: list[dict],
        context_chunks: list[str],
        reasoning_trace: list[dict],
        search_mode: str
    ) -> dict[str, Any]:
        """Generate answer when max iterations reached."""
        messages.append({
            "role": "system",
            "content": "Maximum iterations reached. Please provide your best answer based on information gathered so far."
        })
        response = self._call_llm(messages)

        return {
            "answer": response.content,
            "context": context_chunks,
            "metadata": {
                "files_read": [t["args"].get("path") for t in reasoning_trace if t["tool"] == "read_file"],
                "tool_calls": len(reasoning_trace),
                "reasoning_trace": reasoning_trace,
                "search_mode": search_mode,
                "iterations": self.max_iterations,
                "max_iterations_reached": True
            }
        }
```

### 4.7 Navigation Flow Example

```
Query: "What are the main challenges with RAG implementations?"

Search Mode: exploratory

Iteration 1:
  Thought: "This is an exploratory query about RAG challenges.
           Let me check the question seeds first."
  Tool: read_file("_index/questions/question_seeds.md")
  Result: [Found "What are RAG challenges?" → doc_007 (section 6), doc_012 (section 1)]

Iteration 2:
  Thought: "Found relevant documents. Let me read the summary of doc_007 first."
  Tool: read_file("_summaries/doc_007_summary.md")
  Result: [Summary shows section 6 covers "Common Challenges"]

Iteration 3:
  Thought: "The summary confirms doc_007 section 6 is relevant. Let me read that section."
  Tool: read_file("documents/doc_007.md", start_line=351, end_line=420)
  Result: [Detailed content about RAG challenges: chunking, retrieval accuracy, latency...]

Iteration 4:
  Thought: "Good information from doc_007. Let me also check doc_012 for additional challenges."
  Tool: read_file("_summaries/doc_012_summary.md")
  Result: [Summary shows doc_012 focuses on performance challenges specifically]

Iteration 5:
  Thought: "I have enough information to answer comprehensively."

Final Answer:
  "The main challenges with RAG implementations include:

   1. **Chunking Strategy Selection** (doc_007, section 4)
      - Choosing optimal chunk size affects retrieval quality
      - Too small: loses context; too large: reduces precision

   2. **Retrieval Accuracy vs Latency Tradeoff** (doc_012)
      - More sophisticated retrieval is slower
      - Production systems need to balance quality and speed

   3. **Handling Ambiguous Queries** (doc_007, section 6)
      - Unclear user intent leads to irrelevant retrievals
      - May need query expansion or clarification

   4. **Evaluation Complexity** (doc_007, section 5)
      - Standard metrics don't capture all aspects of RAG quality
      - Requires specialized metrics like faithfulness and relevancy

   Sources: doc_007.md (sections 4, 5, 6), doc_012.md (section 1)
   Confidence: High"
```

---

## 5. Phase 4.3: Integration & Testing

### 5.1 BaseRAG Implementation

```python
class FilesystemRAG(BaseRAG):
    """
    Filesystem-based RAG implementation.

    Uses LLM-guided navigation of a prepared filesystem
    instead of vector similarity search.
    """

    def __init__(self) -> None:
        super().__init__("Filesystem RAG")
        self.prepared_path: str | None = None
        self.agent: FilesystemRAGAgent | None = None
        self.cache: SessionCache | None = None
        self.preparation_metrics: dict[str, Any] = {}
        self.query_metrics: list[dict[str, Any]] = []

    def prepare_documents(self, documents_path: str) -> None:
        """
        Prepare documents by converting to markdown and building indexes.

        Args:
            documents_path: Path to raw documents directory
        """
        # Set prepared path
        self.prepared_path = "data/prepared/filesystem_rag"

        # Run preparation pipeline
        pipeline = PreparationPipeline(
            input_path=documents_path,
            output_path=self.prepared_path
        )
        self.preparation_metrics = pipeline.run()

        # Initialize cache and agent
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the agent with tools and cache."""
        if not self.prepared_path:
            raise ValueError("Documents not prepared")

        self.cache = SessionCache(self.prepared_path)
        self.cache.warm()

        tools = FilesystemRAGTools(self.prepared_path)
        llm = self._create_llm()

        self.agent = FilesystemRAGAgent(
            llm=llm,
            tools=tools,
            cache=self.cache
        )

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """
        Answer a question using filesystem navigation.

        Args:
            question: The question to answer
            top_k: Ignored for filesystem RAG (included for interface compatibility)

        Returns:
            {
                "answer": str,
                "context": list[str],
                "metadata": dict
            }
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call prepare_documents first.")

        start_time = time.time()
        result = self.agent.query(question)
        query_time = time.time() - start_time

        # Track metrics
        query_metric = {
            "question": question,
            "query_time": query_time,
            "tool_calls": result["metadata"]["tool_calls"],
            "files_read": len(result["metadata"]["files_read"]),
            "search_mode": result["metadata"]["search_mode"],
            "iterations": result["metadata"]["iterations"]
        }
        self.query_metrics.append(query_metric)

        # Add timing to metadata
        result["metadata"]["retrieval_time"] = query_time
        result["metadata"]["chunks_retrieved"] = len(result["context"])

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Return performance metrics."""
        if not self.query_metrics:
            return {
                "total_queries": 0,
                "preparation_metrics": self.preparation_metrics
            }

        return {
            "total_queries": len(self.query_metrics),
            "avg_query_time": sum(m["query_time"] for m in self.query_metrics) / len(self.query_metrics),
            "avg_tool_calls": sum(m["tool_calls"] for m in self.query_metrics) / len(self.query_metrics),
            "avg_files_read": sum(m["files_read"] for m in self.query_metrics) / len(self.query_metrics),
            "search_mode_distribution": self._count_search_modes(),
            "preparation_metrics": self.preparation_metrics
        }

    def _count_search_modes(self) -> dict[str, int]:
        """Count queries by search mode."""
        counts: dict[str, int] = {}
        for m in self.query_metrics:
            mode = m["search_mode"]
            counts[mode] = counts.get(mode, 0) + 1
        return counts
```

### 5.2 Unit Tests

#### Test File: tests/unit/test_filesystem_rag.py

```python
"""Unit tests for Filesystem RAG components."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os

from rag_evaluator.rag_implementations.filesystem_rag import (
    FilesystemRAG,
    FilesystemRAGTools,
    SessionCache,
    QueryRouter,
    PreparationPipeline
)


class TestQueryRouter:
    """Tests for query routing logic."""

    def test_known_item_query(self):
        router = QueryRouter()
        assert router.route("Where is authentication defined?") == "known_item"
        assert router.route("Find the API rate limits") == "known_item"
        assert router.route("Search for error handling") == "known_item"

    def test_exploratory_query(self):
        router = QueryRouter()
        assert router.route("What are the main challenges with RAG?") == "exploratory"
        assert router.route("How does the authentication system work?") == "exploratory"
        assert router.route("Explain the architecture overview") == "exploratory"

    def test_default_to_exploratory(self):
        router = QueryRouter()
        assert router.route("RAG performance") == "exploratory"


class TestSessionCache:
    """Tests for session-level caching."""

    def test_cache_warm(self, tmp_prepared_filesystem):
        cache = SessionCache(tmp_prepared_filesystem)
        cache.warm()

        assert cache._loaded is True
        assert "_meta/corpus_overview.md" in cache._cache

    def test_cache_get(self, tmp_prepared_filesystem):
        cache = SessionCache(tmp_prepared_filesystem)
        cache.warm()

        content = cache.get("_meta/corpus_overview.md")
        assert content is not None
        assert "Corpus Overview" in content

    def test_cache_miss(self, tmp_prepared_filesystem):
        cache = SessionCache(tmp_prepared_filesystem)
        cache.warm()

        content = cache.get("nonexistent.md")
        assert content is None


class TestFilesystemRAGTools:
    """Tests for filesystem navigation tools."""

    def test_list_directory(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.list_directory("_index")

        assert len(result) > 0
        assert any(item["name"] == "topics" for item in result)

    def test_read_file_full(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.read_file("_meta/corpus_overview.md")

        assert "content" in result
        assert "total_lines" in result
        assert result["is_partial"] is False

    def test_read_file_line_range(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.read_file("_meta/corpus_overview.md", start_line=1, end_line=5)

        assert result["is_partial"] is True

    def test_read_file_headers_only(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        # Need a file with >500 lines for this test
        result = tools.read_file("documents/large_doc.md", headers_only=True)

        assert "headers" in result

    def test_grep_search(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.grep_search("RAG", "_summaries/")

        assert isinstance(result, list)

    def test_find_files(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.find_files("doc_*.md", "documents/")

        assert isinstance(result, list)

    def test_get_file_info(self, tmp_prepared_filesystem):
        tools = FilesystemRAGTools(tmp_prepared_filesystem)
        result = tools.get_file_info("_meta/corpus_overview.md")

        assert "size" in result
        assert "lines" in result


@pytest.fixture
def tmp_prepared_filesystem(tmp_path):
    """Create a minimal prepared filesystem for testing."""
    # Create directory structure
    (tmp_path / "_meta").mkdir()
    (tmp_path / "_index" / "topics").mkdir(parents=True)
    (tmp_path / "_index" / "entities").mkdir(parents=True)
    (tmp_path / "_summaries").mkdir()
    (tmp_path / "documents").mkdir()

    # Create test files
    (tmp_path / "_meta" / "corpus_overview.md").write_text(
        "# Corpus Overview\n\nTest corpus for unit testing."
    )
    (tmp_path / "_meta" / "navigation_guide.md").write_text(
        "# Navigation Guide\n\nHow to navigate this test corpus."
    )
    (tmp_path / "_index" / "topics" / "_topic_map.md").write_text(
        "# Topic Map\n\n## Technical\n- doc_001"
    )
    (tmp_path / "_index" / "entities" / "_entity_registry.md").write_text(
        "# Entity Registry\n\n## Concepts\n- RAG"
    )
    (tmp_path / "_summaries" / "doc_001_summary.md").write_text(
        "# Summary: Test Document\n\nThis is a test RAG document."
    )
    (tmp_path / "documents" / "doc_001.md").write_text(
        "# Test Document\n\nContent about RAG.\n\n## Section 1\n\nMore content."
    )

    # Create a large file for headers_only test
    large_content = "# Large Document\n\n" + "\n".join([f"Line {i}" for i in range(600)])
    (tmp_path / "documents" / "large_doc.md").write_text(large_content)

    return str(tmp_path)
```

### 5.3 Integration Tests

#### Test File: tests/integration/test_filesystem_rag_integration.py

```python
"""Integration tests for Filesystem RAG."""

import pytest
import os
import tempfile
import shutil

from rag_evaluator.rag_implementations.filesystem_rag import FilesystemRAG


@pytest.fixture
def sample_documents(tmp_path):
    """Create sample documents for testing."""
    docs_path = tmp_path / "raw"
    docs_path.mkdir()

    # Create test documents
    (docs_path / "rag_overview.txt").write_text("""
    Introduction to RAG Systems

    Retrieval Augmented Generation (RAG) is a technique that combines
    retrieval and generation for grounded responses.

    Key Components:
    1. Document Store - Stores and indexes documents
    2. Retriever - Finds relevant documents
    3. Generator - Produces answers using retrieved context

    Common Challenges:
    - Chunking strategy selection
    - Retrieval accuracy
    - Latency optimization
    """)

    (docs_path / "api_reference.txt").write_text("""
    API Reference Guide

    Authentication:
    All API calls require an API key in the Authorization header.

    Endpoints:
    - POST /query - Submit a question
    - GET /documents - List all documents
    - POST /documents - Add a new document

    Rate Limits:
    - 100 requests per minute
    - 1000 requests per day
    """)

    return str(docs_path)


class TestFilesystemRAGIntegration:
    """Integration tests for full Filesystem RAG workflow."""

    @pytest.mark.integration
    def test_prepare_and_query(self, sample_documents):
        """Test full workflow: prepare documents and run queries."""
        rag = FilesystemRAG()

        # Prepare documents
        rag.prepare_documents(sample_documents)

        # Verify preparation created expected structure
        assert rag.prepared_path is not None
        assert os.path.exists(os.path.join(rag.prepared_path, "_meta", "corpus_overview.md"))
        assert os.path.exists(os.path.join(rag.prepared_path, "_index", "topics", "_topic_map.md"))

        # Run a query
        result = rag.query("What are the main challenges with RAG?")

        assert "answer" in result
        assert "context" in result
        assert "metadata" in result
        assert len(result["answer"]) > 0

    @pytest.mark.integration
    def test_known_item_query(self, sample_documents):
        """Test known-item search mode."""
        rag = FilesystemRAG()
        rag.prepare_documents(sample_documents)

        result = rag.query("Find the API rate limits")

        assert "100 requests" in result["answer"] or "rate limit" in result["answer"].lower()
        assert result["metadata"]["search_mode"] == "known_item"

    @pytest.mark.integration
    def test_exploratory_query(self, sample_documents):
        """Test exploratory search mode."""
        rag = FilesystemRAG()
        rag.prepare_documents(sample_documents)

        result = rag.query("What is RAG and how does it work?")

        assert result["metadata"]["search_mode"] == "exploratory"
        assert "retrieval" in result["answer"].lower() or "generation" in result["answer"].lower()

    @pytest.mark.integration
    def test_metrics_tracking(self, sample_documents):
        """Test that metrics are properly tracked."""
        rag = FilesystemRAG()
        rag.prepare_documents(sample_documents)

        # Run multiple queries
        rag.query("What is RAG?")
        rag.query("What are the API endpoints?")

        metrics = rag.get_metrics()

        assert metrics["total_queries"] == 2
        assert "avg_query_time" in metrics
        assert "avg_tool_calls" in metrics
        assert "preparation_metrics" in metrics
```

---

## 6. Phase 4.4: Evaluation

### 6.1 Filesystem-Specific Metrics

In addition to standard DeepEval metrics, track these filesystem-specific metrics:

```python
@dataclass
class FilesystemRAGEvaluationMetrics:
    """Metrics specific to Filesystem RAG evaluation."""

    # Standard RAG metrics (from DeepEval)
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float

    # Filesystem-specific metrics
    navigation_efficiency: float  # relevant_files_read / total_files_read
    index_hit_rate: float         # queries where indexes led to answer / total queries
    tool_call_efficiency: float   # useful_tool_calls / total_tool_calls

    # Behavioral metrics
    avg_iterations: float         # average ReAct loop iterations
    avg_tool_calls: float         # average tool calls per query
    avg_files_read: float         # average files read per query
    known_item_accuracy: float    # accuracy on known-item queries
    exploratory_accuracy: float   # accuracy on exploratory queries

    # Cost metrics
    preparation_cost: float       # LLM cost for preparation
    avg_query_cost: float         # average LLM cost per query
    total_cost: float             # total evaluation cost
```

### 6.2 Evaluation Strategy

#### 6.2.1 Test Case Selection

Create test cases that play to Filesystem RAG's strengths:

| Query Type | Example | Why Filesystem RAG Should Excel |
|------------|---------|--------------------------------|
| Cross-document | "Compare approaches in doc A and doc B" | Can read multiple full documents |
| Metadata-based | "Documents about X from 2024" | JSON metadata is queryable |
| Navigation | "Find information about X in technical docs" | Leverages folder structure |
| Explicit reference | "What does section 3.2 say?" | Can follow references directly |
| Multi-hop | "Who wrote the doc that covers Y?" | Can traverse document relationships |

#### 6.2.2 Comparative Evaluation

Run same test set against all 4 RAG implementations:

```python
def run_comparative_evaluation():
    test_cases = load_test_cases("data/test_set.json")

    rags = [
        VectorSemanticRAG(),
        HybridSearchRAG(),
        Neo4jGraphRAG(),
        FilesystemRAG()
    ]

    results = {}
    for rag in rags:
        rag.prepare_documents("data/raw")
        results[rag.name] = evaluate_rag(rag, test_cases)

    generate_comparison_report(results)
```

### 6.3 Success Criteria

| Metric | Minimum Threshold | Target |
|--------|-------------------|--------|
| Faithfulness | 0.70 | 0.80 |
| Answer Relevancy | 0.70 | 0.80 |
| Contextual Precision | 0.60 | 0.75 |
| Contextual Recall | 0.60 | 0.75 |
| Navigation Efficiency | 0.50 | 0.70 |
| Avg Query Time | < 10s | < 5s |
| Avg Tool Calls | < 15 | < 10 |

### 6.4 Evaluation Report Sections

The final evaluation report should include:

1. **Executive Summary**: How Filesystem RAG compares to other approaches
2. **Metric Comparison Table**: All 4 RAGs side-by-side
3. **Filesystem-Specific Analysis**:
   - Navigation patterns (which indexes were most useful)
   - Search mode effectiveness (known-item vs exploratory)
   - Tool usage patterns
4. **Scenario Analysis**: Where Filesystem RAG wins/loses
5. **Cost Analysis**: Preparation + query costs compared to alternatives
6. **Reasoning Trace Examples**: Show agent navigation for sample queries

---

## 7. Module Structure

```
src/rag_evaluator/rag_implementations/filesystem_rag/
├── __init__.py                     # Public exports
├── filesystem_rag.py               # Main FilesystemRAG class (BaseRAG impl)
│
├── preparation/
│   ├── __init__.py
│   ├── pipeline.py                 # PreparationPipeline orchestrator
│   ├── document_processor.py       # Load and convert documents to markdown
│   ├── analyzer.py                 # LLM + heuristic analysis
│   ├── index_builder.py            # Generate index files
│   └── synthesizer.py              # Corpus-level synthesis
│
├── agent/
│   ├── __init__.py
│   ├── tools.py                    # FilesystemRAGTools class
│   ├── cache.py                    # SessionCache class
│   ├── router.py                   # QueryRouter class
│   ├── agent.py                    # FilesystemRAGAgent class
│   └── prompts.py                  # System prompts and templates
│
└── utils/
    ├── __init__.py
    ├── metrics.py                  # FilesystemRAGMetrics dataclass
    └── validation.py               # Index validation utilities
```

---

## 8. Task Checklist

### Week 9: Preparation Pipeline

#### Day 1-2: Document Processing

- [x] **Task 1.1**: Create `document_processor.py`
  - [x] Implement `RawDocument` dataclass
  - [x] Implement `load_documents()` function using existing loaders
  - [x] Implement `convert_to_markdown()` for each format
  - [x] Add structure detection for TXT files
  - [x] Write unit tests

- [x] **Task 1.2**: Create `analyzer.py`
  - [x] Implement `DocumentAnalysis` dataclass
  - [x] Implement `heuristic_analysis()` for simple docs
  - [x] Implement `llm_analysis()` for complex docs
  - [x] Add hybrid decision logic
  - [x] Write unit tests

#### Day 3-4: Index Generation

- [x] **Task 1.3**: Create `index_builder.py`
  - [x] Implement `build_topic_map()`
  - [x] Implement `build_topic_indexes()`
  - [x] Implement `build_entity_registry()`
  - [x] Implement `build_question_seeds()`
  - [x] Implement `build_timeline()` (if temporal data exists)
  - [x] Write unit tests

- [x] **Task 1.4**: Create `synthesizer.py`
  - [x] Implement `generate_corpus_overview()`
  - [x] Implement `generate_navigation_guide()`
  - [x] Implement `generate_statistics()`
  - [x] Write unit tests

#### Day 5: Pipeline Integration

- [x] **Task 1.5**: Create `pipeline.py`
  - [x] Implement `PreparationPipeline` class
  - [x] Orchestrate all steps
  - [x] Add progress logging
  - [x] Add cost tracking
  - [x] Implement validation step
  - [x] Write integration tests

### Week 10: Agent Implementation

#### Day 1-2: Core Components

- [ ] **Task 2.1**: Create `tools.py`
  - [ ] Implement `FilesystemRAGTools` class
  - [ ] Implement `list_directory()`
  - [ ] Implement `read_file()` with progressive disclosure
  - [ ] Implement `grep_search()`
  - [ ] Implement `find_files()`
  - [ ] Implement `get_file_info()`
  - [ ] Write unit tests for each tool

- [ ] **Task 2.2**: Create `cache.py`
  - [ ] Implement `SessionCache` class
  - [ ] Implement `warm()` method
  - [ ] Implement `get()` method
  - [ ] Implement `get_initial_context()` method
  - [ ] Write unit tests

#### Day 3-4: Agent Logic

- [ ] **Task 2.3**: Create `router.py`
  - [ ] Implement `QueryRouter` class
  - [ ] Define known-item patterns
  - [ ] Define exploratory patterns
  - [ ] Implement `route()` method
  - [ ] Implement `get_strategy_hint()` method
  - [ ] Write unit tests

- [ ] **Task 2.4**: Create `prompts.py`
  - [ ] Define `SYSTEM_PROMPT` template
  - [ ] Define tool descriptions for LLM
  - [ ] Add formatting functions
  - [ ] Write tests for prompt generation

- [ ] **Task 2.5**: Create `agent.py`
  - [ ] Implement `FilesystemRAGAgent` class
  - [ ] Implement ReAct loop
  - [ ] Add tool execution logic
  - [ ] Add limit enforcement (tool calls, file reads)
  - [ ] Implement partial answer synthesis
  - [ ] Write unit tests

#### Day 5: BaseRAG Integration

- [ ] **Task 2.6**: Create `filesystem_rag.py`
  - [ ] Implement `FilesystemRAG(BaseRAG)` class
  - [ ] Implement `prepare_documents()`
  - [ ] Implement `query()`
  - [ ] Implement `get_metrics()`
  - [ ] Write integration tests

### Week 11: Testing & Evaluation

#### Day 1-2: Comprehensive Testing

- [ ] **Task 3.1**: Complete unit test coverage
  - [ ] Achieve >80% coverage on all modules
  - [ ] Test edge cases (empty files, missing indexes, etc.)
  - [ ] Test error handling

- [ ] **Task 3.2**: Complete integration tests
  - [ ] Test full prepare → query workflow
  - [ ] Test with various document types
  - [ ] Test both search modes

- [ ] **Task 3.3**: Add to CLI
  - [ ] Add `filesystem` to `--rag-type` choices in `cli.py`
  - [ ] Test CLI integration

- [ ] **Task 3.4**: Add to UI
  - [ ] Add Filesystem RAG to Streamlit options
  - [ ] Add reasoning trace visualization

#### Day 3-4: Evaluation

- [ ] **Task 3.5**: Run evaluation
  - [ ] Prepare full test document corpus
  - [ ] Run DeepEval evaluation
  - [ ] Collect filesystem-specific metrics
  - [ ] Compare with other RAG implementations

- [ ] **Task 3.6**: Generate evaluation report
  - [ ] Create comparison tables
  - [ ] Document reasoning trace examples
  - [ ] Analyze scenarios where Filesystem RAG excels
  - [ ] Document known limitations

#### Day 5: Documentation & Polish

- [ ] **Task 3.7**: Update documentation
  - [ ] Update README with Filesystem RAG section
  - [ ] Document configuration options
  - [ ] Add usage examples
  - [ ] Document filesystem structure

- [ ] **Task 3.8**: Final polish
  - [ ] Run all quality checks (ruff, mypy)
  - [ ] Fix any remaining issues
  - [ ] Update SPEC.md to mark Phase 4 complete

---

## 9. Success Criteria

### 9.1 Functional Requirements

- [ ] FilesystemRAG implements BaseRAG interface correctly
- [ ] Preparation pipeline creates valid filesystem structure
- [ ] All 5 agent tools work correctly
- [ ] Session caching reduces latency
- [ ] Query routing improves search efficiency
- [ ] Progressive disclosure works for large files
- [ ] Agent respects tool call and file read limits

### 9.2 Quality Requirements

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage >80%
- [ ] No ruff linting errors
- [ ] No mypy type errors
- [ ] All functions have type hints and docstrings

### 9.3 Performance Requirements

- [ ] Preparation completes in <5 minutes for 50 documents
- [ ] Average query time <10 seconds
- [ ] Average tool calls per query <15
- [ ] Faithfulness score >0.70
- [ ] Answer Relevancy score >0.70

### 9.4 Comparative Requirements

- [ ] Outperforms Vector RAG on cross-document reasoning queries
- [ ] Outperforms Vector RAG on metadata-based queries
- [ ] Demonstrates interpretable reasoning traces
- [ ] Shows unique value proposition vs other RAG types

---

## 10. Appendix: Code Templates

### 10.1 LLM Analysis Prompt

```python
ANALYSIS_PROMPT = """Analyze the following document and extract structured information.

DOCUMENT TITLE: {title}
DOCUMENT CONTENT:
{content}

Provide a JSON response with the following structure:
{{
  "summary": "2-3 paragraph summary of the document's key points and purpose",
  "topics": ["topic1", "topic2", "topic3"],
  "topic_scores": {{
    "technical": 0.0,
    "business": 0.0,
    "science": 0.0,
    "general": 0.0
  }},
  "entities": {{
    "people": ["Name1", "Name2"],
    "concepts": ["concept1", "concept2"],
    "organizations": ["org1", "org2"],
    "products": ["product1", "product2"]
  }},
  "temporal_markers": [
    {{"date": "YYYY-MM", "event": "description"}}
  ],
  "question_seeds": [
    "Question this document can answer 1?",
    "Question this document can answer 2?",
    "Question this document can answer 3?",
    "Question this document can answer 4?",
    "Question this document can answer 5?"
  ],
  "key_sections": [
    {{"title": "Section Name", "summary": "Brief description", "start_marker": "first few words"}}
  ],
  "related_topics": ["related1", "related2"]
}}

Rules:
- topic_scores should sum to approximately 1.0
- Include 5-10 question_seeds covering different aspects
- Be specific in entity extraction
- Only include temporal_markers if dates are mentioned

Respond ONLY with valid JSON, no other text.
"""
```

### 10.2 Corpus Synthesis Prompt

```python
SYNTHESIS_PROMPT = """Based on the following document summaries and metadata, create a corpus overview.

DOCUMENTS:
{documents_info}

Create a comprehensive overview that includes:

1. A high-level description of what this corpus contains (2-3 paragraphs)
2. The main themes and topics covered
3. The scope and any limitations
4. Recommended navigation strategies for different query types

Format your response as markdown with clear sections.
"""
```

### 10.3 Tool Definitions for OpenAI

```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from prepared root (e.g., '_index/topics')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. Use headers_only=True for large files to get structure first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (inclusive)"
                    },
                    "headers_only": {
                        "type": "boolean",
                        "description": "If true and file >500 lines, return only headers with line numbers"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern in files. Returns matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search (case-insensitive)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for files to search (default: *.md)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., 'doc_*.md', '**/summary*.md')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "Get metadata about a file without reading its content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file"
                    }
                },
                "required": ["path"]
            }
        }
    }
]
```

---

## References

- [SPEC.md](SPEC.md) - Project specification
- [Phase4_brainstorming_Opus.md](Phase4_brainstorming_Opus.md) - Primary design document
- [Phase4_brainstorming_ChatGPT.md](Phase4_brainstorming_ChatGPT.md) - Cognitive function mapping
- [Phase4_brainstorming_KimiK2.md](Phase4_brainstorming_KimiK2.md) - Session caching and dual-mode routing
- [Phase4_brainstorming_Gemini.md](Phase4_brainstorming_Gemini.md) - Progressive disclosure and custom metrics
- [Phase4_brainstorming_Sonnet.md](Phase4_brainstorming_Sonnet.md) - MVP scoping and index evolution

---

**Document Status:** Ready for Implementation
**Created:** 2026-01-06
**Next Step:** Begin Week 9, Day 1-2 tasks (Document Processing)
