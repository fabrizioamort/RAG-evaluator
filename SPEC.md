# RAG Evaluator - Technical Specification

**Version:** 1.0
**Date:** 2026-01-03
**Status:** Draft

## Executive Summary

RAG Evaluator is a production-ready Python framework for comparing and evaluating four distinct RAG (Retrieval Augmented Generation) architectures. The project serves as both a technical portfolio piece and a community resource for understanding RAG performance tradeoffs across different approaches.

**Core Value Proposition:** Provide objective, reproducible comparisons of RAG architectures using standardized evaluation metrics, enabling informed decisions about RAG implementation strategies.

## Project Goals

### Primary Goals
1. **Working Implementations** - Deliver 4 fully functional RAG implementations with comprehensive tests
2. **Portfolio Impact** - Demonstrate technical depth, production readiness, and full-stack capability
3. **Learning Outcomes** - Deep understanding of RAG architectures, tradeoffs, and evaluation methodologies
4. **Community Contribution** - Provide useful benchmarks and insights to the broader RAG development community

### Success Criteria
- ✅ All 4 RAG types implemented and functional
- ✅ Each RAG meets minimum performance thresholds:
  - Accuracy: DeepEval scores >0.7 (faithfulness)
  - Latency: Query response time <10 seconds
  - Cost: Reasonable API costs relative to performance
- ✅ Comprehensive evaluation framework with statistical analysis
- ✅ Professional documentation and portfolio presentation
- ✅ GitHub repository generates community interest (stars, forks, discussions)

## Architecture Overview

### Design Principles

1. **Abstract Base Class Pattern** - All RAG implementations inherit from `BaseRAG` interface
2. **Fair Comparison** - Standardized evaluation methodology with equal treatment across implementations
3. **RAG-Optimized Configuration** - Each RAG type uses its optimal chunking and configuration strategy
4. **Production Quality** - CI/CD, type checking, comprehensive testing, proper error handling
5. **Cloud-Ready** - Designed for cloud deployment without requiring immediate deployment

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Evaluator System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         4 RAG Implementations (BaseRAG)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  1. Vector Semantic (ChromaDB)           [COMPLETE]  │  │
│  │  2. Hybrid Search (Qdrant/Weaviate)      [TODO]      │  │
│  │  3. Graph RAG (Microsoft GraphRAG/Graphiti) [TODO]   │  │
│  │  4. Filesystem RAG (Agentic Explorer)    [TODO]      │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Evaluation Framework (DeepEval)                 │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Faithfulness (answer from context)                │  │
│  │  • Answer Relevancy (addresses question)             │  │
│  │  • Contextual Precision (right docs retrieved)       │  │
│  │  • Contextual Recall (all relevant info retrieved)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Report Generation                          │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Executive Summary                                 │  │
│  │  • Per-Query Breakdown                               │  │
│  │  • Failure Analysis                                  │  │
│  │  • Statistical Analysis (CI, significance tests)     │  │
│  │  • JSON + Markdown outputs                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Streamlit UI (Pre-computed Results)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Side-by-side comparison tables                    │  │
│  │  • Metric visualization                              │  │
│  │  • Query-level drill-down                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## RAG Implementation Specifications

### 1. Vector Semantic Search (ChromaDB) ✅ COMPLETE

**Status:** Production-ready
**Database:** ChromaDB (local)
**Embeddings:** OpenAI text-embedding-3-small

**Features:**
- Document loading from multiple formats (TXT, PDF, DOCX)
- RecursiveCharacterTextSplitter chunking (configurable size/overlap)
- Cosine similarity search
- LLM-based answer generation with retrieved context
- Performance metrics tracking

**Current Implementation:** 224 lines, 97% test coverage, fully type-hinted

### 2. Hybrid Search RAG 🔲 TODO

**Technology Stack:**
- **Primary Option:** Qdrant (free, open-source, native hybrid search)
- **Alternative:** Weaviate (if Qdrant integration proves difficult)
- **Fallback:** LangChain EnsembleRetriever (most flexible)

**Technical Approach:**
- Dense vector embeddings (OpenAI text-embedding-3-small)
- Sparse vectors for keyword matching (BM25)
- Score fusion: Use Reciprocal Rank Fusion (RRF) as default
  - No normalization required
  - Production-proven approach
  - Simpler than weighted combination

**Chunking Strategy:**
- Smaller chunks than vector-only (500-800 chars) for better keyword matching
- Lower overlap (100 chars) to reduce redundancy
- Preserve sentence boundaries

**Configuration:**
```python
{
    "chunk_size": 700,
    "chunk_overlap": 100,
    "top_k_semantic": 10,
    "top_k_keyword": 10,
    "fusion_method": "rrf",
    "rrf_k": 60  # Standard RRF constant
}
```

**Acceptance Criteria:**
- Successfully combines semantic and keyword search
- Outperforms pure vector search on keyword-heavy queries
- Query latency <3 seconds
- Setup requires minimal configuration (no paid services)

### 3. Graph RAG 🔲 TODO

**Technology Stack:**
- **Framework:** Microsoft GraphRAG OR Graphiti (choose based on ease of implementation)
- **Database:** Neo4j (Community Edition, free)
- **LLM:** OpenAI GPT-4o-mini for entity extraction
- **Embeddings:** OpenAI text-embedding-3-small for node embeddings

**Implementation Level:**
- Use existing framework (not building from scratch)
- Focus on integration and evaluation, not novel graph algorithms
- Minimal viable implementation that demonstrates graph-based retrieval

**Microsoft GraphRAG Approach (if selected):**
- Community detection for hierarchical summarization
- Global search: Use community summaries for broad questions
- Local search: Traverse graph for specific entity queries
- Configurable indexing pipeline

**Graphiti Approach (if selected):**
- Temporal knowledge graphs
- Dynamic entity resolution
- Episodic memory patterns

**Chunking Strategy:**
- Larger chunks (1500-2000 chars) to capture entity context
- Entity-aware chunking: Don't split mid-entity
- May use semantic chunking if framework supports

**Configuration:**
```python
{
    "chunk_size": 1800,
    "entity_types": ["PERSON", "ORG", "LOCATION", "CONCEPT"],
    "relationship_extraction": "llm",  # vs "rule-based"
    "graph_depth": 2,  # max hops for traversal
    "community_detection": True
}
```

**Acceptance Criteria:**
- Successfully builds knowledge graph from documents
- Retrieval leverages graph structure (not just node embeddings)
- Demonstrates advantage on multi-hop reasoning questions
- Neo4j setup documented with Docker Compose

**Evaluation Focus:**
- Multi-hop reasoning performance
- Entity-centric queries
- Relationship discovery
- Compare global vs local search modes

### 4. Filesystem RAG (Agentic Explorer) 🔲 TODO

**Concept:** LLM-guided file system navigation and retrieval, inspired by Claude Code's approach

**Technology Stack:**
- **Agent Framework:** LangGraph or AutoGen
- **Tools:** File operations (read, grep, find, list)
- **LLM:** OpenAI GPT-4o or GPT-4o-mini
- **Filesystem:** Local file system (no vector DB)

**Agentic Architecture:**
```python
Agent Tools:
1. list_directory(path) -> List[str]
2. grep_search(pattern, path) -> List[Match]
3. read_file(path, lines_range=None) -> str
4. find_files(pattern, path) -> List[str]
5. summarize_file(path) -> str  # Quick LLM summary
```

**Workflow:**
1. Agent receives question
2. Plans search strategy (which directories/files to explore)
3. Uses tools to navigate filesystem
4. Reads relevant files
5. Synthesizes answer from file contents
6. Returns answer + file paths as context

**Chunking Strategy:**
- No traditional chunking (works with whole files)
- May summarize large files before reading in detail
- Agent decides what to read based on file metadata

**Configuration:**
```python
{
    "max_tool_calls": 20,
    "max_file_reads": 10,
    "search_depth": 3,  # max directory depth
    "summarize_threshold": 5000,  # chars before summarizing
    "planning_strategy": "react"  # or "plan-and-execute"
}
```

**Acceptance Criteria:**
- Agent successfully navigates filesystem to find relevant information
- Demonstrates reasoning about file structure and content
- Outperforms vector search when file organization provides strong signals
- Transparent reasoning trail (shows which files were explored)

**Unique Value:**
- No indexing overhead (instant startup)
- Leverages filesystem organization as signal
- Interpretable retrieval process
- Useful when documents have meaningful file structure

**Evaluation Focus:**
- Efficiency: Does agent find relevant files quickly?
- Reasoning: Does agent make smart search decisions?
- Coverage: Does agent explore broadly enough?
- Compare to vector search on structured document sets

## Document Processing Pipeline

### Multi-Format Support 🔲 CRITICAL

**Supported Formats:**
1. **TXT** - Direct text extraction ✅ (already supported)
2. **PDF** - PyPDF2 or pdfplumber for text extraction 🔲 TODO
3. **DOCX** - python-docx for Word documents 🔲 TODO
4. **Markdown** - Direct text with optional structure preservation 🔲 TODO (nice-to-have)

**Document Loader Architecture:**
```python
class DocumentLoader:
    """Abstract document loader interface"""

    def load(self, file_path: str) -> Document:
        """Load document from file path"""

class TXTLoader(DocumentLoader): ...
class PDFLoader(DocumentLoader): ...
class DOCXLoader(DocumentLoader): ...

def create_loader(file_path: str) -> DocumentLoader:
    """Factory function based on file extension"""
```

**Handling Complex Documents:**
- **Best-effort per RAG:** Each RAG implementation handles formats as it can
- **Text extraction baseline:** All loaders extract to plain text as minimum
- **Structured data:** Graph RAG may extract tables to entities; others skip
- **Images/Charts:** Not in scope for initial implementation

**Implementation Priority:**
1. TXT (done)
2. PDF (most common, high priority)
3. DOCX (common in enterprise, high priority)
4. Markdown (nice-to-have, low priority)

### RAG-Specific Chunking Strategies

Each RAG type optimizes chunking for its retrieval mechanism:

| RAG Type | Chunk Size | Overlap | Strategy |
|----------|------------|---------|----------|
| Vector Semantic | 1000 chars | 200 chars | Fixed-size, sentence-aware |
| Hybrid Search | 700 chars | 100 chars | Smaller for keyword match |
| Graph RAG | 1800 chars | 250 chars | Large, entity-preserving |
| Filesystem RAG | N/A (whole files) | N/A | Agent decides what to read |

**Chunking Parameters (per RAG):**
```python
# src/rag_evaluator/rag_implementations/chunking_configs.py
CHUNKING_STRATEGIES = {
    "vector_semantic": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "splitter": "recursive_character"
    },
    "hybrid_search": {
        "chunk_size": 700,
        "chunk_overlap": 100,
        "splitter": "recursive_character"
    },
    "graph_rag": {
        "chunk_size": 1800,
        "chunk_overlap": 250,
        "splitter": "semantic"  # or entity-aware
    },
    "filesystem_rag": {
        "chunk_size": None,  # No chunking
        "strategy": "whole_file"
    }
}
```

**Future Enhancement (nice-to-have):**
- Support configuration testing: Compare same RAG with different chunk sizes
- Would require extending evaluation framework to handle config variations
- Not in scope for initial implementation

## Test Dataset Specification

### Data Sources (Hybrid Approach)

**Public Datasets (70%):**
- Wikipedia articles on diverse topics
- Technical documentation (Python, RAG concepts)
- Public domain books/articles
- Rationale: Fast, reproducible, broad coverage

**Custom Documents (30%):**
- Curated articles on specific domains
- Documents designed to test edge cases
- Multi-document reasoning scenarios
- Rationale: Unique insights, controlled complexity

### Document Corpus

**Size:** 30-50 documents
**Domains:** Mixed
- Technical (20%) - Software docs, API references
- Science (20%) - Research summaries, explanations
- Business (20%) - Reports, case studies
- General knowledge (20%) - History, geography, culture
- Edge cases (20%) - Ambiguous, contradictory, sparse info

**Document Characteristics:**
- Mix of lengths: 500-5000 words
- Mix of structure: Narrative, lists, tables, code snippets
- Mix of complexity: Simple facts to complex reasoning

### Test Case Creation (LLM-Generated with Review)

**Process:**
1. **LLM Generation:** Use GPT-4 to generate questions from documents
2. **Human Review:** Manually review and edit questions for quality
3. **Ground Truth:** Humans provide ground truth answers and context
4. **Difficulty Labeling:** Tag questions as easy/medium/hard

**Test Set Size:** 50-100 questions

**Question Type Distribution:**
- Factual lookups (30%) - "What is X?", "Who did Y?"
- Multi-hop reasoning (25%) - Requires connecting info from multiple chunks
- Summarization (20%) - "What are the main points of X?"
- Comparison (15%) - "How does X differ from Y?"
- Adversarial (10%) - Negation, ambiguity, trick questions

**Difficulty Stratification:**
- Easy (30%): Single-chunk retrieval, direct answers
- Medium (50%): Multi-chunk, some reasoning required
- Hard (20%): Multi-hop, inference, handling ambiguity

**Test Case Format (JSON):**
```json
{
  "test_cases": [
    {
      "id": "001",
      "question": "What is Retrieval Augmented Generation?",
      "expected_answer": "RAG is a technique that...",
      "ground_truth_context": ["chunk_1", "chunk_2"],
      "source_documents": ["rag_overview.pdf"],
      "difficulty": "easy",
      "question_type": "factual_lookup",
      "reasoning_hops": 1
    }
  ]
}
```

## Evaluation Framework

### Metrics (DeepEval - All Equal Weight)

All 4 DeepEval metrics weighted equally in final score:

1. **Faithfulness** (0-1): Answer derived only from retrieved context
2. **Answer Relevancy** (0-1): Answer addresses the question
3. **Contextual Precision** (0-1): Retrieved documents are relevant and ranked well
4. **Contextual Recall** (0-1): All relevant information was retrieved

**Overall Score:** Simple average of all 4 metrics

**Thresholds (configurable via .env):**
```bash
EVAL_FAITHFULNESS_THRESHOLD=0.7
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.7
EVAL_CONTEXTUAL_PRECISION_THRESHOLD=0.7
EVAL_CONTEXTUAL_RECALL_THRESHOLD=0.7
```

**No Custom Metrics:** Stick with DeepEval's built-in metrics for initial implementation

### Evaluation Execution

**Mode:** Sequential (not parallel)
- Avoid API rate limits
- More predictable costs
- Easier debugging

**Error Handling:** Skip and continue
- Log errors for each failed query
- Continue evaluation with remaining questions
- Report shows: total/successful/failed counts

**Retry Logic:**
- Exponential backoff for API errors
- Max retries: 3
- Configurable timeout per query: 30 seconds

**Token Optimization (CRITICAL):**
- **`include_reason=False`** in all DeepEval metrics to avoid verbose reasoning chains
- With `include_reason=True`: ~94,000 tokens per test case (10 questions = 940K tokens!)
- With `include_reason=False`: ~500-1,000 tokens per test case (10 questions = 5-10K tokens)
- **95%+ token reduction** while still getting accurate metric scores
- Enable `include_reason=True` only for debugging specific test failures

**Cost Tracking:** Phase-separated
```python
{
    "indexing_cost": {
        "embeddings": 0.XX,
        "llm_calls": 0.XX,  # For entity extraction in Graph RAG
        "total": 0.XX
    },
    "query_cost": {
        "retrieval": 0.XX,
        "generation": 0.XX,
        "total_per_query": 0.XX,
        "total_all_queries": 0.XX
    },
    "total_cost": 0.XX
}
```

### Report Generation

**Report Structure (all sections included):**

1. **Executive Summary**
   - Winner by overall score
   - Key findings (1-2 sentences per RAG)
   - Recommendation for use cases

2. **Metric Comparison Table**
   ```
   | RAG Type | Faithfulness | Relevancy | Precision | Recall | Overall |
   |----------|--------------|-----------|-----------|--------|---------|
   | Vector   | 0.82         | 0.79      | 0.75      | 0.71   | 0.77    |
   | Hybrid   | ...          | ...       | ...       | ...    | ...     |
   ```

3. **Performance Metrics**
   ```
   | RAG Type | Avg Latency | Indexing Cost | Per-Query Cost | Total Cost |
   |----------|-------------|---------------|----------------|------------|
   | Vector   | 1.2s        | $0.05         | $0.02          | $1.05      |
   ```

4. **Per-Query Breakdown**
   - Table showing each question
   - Scores from each RAG
   - Answer snippets
   - Context retrieved

5. **Failure Analysis**
   - Questions where all RAGs scored <0.5
   - Questions with high variance in scores
   - Common failure patterns
   - Examples of bad retrievals

6. **Statistical Analysis**
   - Mean scores with confidence intervals (95% CI)
   - Statistical significance tests (t-tests between RAG pairs)
   - Score distribution histograms
   - Correlation analysis (e.g., latency vs accuracy)

7. **Difficulty Breakdown**
   ```
   Performance by Question Difficulty:

   Easy Questions (n=30):
     Vector: 0.85, Hybrid: 0.87, Graph: 0.83, Filesystem: 0.81

   Medium Questions (n=50):
     Vector: 0.78, Hybrid: 0.80, Graph: 0.82, Filesystem: 0.75

   Hard Questions (n=20):
     Vector: 0.65, Hybrid: 0.68, Graph: 0.74, Filesystem: 0.61
   ```

**Output Formats:**
- JSON: `reports/evaluation_TIMESTAMP.json` (complete data)
- Markdown: `reports/evaluation_TIMESTAMP.md` (human-readable)
- Future: HTML with interactive charts (nice-to-have)

**Report Generator Enhancement:**
```python
# src/rag_evaluator/evaluation/report_generator.py
class ReportGenerator:
    def generate_executive_summary(self, results: dict) -> str: ...
    def generate_metric_comparison(self, results: dict) -> str: ...
    def generate_performance_metrics(self, results: dict) -> str: ...
    def generate_per_query_breakdown(self, results: dict) -> str: ...
    def generate_failure_analysis(self, results: dict) -> str: ...
    def generate_statistical_analysis(self, results: dict) -> str: ...
    def generate_difficulty_breakdown(self, results: dict) -> str: ...
```

## User Interface Specification

### Streamlit UI (Pre-computed Results Only)

**No Real-Time Querying:** UI displays pre-computed evaluation results only
- Rationale: Simpler, faster, no API costs during demos
- Users run evaluations via CLI, then view results in UI

**Three Tabs:**

#### Tab 1: Overview
- Summary statistics
- Winner announcement
- Key findings highlights
- Cost comparison chart
- Latency comparison chart

#### Tab 2: Detailed Comparison
- Side-by-side comparison table (sortable)
- Metric visualizations:
  - Bar charts for each metric
  - Grouped bar chart (all metrics, all RAGs)
- Performance scatter plot: Accuracy vs Latency vs Cost
- Difficulty breakdown charts

#### Tab 3: Query Explorer
- Dropdown to select test question
- Shows for selected question:
  - Question text
  - Difficulty and type
  - Ground truth answer
  - Table: Each RAG's answer, score, context chunks, latency
- Filter by difficulty, question type, RAG performance

**UI Implementation:**
```python
# src/rag_evaluator/ui/streamlit_app.py

def load_latest_report() -> dict:
    """Load most recent evaluation report JSON"""

def render_overview_tab(report: dict) -> None:
    """Summary statistics and charts"""

def render_comparison_tab(report: dict) -> None:
    """Detailed side-by-side comparison"""

def render_query_explorer_tab(report: dict) -> None:
    """Per-query drill-down"""
```

**Styling:**
- Clean, professional theme
- Color-coded scores (green >0.8, yellow 0.6-0.8, red <0.6)
- Responsive layout
- Exportable charts (PNG/SVG)

## Dependency Management

### uv Extras Strategy

**Optional dependencies per RAG type:**

```toml
[project.optional-dependencies]
# Base dependencies (always installed)
base = [
    "langchain>=0.1.0",
    "openai>=1.0.0",
    "deepeval>=0.20.0",
    "streamlit>=1.28.0",
]

# RAG-specific dependencies
vector-semantic = [
    "chromadb>=0.4.0",
    "langchain-community>=0.0.1",
]

hybrid-search = [
    "qdrant-client>=1.7.0",
    # Alternative: "weaviate-client>=3.25.0",
    "rank-bm25>=0.2.2",
]

graph-rag = [
    "neo4j>=5.14.0",
    "langchain-neo4j>=0.0.1",
    # If using Microsoft GraphRAG:
    "graphrag>=0.1.0",
    # If using Graphiti:
    # "graphiti-ai>=0.1.0",
]

filesystem-rag = [
    "langgraph>=0.0.1",
    # Alternative: "autogen>=0.2.0",
]

# Document format support
documents = [
    "pypdf2>=3.0.0",  # or "pdfplumber>=0.10.0"
    "python-docx>=1.0.0",
]

# Development dependencies
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]

# Install all RAG types
all-rags = [
    "rag_evaluator[vector-semantic]",
    "rag_evaluator[hybrid-search]",
    "rag_evaluator[graph-rag]",
    "rag_evaluator[filesystem-rag]",
]
```

**Installation Examples:**
```bash
# Install base + specific RAG
uv sync --extra vector-semantic

# Install base + multiple RAGs
uv sync --extra vector-semantic --extra hybrid-search

# Install everything
uv sync --all-extras

# Development setup
uv sync --all-extras --dev
```

**Rationale:**
- Users install only what they need
- Lighter initial install
- Clearer dependency separation
- Easier troubleshooting

## Implementation Roadmap

### Phase 1: Foundation Enhancement (Weeks 1-2) ✅ MOSTLY COMPLETE

**Status:** ChromaDB RAG + Evaluation framework complete

**Remaining Work:**
- [ ] Multi-format document loading (PDF, DOCX)
- [ ] Enhanced report generation (statistical analysis, difficulty breakdown)
- [ ] Update UI to display pre-computed results

**Deliverables:**
- Document loaders for PDF/DOCX
- Enhanced ReportGenerator with all sections
- Updated Streamlit UI with 3 tabs

### Phase 2: Hybrid Search RAG (Weeks 3-4)

**Goal:** Implement hybrid search RAG using easiest open-source option

**Tasks:**
1. Research Qdrant vs Weaviate setup complexity
2. Choose technology based on:
   - Ease of local installation (Docker Compose)
   - Documentation quality
   - Community support
   - No license costs
3. Implement HybridRAG class (inherits BaseRAG)
4. Configure RRF fusion
5. Optimize chunking strategy (smaller chunks)
6. Write unit + integration tests
7. Run evaluation and generate first 2-RAG comparison report
8. Document setup in README

**Acceptance Test:**
- Qdrant/Weaviate runs locally via Docker Compose
- HybridRAG successfully combines semantic + keyword search
- Evaluation shows improvement on keyword-heavy queries
- Setup documented, reproducible

**Deliverables:**
- `src/rag_evaluator/rag_implementations/vector_hybrid/`
- `docker-compose.yml` for Qdrant/Weaviate
- Tests in `tests/unit/` and `tests/integration/`
- First 2-RAG comparison report
- Updated README with hybrid search setup

### Phase 3: Graph RAG (Weeks 5-8)

**Goal:** Implement graph-based RAG using Microsoft GraphRAG or Graphiti

**Tasks:**
1. Evaluate Microsoft GraphRAG vs Graphiti:
   - Installation complexity
   - Documentation completeness
   - Community/support
   - Integration with existing stack
2. Set up Neo4j (Docker Compose)
3. Implement framework integration
4. Configure entity extraction and graph building
5. Implement global + local search modes (if using GraphRAG)
6. Write tests
7. Run evaluation focusing on multi-hop reasoning questions
8. Document graph construction process

**Acceptance Test:**
- Neo4j runs locally with sample graph visible
- GraphRAG builds knowledge graph from documents
- Retrieval leverages graph structure (not just embeddings)
- Shows advantage on multi-hop reasoning questions
- All components tested

**Deliverables:**
- `src/rag_evaluator/rag_implementations/graph_rag/`
- Updated `docker-compose.yml` with Neo4j
- Tests
- 3-RAG comparison report
- Graph visualization examples (screenshots/notebook)

### Phase 4: Filesystem RAG (Weeks 9-11)

**Goal:** Implement agentic filesystem explorer RAG

**Tasks:**
1. Choose agent framework (LangGraph vs AutoGen)
2. Implement agent tools (list, read, grep, find)
3. Implement ReAct or Plan-and-Execute agent loop
4. Add reasoning trace logging
5. Optimize for efficiency (limit tool calls)
6. Write tests (mock filesystem for unit tests)
7. Create structured document corpus to showcase advantage
8. Run evaluation
9. Document agent reasoning process

**Acceptance Test:**
- Agent successfully navigates filesystem to answer questions
- Reasoning trace is transparent and logical
- Outperforms vector search on structured document sets
- Demonstrates unique value (no indexing, interpretable)
- Tests cover agent decision-making

**Deliverables:**
- `src/rag_evaluator/rag_implementations/filesystem_rag/`
- Tests with mocked filesystem
- 4-RAG comparison report (FINAL)
- Agent reasoning trace examples

### Phase 5: Portfolio Presentation (Weeks 12-13)

**Goal:** Professional presentation of results

**Tasks:**
1. Write comprehensive GitHub README showcase
   - Badges (tests, coverage, license)
   - Architecture diagram
   - Key findings summary
   - Quick start guide
   - Results visualization (charts, tables)
2. Write blog post-style analysis
   - Narrative explaining each RAG approach
   - Methodology explanation
   - Results analysis with insights
   - When to use each RAG type
   - Lessons learned
3. Create Jupyter notebook walkthrough (optional)
4. Add architecture diagrams
5. Record demo video (optional)
6. Final documentation polish
7. GitHub release with DOI (for citations)

**Deliverables:**
- Enhanced README with results
- `RESULTS.md` or blog post markdown
- Architecture diagrams (draw.io or similar)
- Optional: Jupyter notebook, demo video
- GitHub release v1.0.0

## Technical Constraints & Requirements

### Performance Requirements

**Minimum Acceptable Performance (per RAG):**
- Faithfulness: >0.70
- Answer Relevancy: >0.70
- Contextual Precision: >0.60
- Contextual Recall: >0.60
- Query Latency: <10 seconds (95th percentile)
- Cost per query: <$0.10

**If RAG doesn't meet minimums:**
- Document why (too experimental, wrong use case, etc.)
- Consider excluding from final comparison
- Or include with caveats about limitations

### Infrastructure Requirements

**Local Development:**
- Python 3.12+
- uv package manager
- Docker + Docker Compose (for Neo4j, Qdrant/Weaviate)
- OpenAI API key
- 16GB RAM recommended (for local DBs)

**Cloud-Ready Architecture:**
- Environment-based configuration (12-factor app)
- Stateless application design
- External storage for vector DBs (Cloud SQL, managed services)
- API key management via secrets
- No hardcoded paths or credentials

**Not Implementing (but designed for):**
- Actual cloud deployment
- Managed database provisioning
- Scaling/load balancing
- Production monitoring/alerting

### Code Quality Requirements

**All code must:**
- Pass ruff linting (100% compliance)
- Pass mypy type checking (strict mode)
- Have type hints on all functions
- Have docstrings on public methods
- Have unit tests (target 80%+ coverage)
- Have integration tests for critical paths
- Follow existing patterns (BaseRAG interface)

**CI/CD:**
- GitHub Actions runs on all PRs
- Tests, linting, type checking must pass
- No merge without passing CI

## Risks & Mitigations

### Risk 1: Technical Complexity

**Risk:** Graph RAG or Filesystem RAG prove too complex to implement properly
**Impact:** High - Core deliverable at risk
**Probability:** Medium

**Mitigation:**
- Use existing frameworks (don't build from scratch)
- Accept "minimal viable implementation" approach
- Focus on integration and evaluation, not novel algorithms
- Allow framework choice flexibility (GraphRAG vs Graphiti)
- If truly blocked: Document attempt, use 3-RAG comparison

### Risk 2: Evaluation Validity

**Risk:** DeepEval metrics don't capture meaningful differences
**Impact:** Medium - Results may not be insightful
**Probability:** Low

**Mitigation:**
- Diverse test question types
- Manual review of sample answers
- Statistical analysis shows significance
- Failure analysis surfaces patterns
- Compare against human judgments for validation

### Risk 3: API Costs

**Risk:** Evaluation costs exceed budget
**Impact:** Low - Slows iteration
**Probability:** Medium

**Mitigation:**
- Phase-separated cost tracking
- Use GPT-4o-mini where possible
- Cache embeddings to avoid recomputation
- Small test set for development (10 questions)
- Full evaluation only when ready

### Risk 4: Dependency Conflicts

**Risk:** RAG-specific dependencies conflict with each other
**Impact:** Medium - Blocks multi-RAG testing
**Probability:** Low

**Mitigation:**
- uv extras isolate dependencies
- Pin major versions in pyproject.toml
- Test `uv sync --all-extras` regularly
- Document known conflicts
- Worst case: Separate virtual environments per RAG

### Risk 5: Time Investment

**Risk:** Implementation takes longer than 2-3 months
**Impact:** Low - This is learning project
**Probability:** Medium

**Mitigation:**
- Incremental releases (one RAG at a time)
- No hard deadlines
- Can release 2-RAG or 3-RAG comparison as intermediate milestone
- Filesystem RAG is lowest priority (can defer)
- Portfolio value exists even with partial completion

## Success Metrics

### Technical Metrics
- [ ] 4 RAG implementations complete and tested
- [ ] Test coverage >80% overall
- [ ] All CI/CD checks passing
- [ ] Evaluation framework statistically sound
- [ ] Reports generated successfully

### Portfolio Metrics
- [ ] GitHub README is professional and compelling
- [ ] Results document provides actionable insights
- [ ] Code demonstrates best practices
- [ ] Documentation enables others to reproduce
- [ ] Project generates community interest (GitHub stars, discussions)

### Learning Metrics
- [ ] Deep understanding of RAG tradeoffs
- [ ] Experience with multiple vector DBs
- [ ] Knowledge graph implementation experience
- [ ] Agent framework proficiency
- [ ] Evaluation methodology expertise

## Open Questions

1. **Hybrid Search Technology:** Qdrant vs Weaviate vs EnsembleRetriever?
   - Decision needed in Phase 2
   - Criteria: Ease of setup, documentation, community support

2. **Graph RAG Framework:** Microsoft GraphRAG vs Graphiti?
   - Decision needed in Phase 3
   - Criteria: Installation complexity, integration ease, documentation

3. **Agent Framework:** LangGraph vs AutoGen for Filesystem RAG?
   - Decision needed in Phase 4
   - Criteria: Ease of use, tool integration, debugging capability

4. **Test Set Size:** 50 vs 100 questions?
   - Depends on API cost budget
   - More questions = better statistics but higher cost
   - Can start with 50, expand if budget allows

5. **PDF Library:** PyPDF2 vs pdfplumber vs pymupdf?
   - Test all three on sample PDFs
   - Choose based on extraction quality and ease of use

## Appendix

### BaseRAG Interface (Current)

```python
# src/rag_evaluator/common/base_rag.py
from abc import ABC, abstractmethod
from typing import Any

class BaseRAG(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def prepare_documents(self, documents_path: str) -> None:
        """Index/prepare documents for retrieval"""
        pass

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """
        Execute query and return results

        Returns:
            {
                "answer": str,
                "context": List[str],  # Retrieved chunks
                "metadata": {
                    "retrieval_time": float,
                    "chunks_retrieved": int,
                    # RAG-specific metadata
                }
            }
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Return performance metrics

        Returns:
            {
                "total_queries": int,
                "avg_retrieval_time": float,
                "total_chunks_indexed": int,
                # RAG-specific metrics
            }
        """
        pass
```

### CLI Commands (Current + Planned)

```bash
# Document preparation
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
uv run rag-eval prepare --rag-type all --input-dir data/raw

# Run evaluation
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set.json
uv run rag-eval evaluate --rag-type all --output reports/my_eval/

# Launch UI
uv run rag-eval ui

# Show metrics (planned)
uv run rag-eval metrics --rag-type vector_semantic

# Generate test set (planned)
uv run rag-eval generate-tests --input-dir data/raw --count 50 --output data/test_set.json
```

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-...

# Models
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# DeepEval Configuration
EVAL_FAITHFULNESS_THRESHOLD=0.7
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.7
EVAL_CONTEXTUAL_PRECISION_THRESHOLD=0.7
EVAL_CONTEXTUAL_RECALL_THRESHOLD=0.7
DEEPEVAL_ASYNC_MODE=False  # Sequential evaluation to avoid rate limits
DEEPEVAL_PER_TASK_TIMEOUT=900
DEEPEVAL_PER_ATTEMPT_TIMEOUT=300
DEEPEVAL_MAX_RETRIES=3

# Vector DB Configuration
CHROMA_PERSIST_DIR=data/chroma_db
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Graph DB Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Cost Tracking
TRACK_COSTS=true

# Logging
LOG_LEVEL=INFO
```

---

**Document Status:** Draft v1.0
**Next Review:** After Phase 2 completion (Hybrid Search RAG)
**Approval:** Pending user review and discussion
