# Architecture Overview

> **A comprehensive guide to the RAG Evaluator Platform's internal architecture, component relationships, and data flows.**

---

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Architecture](#high-level-architecture)
- [Component Deep Dive](#component-deep-dive)
  - [Core Engine](#core-engine)
  - [Backend API](#backend-api)
  - [Frontend Application](#frontend-application)
  - [Database Layer](#database-layer)
  - [Vector Stores](#vector-stores)
- [Data Flow Diagrams](#data-flow-diagrams)
- [RAG Pipeline Architecture](#rag-pipeline-architecture)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Database Schema](#database-schema)
- [Security Architecture](#security-architecture)
- [Deployment Topologies](#deployment-topologies)

---

## System Overview

The RAG Evaluator Platform is designed as a **modular, multi-tier architecture** that separates concerns between data ingestion, retrieval strategies, evaluation, and presentation. This design enables:

- **Flexibility**: Swap RAG implementations without changing the evaluation pipeline
- **Scalability**: Each component can be scaled independently
- **Extensibility**: Add new RAG strategies or metrics with minimal changes
- **Testability**: Isolated components enable comprehensive unit and integration testing

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Core engine, API, and UI are distinct layers |
| **Plugin Architecture** | RAG implementations inherit from `BaseRAG` |
| **Configuration-Driven** | Behavior controlled via environment variables |
| **Async-First** | Backend uses async/await for I/O operations |
| **Type Safety** | Full type annotations with mypy enforcement |

---

## High-Level Architecture

The platform consists of three main tiers that share a common core engine:

![High-Level Architecture](images/architecture-overview.png)
<!-- PLACEHOLDER: architecture-overview.png - A diagram showing the three tiers (CLI, Web Platform, Core Engine) and their relationships -->

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                 │
├─────────────────────────────────┬───────────────────────────────────────────┤
│          CLI Tool               │            Web Platform                    │
│  ┌─────────────────────────┐    │    ┌─────────────────────────────────┐    │
│  │  $ rag-eval prepare     │    │    │         React Frontend          │    │
│  │  $ rag-eval evaluate    │    │    │  ┌───────────────────────────┐  │    │
│  │  $ rag-eval ui          │    │    │  │  Dashboard │ Projects     │  │    │
│  └───────────┬─────────────┘    │    │  │  KB Mgmt   │ Evaluations  │  │    │
│              │                  │    │  └───────────────────────────┘  │    │
│              │                  │    └──────────────┬──────────────────┘    │
│              │                  │                   │                       │
├──────────────┼──────────────────┼───────────────────┼───────────────────────┤
│              │                  │      FastAPI Backend (REST API)           │
│              │                  │    ┌─────────────────────────────────┐    │
│              │                  │    │  /api/v1/projects               │    │
│              │                  │    │  /api/v1/evaluations            │    │
│              │                  │    │  /api/v1/knowledge-bases        │    │
│              └──────────────────┼────┴──────────────┬──────────────────┘    │
│                                 │                   │                       │
├─────────────────────────────────┴───────────────────┴───────────────────────┤
│                              CORE ENGINE                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         BaseRAG Interface                              │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │   Vector    │ │   Hybrid    │ │    Graph    │ │   Filesystem    │  │  │
│  │  │  Semantic   │ │   Search    │ │     RAG     │ │      RAG        │  │  │
│  │  │  (Chroma)   │ │  (Qdrant)   │ │   (Neo4j)   │ │   (Agentic)     │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DeepEval Integration                              │  │
│  │    Faithfulness │ Answer Relevancy │ Precision │ Recall │ G-Eval     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           STORAGE LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │   ChromaDB   │  │    Qdrant    │  │    Neo4j     │     │
│  │  (Metadata)  │  │  (Vectors)   │  │  (Hybrid)    │  │   (Graph)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### Core Engine

The Core Engine is the heart of the platform, containing all RAG implementations and evaluation logic. It's designed to be **interface-independent** - the same code powers both the CLI and the web platform.

![Core Engine Architecture](images/core-engine-architecture.png)
<!-- PLACEHOLDER: core-engine-architecture.png - Detailed diagram of core engine components -->

#### Directory Structure

```
src/rag_evaluator/
├── common/                      # Shared abstractions
│   ├── base_rag.py             # BaseRAG abstract class
│   ├── provider_interfaces.py  # Data transfer objects
│   ├── token_tracker.py        # Thread-safe token counting
│   └── document_loaders.py     # PDF, DOCX, TXT loaders
│
├── rag_implementations/         # RAG strategies
│   ├── vector_semantic/        # ChromaDB implementation
│   ├── vector_hybrid/          # Qdrant + SPLADE
│   ├── graph_rag/              # Neo4j knowledge graph
│   └── filesystem_rag/         # Agentic file navigation
│
├── evaluation/                  # Evaluation engine
│   ├── evaluator.py            # DeepEval integration
│   ├── report_generator.py     # Report creation
│   └── difficulty_analysis.py  # Question difficulty scoring
│
└── cli.py                       # CLI entry point
```

#### BaseRAG Interface

All RAG implementations inherit from `BaseRAG`, ensuring consistent behavior:

```python
class BaseRAG(ABC):
    """Abstract base class for all RAG implementations."""

    @abstractmethod
    def prepare_documents(self, documents_path: str) -> None:
        """Index documents for retrieval."""

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Execute RAG pipeline: retrieve + generate."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return implementation metrics."""

    # Optional overrides
    def retrieve(self, question: str, top_k: int) -> RetrievedContext: ...
    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer: ...
```

#### Data Classes

The `provider_interfaces.py` module defines the data structures that flow through the system:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RetrievedContext                            │
├─────────────────────────────────────────────────────────────────┤
│  chunks: list[str]              # Raw text chunks               │
│  chunk_details: list[RetrievedChunk]  # Detailed chunk info    │
│  trace: RetrievalTrace          # Debug/observability info      │
│  retrieval_time: float          # Latency in seconds            │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GeneratedAnswer                             │
├─────────────────────────────────────────────────────────────────┤
│  text: str                      # The generated answer          │
│  generation_time: float         # LLM latency                   │
│  prompt_tokens: int             # Input tokens used             │
│  completion_tokens: int         # Output tokens generated       │
└─────────────────────────────────────────────────────────────────┘
```

---

### Backend API

The Backend API is a **FastAPI** application providing RESTful endpoints for the web platform.

![Backend Architecture](images/backend-architecture.png)
<!-- PLACEHOLDER: backend-architecture.png - Backend service layer diagram -->

#### Directory Structure

```
platform/backend/
├── app/
│   ├── main.py                  # FastAPI app initialization
│   ├── config.py                # Configuration management
│   ├── database.py              # Database connection
│   │
│   ├── api/                     # API route handlers
│   │   ├── projects.py          # Project CRUD
│   │   ├── knowledge_bases.py   # KB management
│   │   ├── test_sets.py         # Test case management
│   │   ├── rag_configs.py       # RAG configuration
│   │   ├── evaluations.py       # Evaluation execution
│   │   ├── comparisons.py       # Result comparison
│   │   ├── trends.py            # Performance trends
│   │   └── webhooks.py          # External integrations
│   │
│   ├── models/                  # SQLModel ORM definitions
│   │   ├── project.py
│   │   ├── knowledge_base.py
│   │   ├── test_set.py
│   │   ├── rag_config.py
│   │   ├── evaluation.py
│   │   └── webhook.py
│   │
│   └── services/                # Business logic layer
│       ├── rag_adapter.py       # RAG instantiation factory
│       ├── rag_registry.py      # RAG type definitions
│       ├── index_build_service.py  # Index management
│       └── job_event_log.py     # Async job tracking
│
├── tests/                       # Test suite
└── alembic/                     # Database migrations
```

#### Request Flow

```
┌──────────┐    HTTP     ┌──────────────┐    ┌────────────────┐    ┌──────────┐
│  Client  │ ─────────▶  │   FastAPI    │ ──▶│    Service     │ ──▶│    DB    │
│          │             │   Router     │    │    Layer       │    │          │
└──────────┘             └──────────────┘    └────────────────┘    └──────────┘
                               │                    │
                               │                    ▼
                               │             ┌────────────────┐
                               │             │   RAG Adapter  │
                               │             │  (Core Engine) │
                               │             └────────────────┘
                               │                    │
                               ▼                    ▼
                         ┌────────────────────────────────────┐
                         │        Vector Stores / Neo4j       │
                         └────────────────────────────────────┘
```

#### Key Services

| Service | Responsibility |
|---------|----------------|
| `rag_adapter.py` | Dynamically imports and instantiates RAG classes based on config |
| `rag_registry.py` | Defines available RAG types and their parameter schemas |
| `index_build_service.py` | Manages knowledge base indexing with progress tracking |
| `job_event_log.py` | Provides event logging for async jobs (evaluations, indexing) |

---

### Frontend Application

The Frontend is a **React** single-page application built with **Vite** and **Tailwind CSS**.

![Frontend Architecture](images/frontend-architecture.png)
<!-- PLACEHOLDER: frontend-architecture.png - React component hierarchy diagram -->

#### Directory Structure

```
platform/frontend/
├── src/
│   ├── main.tsx                 # Application entry point
│   ├── App.tsx                  # Root component with routing
│   │
│   ├── api/                     # API client layer
│   │   └── client.ts            # Axios-based API wrapper
│   │
│   ├── pages/                   # Route-level components
│   │   ├── Dashboard.tsx        # Home overview
│   │   ├── Projects.tsx         # Project list
│   │   ├── ProjectDetail.tsx    # Single project view
│   │   ├── KBDetail.tsx         # Knowledge base detail
│   │   └── IndexDetail.tsx      # Index configuration
│   │
│   ├── components/              # Reusable UI components
│   │   ├── evaluations/         # Evaluation-specific
│   │   │   ├── StartEvaluationWizard.tsx
│   │   │   ├── EvaluationProgress.tsx
│   │   │   ├── EvaluationResults.tsx
│   │   │   ├── MetricExplainability.tsx
│   │   │   └── RetrievalTraceViewer.tsx
│   │   │
│   │   ├── knowledge-bases/     # KB management
│   │   ├── test-sets/           # Test set management
│   │   ├── rag-configs/         # RAG configuration
│   │   ├── comparisons/         # Result comparisons
│   │   ├── trends/              # Analytics charts
│   │   └── ui/                  # Base UI components
│   │
│   ├── hooks/                   # Custom React hooks
│   └── lib/                     # Utility functions
│
├── public/                      # Static assets
└── index.html                   # HTML template
```

#### Component Hierarchy

```
App
├── Layout (Header, Navigation, Footer)
│
├── Dashboard
│   ├── ProjectCard[]
│   └── QuickActions
│
├── ProjectDetail
│   ├── ProjectHeader
│   ├── KnowledgeBaseList
│   │   └── KBCard[]
│   ├── TestSetList
│   │   └── TestSetCard[]
│   ├── RAGConfigList
│   │   └── RAGConfigCard[]
│   └── EvaluationList
│       └── EvaluationCard[]
│
├── EvaluationDetail
│   ├── EvaluationProgress (SSE)
│   ├── EvaluationResults
│   │   ├── MetricCards[]
│   │   ├── ResultsTable
│   │   └── MetricExplainability
│   ├── RetrievalTraceViewer
│   └── BaselineComparison
│
└── TrendsView
    ├── MetricTrendChart
    └── ComparisonTable
```

#### State Management

The frontend uses **React Query** for server state and local component state for UI interactions:

```
┌──────────────────────────────────────────────────────────────┐
│                     State Architecture                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐      ┌────────────────┐                 │
│  │  React Query   │◀────▶│   Backend API  │                 │
│  │  (Server State)│      └────────────────┘                 │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐      ┌────────────────┐                 │
│  │   Components   │◀────▶│  Local State   │                 │
│  │                │      │ (UI Interactions)│                │
│  └────────────────┘      └────────────────┘                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### Database Layer

The platform supports both **SQLite** (development) and **PostgreSQL** (production).

![Database Schema](images/database-schema.png)
<!-- PLACEHOLDER: database-schema.png - Entity-relationship diagram -->

#### Schema Overview

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│     Project     │───┬──▶│  KnowledgeBase   │──────▶│    Document     │
├─────────────────┤   │   ├──────────────────┤       ├─────────────────┤
│ id              │   │   │ id               │       │ id              │
│ name            │   │   │ project_id (FK)  │       │ kb_id (FK)      │
│ description     │   │   │ name             │       │ filename        │
│ status          │   │   │ description      │       │ content_hash    │
│ tags            │   │   │ status           │       │ size_bytes      │
│ created_at      │   │   └──────────────────┘       └─────────────────┘
│ updated_at      │   │           │
└─────────────────┘   │           ▼
         │            │   ┌──────────────────┐
         │            │   │ KBIndex          │
         │            │   ├──────────────────┤
         │            │   │ id               │
         │            │   │ kb_id (FK)       │
         │            │   │ rag_type         │
         │            │   │ status           │
         │            │   │ storage_path     │
         │            │   └──────────────────┘
         │            │
         ├───────────▶│   ┌──────────────────┐       ┌─────────────────┐
         │            └──▶│    TestSet       │──────▶│    TestCase     │
         │                ├──────────────────┤       ├─────────────────┤
         │                │ id               │       │ id              │
         │                │ project_id (FK)  │       │ test_set_id(FK) │
         │                │ name             │       │ question        │
         │                │ description      │       │ expected_answer │
         │                └──────────────────┘       │ difficulty      │
         │                                           │ tags            │
         │                                           └─────────────────┘
         │
         ├───────────────▶┌──────────────────┐
         │                │    RAGConfig     │
         │                ├──────────────────┤
         │                │ id               │
         │                │ project_id (FK)  │
         │                │ name             │
         │                │ rag_type         │
         │                │ llm_provider     │
         │                │ llm_model        │
         │                │ parameters (JSON)│
         │                └──────────────────┘
         │                        │
         │                        ▼
         │                ┌──────────────────┐       ┌─────────────────┐
         └───────────────▶│   Evaluation     │──────▶│EvaluationResult │
                          ├──────────────────┤       ├─────────────────┤
                          │ id               │       │ id              │
                          │ project_id (FK)  │       │ evaluation_id   │
                          │ kb_id (FK)       │       │ test_case_id    │
                          │ test_set_id (FK) │       │ question        │
                          │ rag_config_id    │       │ answer          │
                          │ status           │       │ context         │
                          │ metrics_summary  │       │ scores (JSON)   │
                          │ is_baseline      │       │ token_usage     │
                          └──────────────────┘       └─────────────────┘
```

---

### Vector Stores

The platform integrates with multiple vector databases, each optimized for different use cases:

![Vector Store Integration](images/vector-stores.png)
<!-- PLACEHOLDER: vector-stores.png - Vector store comparison diagram -->

| Store | Use Case | Index Type | Query Type |
|-------|----------|------------|------------|
| **ChromaDB** | Semantic search | Dense vectors (1536d) | KNN similarity |
| **Qdrant** | Hybrid search | Dense + Sparse (SPLADE) | RRF fusion |
| **Neo4j** | Knowledge graph | Vector index + Graph | Cypher + similarity |

#### Storage Isolation

Each knowledge base index is isolated by a unique storage path:

```
storage/
├── indexes/
│   ├── {index_id_1}/
│   │   └── chroma/           # ChromaDB persistence
│   │       └── chroma.sqlite3
│   │
│   ├── {index_id_2}/
│   │   └── qdrant/           # Qdrant data (if local)
│   │
│   └── {index_id_3}/
│       └── filesystem_rag/   # Prepared filesystem
│           ├── _meta/
│           ├── _index/
│           ├── _summaries/
│           └── documents/
│
└── uploads/
    └── {kb_id}/
        └── documents/        # Original uploaded files
```

---

## Data Flow Diagrams

### Document Ingestion Flow

![Document Ingestion Flow](images/document-ingestion-flow.png)
<!-- PLACEHOLDER: document-ingestion-flow.png - Sequence diagram for document upload and indexing -->

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  User   │    │Frontend │    │   Backend   │    │ RAG Adapter  │    │Vector Store│
└────┬────┘    └────┬────┘    └──────┬──────┘    └──────┬───────┘    └─────┬──────┘
     │              │                │                  │                   │
     │ Upload PDF   │                │                  │                   │
     │─────────────▶│                │                  │                   │
     │              │ POST /documents│                  │                   │
     │              │───────────────▶│                  │                   │
     │              │                │ Store file       │                   │
     │              │                │─────────────────▶│                   │
     │              │                │                  │                   │
     │              │ POST /index    │                  │                   │
     │              │───────────────▶│                  │                   │
     │              │                │ prepare_documents│                   │
     │              │                │─────────────────▶│                   │
     │              │                │                  │ Load documents    │
     │              │                │                  │──────────────────▶│
     │              │                │                  │ Chunk & embed     │
     │              │                │                  │──────────────────▶│
     │              │                │                  │ Store vectors     │
     │              │                │                  │──────────────────▶│
     │              │                │                  │◀──────────────────│
     │              │                │◀─────────────────│                   │
     │              │◀───────────────│ Index complete   │                   │
     │◀─────────────│                │                  │                   │
     │              │                │                  │                   │
```

### Evaluation Execution Flow

![Evaluation Flow](images/evaluation-flow.png)
<!-- PLACEHOLDER: evaluation-flow.png - Sequence diagram for evaluation execution -->

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  User   │    │Frontend │    │   Backend   │    │   RAG    │    │ DeepEval │
└────┬────┘    └────┬────┘    └──────┬──────┘    └────┬─────┘    └────┬─────┘
     │              │                │                │               │
     │ Start Eval   │                │                │               │
     │─────────────▶│                │                │               │
     │              │ POST /evaluate │                │               │
     │              │───────────────▶│                │               │
     │              │                │                │               │
     │              │ SSE /stream    │                │               │
     │              │◀──────────────▶│                │               │
     │              │                │                │               │
     │              │                │ For each test case:            │
     │              │                │────────────────┼───────────────│
     │              │                │                │               │
     │              │                │ query()        │               │
     │              │                │───────────────▶│               │
     │              │                │◀───────────────│ answer+context│
     │              │                │                │               │
     │              │                │ evaluate()     │               │
     │              │                │───────────────────────────────▶│
     │              │                │◀───────────────────────────────│
     │              │                │                │   scores      │
     │              │                │────────────────┼───────────────│
     │              │                │                │               │
     │              │◀─ progress ────│                │               │
     │◀─────────────│                │                │               │
     │              │                │                │               │
     │              │◀─ complete ────│                │               │
     │◀─────────────│                │                │               │
```

### Query Processing Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Query Processing                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐                                                         │
│  │   Question  │                                                         │
│  │  "What is   │                                                         │
│  │   RAG?"     │                                                         │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      RETRIEVAL PHASE                             │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │    │
│  │  │   Embedding  │   │    Vector    │   │    Rank &    │        │    │
│  │  │   Question   │──▶│    Search    │──▶│    Filter    │        │    │
│  │  │              │   │              │   │              │        │    │
│  │  └──────────────┘   └──────────────┘   └──────────────┘        │    │
│  │                                              │                  │    │
│  └──────────────────────────────────────────────┼──────────────────┘    │
│                                                 │                       │
│                                                 ▼                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     GENERATION PHASE                             │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │    │
│  │  │   Construct  │   │     LLM      │   │    Format    │        │    │
│  │  │    Prompt    │──▶│   Generate   │──▶│    Answer    │        │    │
│  │  │              │   │              │   │              │        │    │
│  │  └──────────────┘   └──────────────┘   └──────────────┘        │    │
│  │                                              │                  │    │
│  └──────────────────────────────────────────────┼──────────────────┘    │
│                                                 │                       │
│                                                 ▼                       │
│  ┌─────────────┐                                                        │
│  │   Answer    │                                                        │
│  │  + Context  │                                                        │
│  │  + Metadata │                                                        │
│  └─────────────┘                                                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline Architecture

Each RAG implementation follows the same high-level pipeline but with different retrieval strategies:

### Vector Semantic (ChromaDB)

```
Documents ──▶ Chunk (1000 chars) ──▶ Embed (OpenAI) ──▶ Store (Chroma)
                                                              │
Question ──▶ Embed ──▶ KNN Search ──▶ Top-K Chunks ──▶ LLM ──┼──▶ Answer
                                                              │
                                      Retrieved Context ◀─────┘
```

### Hybrid Search (Qdrant)

```
Documents ──┬──▶ Dense Embed (OpenAI) ──┬──▶ Store (Qdrant)
            │                           │
            └──▶ Sparse Embed (SPLADE) ─┘
                                              │
Question ──┬──▶ Dense Search ──┬──▶ RRF Fusion ──▶ Top-K ──▶ LLM ──▶ Answer
           │                   │
           └──▶ Sparse Search ─┘
```

### Graph RAG (Neo4j)

```
Documents ──▶ LLM Extract ──▶ Entities + Relations ──▶ Store (Neo4j Graph)
                                        │
                                        ├──▶ Vector Index (embeddings)
                                        │
Question ──▶ Vector Search ──▶ Entry Nodes ──▶ Graph Traverse ──▶ Context
                                                                    │
                                                   LLM ──▶ Answer ◀─┘
```

### Filesystem RAG (Agentic)

```
Documents ──▶ Prepare ──▶ _meta/ + _index/ + _summaries/ + documents/
                                              │
Question ──▶ ReAct Agent ──▶ Plan ──▶ Navigate ──▶ Read ──▶ Synthesize
                  │                              ▲
                  │    Tools: ls, read, grep     │
                  └──────────────────────────────┘
```

---

## Evaluation Pipeline

The evaluation pipeline measures RAG quality using DeepEval's LLM-as-judge approach:

![Evaluation Pipeline](images/evaluation-pipeline.png)
<!-- PLACEHOLDER: evaluation-pipeline.png - Evaluation pipeline diagram -->

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────┐     ┌─────────────┐     ┌─────────────────────────────────┐│
│  │  Test Set  │────▶│  Test Case  │────▶│         RAG Query               ││
│  │            │     │  question   │     │  ┌─────────┐   ┌─────────────┐  ││
│  │ - question │     │  expected   │     │  │Retrieved│   │  Generated  │  ││
│  │ - expected │     │             │     │  │ Context │   │   Answer    │  ││
│  │            │     └─────────────┘     │  └─────────┘   └─────────────┘  ││
│  └────────────┘                         └─────────────────────────────────┘│
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        DeepEval Metrics                               │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │ Faithfulness │  │   Answer     │  │  Contextual  │               │  │
│  │  │              │  │  Relevancy   │  │  Precision   │               │  │
│  │  │ answer vs    │  │ answer vs    │  │ ranking      │               │  │
│  │  │ context      │  │ question     │  │ quality      │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐                                  │  │
│  │  │  Contextual  │  │ Correctness  │                                  │  │
│  │  │   Recall     │  │  (G-Eval)    │                                  │  │
│  │  │ completeness │  │ semantic     │                                  │  │
│  │  │              │  │ equivalence  │                                  │  │
│  │  └──────────────┘  └──────────────┘                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Results & Reports                              │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │   Summary    │  │  Per-Case    │  │  Difficulty  │               │  │
│  │  │   Metrics    │  │   Results    │  │   Analysis   │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metric Calculation

Each metric is calculated by an LLM judge:

| Metric | Input | Judgment |
|--------|-------|----------|
| **Faithfulness** | Answer + Context | "Can every claim be verified from context?" |
| **Answer Relevancy** | Answer + Question | "Does the answer address the question?" |
| **Contextual Precision** | Context + Ground Truth | "Are relevant chunks ranked higher?" |
| **Contextual Recall** | Context + Ground Truth | "Does context contain all needed info?" |
| **Correctness** | Answer + Expected | "Are they semantically equivalent?" |

---

## Security Architecture

### Authentication & Authorization

The open-source edition runs without authentication for local development. For production:

```
┌───────────────────────────────────────────────────────────────┐
│                    Production Deployment                       │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │   Client    │───▶│   Reverse    │───▶│    Backend      │  │
│  │             │    │    Proxy     │    │                 │  │
│  │             │    │  (nginx/ALB) │    │  - No internal  │  │
│  │             │    │              │    │    auth         │  │
│  │             │    │  - TLS       │    │  - Trust proxy  │  │
│  │             │    │  - Auth      │    │                 │  │
│  └─────────────┘    └──────────────┘    └─────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Data Security

| Layer | Protection |
|-------|------------|
| **Transport** | TLS 1.3 (via reverse proxy) |
| **Storage** | Filesystem permissions, optional encryption at rest |
| **API Keys** | Environment variables, never logged |
| **Uploads** | File type validation, size limits |

---

## Deployment Topologies

### Development (Local)

```
┌─────────────────────────────────────────────┐
│              Developer Machine              │
│                                             │
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Frontend   │  │      Backend        │  │
│  │  (Vite)     │  │    (Uvicorn)        │  │
│  │  :5173      │  │      :8000          │  │
│  └─────────────┘  └─────────────────────┘  │
│                          │                  │
│  ┌───────────────────────┴──────────────┐  │
│  │           Docker Compose             │  │
│  │  ┌─────────┐ ┌───────┐ ┌──────────┐ │  │
│  │  │Postgres │ │Qdrant │ │  Neo4j   │ │  │
│  │  │ :5432   │ │:6333  │ │  :7687   │ │  │
│  │  └─────────┘ └───────┘ └──────────┘ │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Production (Docker Compose)

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Server                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Docker Compose                      │   │
│  │                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │  nginx   │  │ Frontend │  │     Backend      │  │   │
│  │  │  :80/443 │──▶│  :3000   │  │      :8000       │  │   │
│  │  │          │──────────────────▶                 │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  │                                      │              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │ Postgres │  │  Qdrant  │  │      Neo4j       │  │   │
│  │  │  :5432   │  │  :6333   │  │      :7687       │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  │                                                      │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │              Persistent Volumes              │  │   │
│  │  │   postgres_data  qdrant_data  neo4j_data    │  │   │
│  │  │                  storage                     │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Cloud-Native (Kubernetes)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Kubernetes Cluster                               │
│                                                                          │
│  ┌───────────────┐                                                       │
│  │    Ingress    │                                                       │
│  │  (nginx/ALB)  │                                                       │
│  └───────┬───────┘                                                       │
│          │                                                               │
│  ┌───────┴────────────────────────────────────────────────────────────┐ │
│  │                         Services                                    │ │
│  │                                                                     │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐│ │
│  │  │    Frontend     │  │     Backend     │  │    Worker Pods      ││ │
│  │  │   Deployment    │  │   Deployment    │  │  (Eval Jobs)        ││ │
│  │  │   (3 replicas)  │  │   (3 replicas)  │  │  (Horizontal Pod    ││ │
│  │  │                 │  │                 │  │   Autoscaler)       ││ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌─────────────────────────────────┴──────────────────────────────────┐ │
│  │                      Managed Services                               │ │
│  │                                                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │  Cloud SQL   │  │Qdrant Cloud │  │   Neo4j AuraDB           │ │ │
│  │  │  (Postgres)  │  │             │  │                          │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  │                                                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Cloud Storage                              │ │ │
│  │  │              (Documents, Indexes, Artifacts)                  │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix

### Technology Stack Summary

| Component | Technology | Version |
|-----------|------------|---------|
| **Core Language** | Python | 3.11+ |
| **Backend Framework** | FastAPI | 0.100+ |
| **Frontend Framework** | React | 18+ |
| **Build Tool** | Vite | 5+ |
| **CSS Framework** | Tailwind CSS | 3+ |
| **ORM** | SQLModel | 0.0.14+ |
| **Vector DB (Semantic)** | ChromaDB | 0.4+ |
| **Vector DB (Hybrid)** | Qdrant | 1.7+ |
| **Graph DB** | Neo4j | 5+ |
| **Evaluation Framework** | DeepEval | 0.21+ |
| **Package Manager (Python)** | uv | latest |
| **Package Manager (Node)** | npm | 9+ |
| **Container Runtime** | Docker | 24+ |

### Related Documentation

- [Getting Started Guide](guides/getting-started.md) - First steps with the platform
- [RAG Strategies Guide](rag_strategies.md) - Deep dive into each RAG implementation
- [API Reference](api.md) - Complete API documentation
- [Custom RAG Integration](custom_rag_integration.md) - Build your own RAG
- [Deployment Guide](deployment.md) - Production deployment instructions
- [Troubleshooting Guide](guides/troubleshooting.md) - Common issues and solutions
