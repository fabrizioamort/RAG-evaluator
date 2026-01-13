# RAG Evaluation Platform - Implementation Tasks

This document breaks down the Evolution Plan into actionable, granular tasks.
Each task has clear acceptance criteria and dependencies.

**Branch:** `feature/platform-v2`
**Started:** 2026-01-12
**Target:** Open Source Edition v1.0

---

## Legend

- [ ] Not started
- [x] Completed
- 🔄 In progress
- ⏸️ Blocked
- 📋 Has subtasks

Dependencies are noted as `[Depends: TASK_ID]`

---

## Phase 1: Foundation

**Goal:** Working backend/frontend skeletons with database connectivity

### 1.1 Project Structure Setup

- [x] **P1.1.1** Create `platform/` directory structure

  ```
  platform/
  ├── backend/
  │   ├── app/
  │   │   ├── __init__.py
  │   │   ├── models/
  │   │   ├── schemas/
  │   │   ├── api/
  │   │   ├── services/
  │   │   └── utils/
  │   ├── alembic/
  │   ├── tests/
  │   └── pyproject.toml
  └── frontend/
      ├── src/
      ├── package.json
      └── vite.config.ts
  ```

  **Acceptance:** All directories exist with `__init__.py` where needed

- [x] **P1.1.2** Create `platform/backend/pyproject.toml`
  - Dependencies: fastapi, uvicorn, sqlalchemy[asyncio], alembic, pydantic-settings, asyncpg, aiosqlite, litellm, httpx, structlog
  - Dev dependencies: pytest, pytest-asyncio, pytest-cov, ruff, mypy
  **Acceptance:** `uv sync` succeeds in `platform/backend/`

- [x] **P1.1.3** Create `platform/frontend/package.json`
  - Dependencies: react, react-dom, react-router-dom, @tanstack/react-query, axios
  - Dev dependencies: vite, typescript, tailwindcss, postcss, autoprefixer, @types/react
  **Acceptance:** `npm install` succeeds

- [x] **P1.1.4** Configure Tailwind CSS + shadcn/ui
  - Create `tailwind.config.js`
  - Create `postcss.config.js`
  - Initialize shadcn/ui with `npx shadcn-ui@latest init`
  **Acceptance:** Tailwind classes work in components

### 1.2 Backend Core Setup

- [x] **P1.2.1** Create `app/config.py` with Pydantic Settings

  ```python
  class Settings(BaseSettings):
      DATABASE_URL: str
      STORAGE_PATH: str = "./storage"
      LOG_LEVEL: str = "INFO"
      OPENAI_API_KEY: str | None = None
      # ... etc
  ```

  **Acceptance:** Settings load from `.env` and environment variables

- [x] **P1.2.2** Create `app/database.py` with SQLAlchemy async engine
  - Support both PostgreSQL (`asyncpg`) and SQLite (`aiosqlite`)
  - Async session factory
  - `get_db` dependency for FastAPI
  **Acceptance:** Can connect to both SQLite and PostgreSQL

- [x] **P1.2.3** Create `app/main.py` FastAPI application
  - CORS middleware configured
  - Exception handlers
  - Router includes
  - Lifespan events (startup/shutdown)
  **Acceptance:** `uvicorn app.main:app` starts without errors

- [x] **P1.2.4** Create health endpoint `GET /api/v1/health`
  - Returns `{"status": "healthy", "database": "connected"}`
  - Actually checks DB connectivity
  **Acceptance:** Returns 200 when DB is up, 503 when down

- [x] **P1.2.5** Setup structured logging (`app/utils/logging_config.py`)
  - JSON format with structlog
  - Request ID context variable
  - Log level from config
  **Acceptance:** Logs are JSON formatted with request_id

### 1.3 Database Models (SQLAlchemy)

- [x] **P1.3.1** Create base model class (`app/models/base.py`)
  - UUID primary key mixin
  - Timestamp mixin (created_at, updated_at)
  - Declarative base
  **Acceptance:** Base classes importable

- [x] **P1.3.2** Create `app/models/project.py`
  - Fields: id, name, description, status, tags, created_at, updated_at
  - Relationships to knowledge_bases, test_sets, rag_configs, evaluations, webhooks
  **Acceptance:** Model can be imported, relationships defined

- [x] **P1.3.3** Create `app/models/knowledge_base.py`
  - Fields per schema
  - Relationship to documents, versions
  **Acceptance:** Model matches schema

- [x] **P1.3.4** Create `app/models/knowledge_base_version.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.5** Create `app/models/document.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.6** Create `app/models/test_template.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.7** Create `app/models/test_set.py`
  - Relationship to test_cases
  **Acceptance:** Model matches schema

- [x] **P1.3.8** Create `app/models/test_case.py`
  - Relationship to template, provenance artifact
  **Acceptance:** Model matches schema

- [x] **P1.3.9** Create `app/models/test_generation_job.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.10** Create `app/models/rag_config.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.11** Create `app/models/artifact.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.12** Create `app/models/run_manifest.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.13** Create `app/models/evaluation.py`
  - Relationships to kb, kb_version, test_set, rag_config, manifest, results
  **Acceptance:** Model matches schema

- [x] **P1.3.14** Create `app/models/evaluation_job.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.15** Create `app/models/evaluation_result.py`
  - Relationships to artifacts
  **Acceptance:** Model matches schema

- [x] **P1.3.16** Create `app/models/webhook.py`
  **Acceptance:** Model matches schema

- [x] **P1.3.17** Create `app/models/__init__.py` exporting all models
  **Acceptance:** `from app.models import Project, KnowledgeBase, ...` works

### 1.4 Alembic Migrations

- [x] **P1.4.1** Initialize Alembic (`alembic init alembic`)
  - Configure `alembic.ini` for async
  - Update `env.py` for async engine
  **Acceptance:** `alembic` commands work

- [x] **P1.4.2** Create initial migration
  - All tables from Phase 1.3
  - All indexes
  **Acceptance:** `alembic upgrade head` creates all tables

- [x] **P1.4.3** Test migration on SQLite
  **Acceptance:** Migration works with SQLite URL

- [x] **P1.4.4** Test migration on PostgreSQL (Docker)
  **Acceptance:** Migration works with PostgreSQL URL

### 1.5 Docker Configuration

- [x] **P1.5.1** Create `platform/backend/Dockerfile`
  - Multi-stage build
  - UV for dependency management
  - Non-root user
  **Acceptance:** `docker build` succeeds

- [x] **P1.5.2** Create `docker/docker-compose.yml`
  - PostgreSQL service
  - Backend service
  - Frontend service (placeholder)
  - Volumes for data persistence
  **Acceptance:** `docker-compose up` starts all services

- [x] **P1.5.3** Create `docker/docker-compose.dev.yml`
  - SQLite mode
  - Hot reload for backend
  **Acceptance:** Dev mode works without PostgreSQL

- [x] **P1.5.4** Create `docker/init-db.sql`
  - Create database if not exists
  **Acceptance:** PostgreSQL initializes correctly

### 1.6 Frontend Foundation

- [x] **P1.6.1** Create Vite + React + TypeScript setup
  - `vite.config.ts`
  - `tsconfig.json`
  - `src/main.tsx`
  **Acceptance:** `npm run dev` starts dev server

- [x] **P1.6.2** Configure React Router
  - `src/App.tsx` with router setup
  - Basic route structure (/, /projects, /evaluations)
  **Acceptance:** Routes navigate correctly

- [x] **P1.6.3** Setup TanStack Query
  - QueryClient provider
  - Default options (staleTime, retry)
  **Acceptance:** useQuery works in components

- [x] **P1.6.4** Create API client (`src/api/client.ts`)
  - Axios instance with base URL from env
  - Request/response interceptors for errors
  **Acceptance:** API calls work

- [x] **P1.6.5** Create basic layout component
  - Sidebar navigation
  - Header
  - Main content area
  **Acceptance:** Layout renders correctly

- [x] **P1.6.6** Install core shadcn/ui components
  - Button, Card, Input, Label, Select, Table, Tabs, Dialog, Toast
  **Acceptance:** Components importable and styled

- [x] **P1.6.7** Create placeholder pages
  - Dashboard (/)
  - Projects (/projects)
  - Not Found (404)
  **Acceptance:** Pages render with placeholder content

### 1.7 Backend Tests Setup

- [x] **P1.7.1** Create `tests/conftest.py`
  - Test database fixture (SQLite in-memory)
  - Async client fixture
  - Test settings override
  **Acceptance:** Fixtures available in tests

- [x] **P1.7.2** Create `tests/test_health.py`
  - Test health endpoint returns 200
  - Test health with DB down returns 503
  **Acceptance:** Tests pass

- [x] **P1.7.3** Setup pytest configuration in `pyproject.toml`
  - asyncio_mode = "auto"
  - Test paths
  - Coverage settings
  **Acceptance:** `pytest` runs correctly

### 1.8 Documentation

- [x] **P1.8.1** Create `platform/backend/README.md`
  - Setup instructions
  - Environment variables
  - Running locally
  **Acceptance:** New developer can set up from README

- [x] **P1.8.2** Create `.env.example` for backend
  **Acceptance:** All required vars documented

---

## Phase 2: Core CRUD + Storage

**Goal:** Full CRUD for all entities, document upload, KB versioning

### 2.1 Pydantic Schemas

- [x] **P2.1.1** Create `app/schemas/base.py`
  - Base response model with id, created_at
  - Pagination schema (offset, limit, total)
  **Acceptance:** Base schemas importable

- [x] **P2.1.2** Create `app/schemas/project.py`
  - ProjectCreate, ProjectUpdate, ProjectResponse, ProjectList
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.3** Create `app/schemas/knowledge_base.py`
  - KBCreate, KBResponse, KBWithDocuments, DocumentUploadResponse
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.4** Create `app/schemas/test_set.py`
  - TestSetCreate, TestSetResponse, TestCaseCreate, TestCaseResponse
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.5** Create `app/schemas/test_template.py`
  - TemplateCreate, TemplateResponse
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.6** Create `app/schemas/rag_config.py`
  - RAGConfigCreate, RAGConfigResponse, RAGTypeInfo
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.7** Create `app/schemas/evaluation.py`
  - EvaluationCreate, EvaluationResponse, EvaluationResultResponse
  - ProgressEvent schema for SSE
  **Acceptance:** Schemas validate correctly

- [x] **P2.1.8** Create `app/schemas/webhook.py`
  - WebhookCreate, WebhookResponse, WebhookTest
  **Acceptance:** Schemas validate correctly

### 2.2 Projects API

- [x] **P2.2.1** Create `app/api/deps.py`
  - get_db dependency
  - Common query params (pagination)
  **Acceptance:** Dependencies injectable

- [x] **P2.2.2** Create `app/api/projects.py`
  - GET /projects (list with filters)
  - POST /projects (create)
  - GET /projects/{id} (detail)
  - PUT /projects/{id} (update)
  - DELETE /projects/{id} (delete)
  - POST /projects/{id}/archive
  **Acceptance:** All endpoints work, tested

- [x] **P2.2.3** Write tests for projects API
  - CRUD operations
  - Validation errors
  - Not found errors
  **Acceptance:** 100% endpoint coverage

### 2.3 Knowledge Bases API

- [x] **P2.3.1** Create `app/services/storage_service.py`
  - Save uploaded file to storage path
  - Generate unique filenames
  - Calculate checksums
  **Acceptance:** Files saved correctly

- [x] **P2.3.2** Create `app/api/knowledge_bases.py`
  - GET /projects/{pid}/knowledge-bases
  - POST /projects/{pid}/knowledge-bases
  - GET /knowledge-bases/{id}
  - DELETE /knowledge-bases/{id}
  - POST /knowledge-bases/{id}/documents (multipart upload)
  - DELETE /knowledge-bases/{id}/documents/{docId}
  - GET /knowledge-bases/{id}/versions
  **Acceptance:** All endpoints work

- [x] **P2.3.3** Implement KB versioning logic
  - Auto-increment version on document changes
  - Store document snapshot per version
  **Acceptance:** Versions created on changes

- [x] **P2.3.4** Write tests for KB API
  **Acceptance:** Full coverage

### 2.4 Artifact Store Service

- [x] **P2.4.1** Create `app/services/artifact_store.py`
  - Content-addressed storage (SHA256)
  - Store to filesystem
  - Retrieve by key
  - Deduplication
  **Acceptance:** Artifacts stored and retrievable

- [x] **P2.4.2** Write tests for artifact store
  - Store/retrieve cycle
  - Deduplication works
  - Large content handling
  **Acceptance:** Tests pass

### 2.5 Test Sets API

- [x] **P2.5.1** Create `app/api/test_sets.py`
  - Full CRUD for test sets
  - CRUD for test cases within sets
  - Import/export JSON endpoints
  **Acceptance:** All endpoints work

- [x] **P2.5.2** Implement JSON import/export
  - Validate imported structure
  - Export with metadata
  **Acceptance:** Round-trip works

- [x] **P2.5.3** Write tests for test sets API
  **Acceptance:** Full coverage

### 2.6 Test Templates API

- [ ] **P2.6.1** Create `app/api/test_templates.py`
  - GET /test-templates (list builtin + custom)
  - POST /test-templates (create custom)
  - PUT /test-templates/{id}
  - DELETE /test-templates/{id} (not builtin)
  **Acceptance:** All endpoints work

- [ ] **P2.6.2** Load builtin templates on startup
  - Read from `data/templates/builtin_templates.json`
  - Insert if not exists
  **Acceptance:** Builtin templates available

- [ ] **P2.6.3** Create `data/templates/builtin_templates.json`
  - 6 template types per plan
  **Acceptance:** Valid JSON, templates useful

### 2.7 RAG Configs API

- [ ] **P2.7.1** Create `app/api/rag_configs.py`
  - CRUD for RAG configs
  - GET /rag-types (list available)
  - GET /rag-types/{type}/parameters (schema)
  - GET /llm-providers (list available)
  **Acceptance:** All endpoints work

- [ ] **P2.7.2** Define RAG type registry
  - Map type name to implementation class
  - Parameter schemas per type
  **Acceptance:** Types discoverable

### 2.8 LLM Provider Service

- [ ] **P2.8.1** Create `app/services/llm_provider.py`
  - LiteLLM integration
  - Support openai, ollama, anthropic
  - Token counting
  **Acceptance:** Can call multiple providers

- [ ] **P2.8.2** Write tests with mocked LLM calls
  **Acceptance:** Provider switching works

### 2.9 Frontend - Projects UI

- [ ] **P2.9.1** Create Projects list page
  - Table with name, status, KB count, evaluation count
  - Create button
  - Status filter
  **Acceptance:** Lists projects from API

- [ ] **P2.9.2** Create Project detail page
  - Project info card
  - Tabs: Knowledge Bases, Test Sets, RAG Configs, Evaluations
  - Edit/Archive buttons
  **Acceptance:** Shows project details

- [ ] **P2.9.3** Create Project create/edit dialog
  - Form with validation
  - Tags input
  **Acceptance:** Can create/edit projects

### 2.10 Frontend - Knowledge Bases UI

- [ ] **P2.10.1** Create KB list component (within project)
  - Table with name, status, doc count, version
  - Create button
  **Acceptance:** Lists KBs

- [ ] **P2.10.2** Create KB detail view
  - Document list with upload
  - Version history
  - Index status
  **Acceptance:** Shows KB details

- [ ] **P2.10.3** Create document upload component
  - Drag-and-drop zone
  - Multi-file upload
  - Progress indicator
  **Acceptance:** Documents upload successfully

### 2.11 Frontend - Test Sets UI

- [ ] **P2.11.1** Create Test Sets list component
  **Acceptance:** Lists test sets

- [ ] **P2.11.2** Create Test Set detail view
  - Test cases table
  - Add/edit/delete cases
  - Import/export buttons
  **Acceptance:** Full CRUD works

- [ ] **P2.11.3** Create Test Case editor
  - Question, expected answer, context fields
  - Difficulty, category selects
  **Acceptance:** Can edit test cases

### 2.12 Frontend - RAG Configs UI

- [ ] **P2.12.1** Create RAG Configs list component
  **Acceptance:** Lists configs

- [ ] **P2.12.2** Create RAG Config editor
  - Type selector
  - Dynamic parameter form based on type
  - LLM provider selector
  **Acceptance:** Can create/edit configs

---

## Phase 3: Evaluation Engine + Progress

**Goal:** Working evaluation pipeline with real-time progress

### 3.1 Core Library Modifications

- [ ] **P3.1.1** Modify `src/rag_evaluator/common/base_rag.py`
  - Add RAGConfig dataclass
  - Add retrieve() abstract method
  - Add generate() abstract method
  - Update query() to use retrieve+generate
  **Acceptance:** Existing tests still pass

- [ ] **P3.1.2** Create `src/rag_evaluator/common/provider_interfaces.py`
  - RetrievedChunk, RetrievalTrace, RetrievedContext
  - GeneratedAnswer
  - LLMProvider, EmbeddingProvider ABCs
  **Acceptance:** Interfaces importable

- [ ] **P3.1.3** Create `src/rag_evaluator/common/token_tracker.py`
  - TokenUsage dataclass
  **Acceptance:** Token tracking works

- [ ] **P3.1.4** Update VectorSemanticRAG for new interface
  - Implement retrieve()
  - Implement generate()
  - Return RetrievalTrace
  **Acceptance:** Existing functionality preserved

- [ ] **P3.1.5** Update VectorHybridRAG for new interface
  **Acceptance:** Works with new interface

- [ ] **P3.1.6** Update GraphRAG for new interface
  **Acceptance:** Works with new interface

- [ ] **P3.1.7** Update FilesystemRAG for new interface
  **Acceptance:** Works with new interface

### 3.2 RAG Adapter Service

- [ ] **P3.2.1** Create `app/services/rag_adapter.py`
  - Instantiate RAG from RAGConfig model
  - Map config parameters to RAG constructor
  - Handle index paths
  **Acceptance:** Can create any RAG type from config

### 3.3 Evaluation Job Management

- [ ] **P3.3.1** Create `app/services/job_event_log.py`
  - Persisted event log
  - SSE stream creation
  - Checkpoint save/restore
  **Acceptance:** Events persisted, streams work

- [ ] **P3.3.2** Create `app/services/job_checkpoint_service.py`
  - Save checkpoint with results so far
  - Restore checkpoint for resume
  **Acceptance:** Checkpoints work

- [ ] **P3.3.3** Create `app/services/evaluation_runner.py`
  - Main evaluation loop
  - Progress reporting
  - Checkpoint every N test cases
  - Handle pause/cancel
  **Acceptance:** Evaluations run to completion

### 3.4 Evaluation API

- [ ] **P3.4.1** Create `app/api/evaluations.py`
  - POST /evaluations (start new)
  - GET /evaluations/{id}
  - GET /evaluations/{id}/results (paginated)
  - GET /evaluations/{id}/stream (SSE)
  - POST /evaluations/{id}/cancel
  - POST /evaluations/{id}/pause
  - POST /evaluations/{id}/resume
  - POST /evaluations/{id}/retry
  **Acceptance:** All endpoints work

- [ ] **P3.4.2** Implement SSE endpoint
  - EventSourceResponse
  - Reconnection support (Last-Event-ID)
  - State reconstruction from DB
  **Acceptance:** Frontend receives progress events

- [ ] **P3.4.3** Create run manifest on evaluation start
  - Snapshot all config
  - Store library versions
  **Acceptance:** Manifest created and retrievable

### 3.5 Cost Tracking

- [ ] **P3.5.1** Create `app/services/cost_tracker.py`
  - Token to cost calculation
  - Per-model pricing
  - Aggregate costs
  **Acceptance:** Costs calculated correctly

- [ ] **P3.5.2** Create `app/utils/pricing_defaults.py`
  - Default prices for common models
  - Configurable overrides
  **Acceptance:** Pricing available

### 3.6 Store Results with Artifacts

- [ ] **P3.6.1** Store retrieved_context as artifact
  **Acceptance:** Context stored as artifact

- [ ] **P3.6.2** Store retrieval_trace as artifact
  **Acceptance:** Trace stored as artifact

- [ ] **P3.6.3** Store raw_metrics as artifact
  **Acceptance:** Metrics stored as artifact

### 3.7 Frontend - Evaluation Progress

- [ ] **P3.7.1** Create `useEvaluationStream` hook
  - SSE connection management
  - Reconnection logic
  - State updates
  **Acceptance:** Hook works with SSE

- [ ] **P3.7.2** Create EvaluationProgress component
  - Progress bar
  - Test case counter
  - Elapsed/remaining time
  - Pause/Cancel buttons
  **Acceptance:** Shows live progress

- [ ] **P3.7.3** Create "Start Evaluation" wizard
  - Select KB, Test Set, RAG Config
  - Review and confirm
  - Navigate to progress view
  **Acceptance:** Can start evaluations

### 3.8 Tests

- [ ] **P3.8.1** Test evaluation lifecycle
  - Start, progress, complete
  **Acceptance:** Full lifecycle tested

- [ ] **P3.8.2** Test pause/resume
  **Acceptance:** Checkpoints work

- [ ] **P3.8.3** Test cancel
  **Acceptance:** Cancellation works

---

## Phase 4: Test Generation + Quality

**Goal:** LLM-based test generation with quality gates

### 4.1 Test Generator Service

- [ ] **P4.1.1** Create `app/services/test_generator_service.py`
  - Generate questions from KB chunks
  - Use templates for structure
  - Progress tracking
  **Acceptance:** Generates test cases

- [ ] **P4.1.2** Implement generation prompts
  - System prompt for question generation
  - Template-guided generation
  **Acceptance:** Quality questions generated

### 4.2 Quality Gate Service

- [ ] **P4.2.1** Create `app/services/test_quality_gate.py`
  - Exact duplicate detection
  - Semantic duplicate detection
  - Answerability validation
  - Length checks
  **Acceptance:** Low-quality rejected

- [ ] **P4.2.2** Implement provenance tracking
  - Store which chunks generated each question
  **Acceptance:** Provenance stored

### 4.3 Generation API

- [ ] **P4.3.1** Add POST /test-sets/{id}/generate
  - Config: count, templates, difficulty distribution
  - Returns job ID
  **Acceptance:** Generation starts

- [ ] **P4.3.2** Add GET /test-sets/{id}/generation-status
  - Progress, rejected count
  **Acceptance:** Status retrievable

- [ ] **P4.3.3** Add POST /test-sets/{id}/cases/bulk-review
  - Approve/reject multiple cases
  **Acceptance:** Bulk operations work

### 4.4 Frontend - Test Generation

- [ ] **P4.4.1** Create TestGeneratorWizard component
  - Select KB
  - Configure count, difficulty
  - Select templates
  **Acceptance:** Can configure generation

- [ ] **P4.4.2** Create generation progress view
  - Progress bar
  - Generated/rejected counters
  **Acceptance:** Shows progress

- [ ] **P4.4.3** Create review interface
  - List generated cases
  - Approve/reject/edit buttons
  - Bulk actions
  **Acceptance:** Can review cases

### 4.5 Tests

- [ ] **P4.5.1** Test generation with mocked LLM
  **Acceptance:** Generation tested

- [ ] **P4.5.2** Test quality gates
  **Acceptance:** Gates working

---

## Phase 5: Results, Traces & Explainability

**Goal:** Rich results display with explanations

### 5.1 Metric Explainability

- [ ] **P5.1.1** Store judge reasoning per metric
  - Modify DeepEval calls to capture reasoning
  - Store in evaluation_result
  **Acceptance:** Reasoning captured

- [ ] **P5.1.2** Create MetricExplainability component
  - Show score with expandable reasoning
  - Color coding by score
  **Acceptance:** Explanations visible

### 5.2 Retrieval Trace Viewer

- [ ] **P5.2.1** Add GET /evaluations/{id}/trace/{resultId}
  - Return trace artifact content
  **Acceptance:** Trace retrievable

- [ ] **P5.2.2** Create RetrievalTraceViewer component
  - Step-by-step visualization
  - Chunk details
  - Timing breakdown
  **Acceptance:** Traces visualized

### 5.3 Run Manifest Display

- [ ] **P5.3.1** Add GET /evaluations/{id}/manifest
  **Acceptance:** Manifest retrievable

- [ ] **P5.3.2** Create ManifestViewer component
  - Show all config at eval time
  - Collapsible sections
  **Acceptance:** Manifest displayed

### 5.4 Baseline Tracking

- [ ] **P5.4.1** Add POST /evaluations/{id}/set-baseline
  - Mark as baseline with reason
  - Only one baseline per project
  **Acceptance:** Baseline settable

- [ ] **P5.4.2** Add GET /projects/{pid}/baseline
  **Acceptance:** Baseline retrievable

- [ ] **P5.4.3** Create BaselineComparison component
  - Side-by-side with baseline
  - Delta indicators
  **Acceptance:** Comparison works

### 5.5 Comparison API

- [ ] **P5.5.1** Create `app/api/comparisons.py`
  - POST /comparisons (compare 2+ evals)
  - GET /comparisons/{id}
  **Acceptance:** Comparisons work

- [ ] **P5.5.2** Create comparison logic
  - Aggregate metrics
  - Per-question deltas
  **Acceptance:** Comparison calculated

### 5.6 Trend Analysis

- [ ] **P5.6.1** Create `app/services/trend_analysis_service.py`
  - Aggregate metrics over time
  - Per RAG config trends
  **Acceptance:** Trends calculated

- [ ] **P5.6.2** Create `app/api/trends.py`
  - GET /projects/{pid}/trends
  - GET /rag-configs/{id}/trends
  **Acceptance:** Trends retrievable

- [ ] **P5.6.3** Create TrendChart component
  - Line chart with metrics over time
  - Configurable date range
  **Acceptance:** Charts render

### 5.7 Export

- [ ] **P5.7.1** Create `app/services/report_exporter.py`
  - JSON export
  - Markdown export
  **Acceptance:** Exports work

- [ ] **P5.7.2** Add GET /evaluations/{id}/report?format=
  **Acceptance:** Download works

### 5.8 Frontend - Evaluation Detail Page

- [ ] **P5.8.1** Create full EvaluationDetail page
  - Summary metrics cards
  - Results by difficulty chart
  - Detailed results table
  **Acceptance:** Full detail view

- [ ] **P5.8.2** Add expandable result rows
  - Show reasoning on expand
  - Link to trace viewer
  **Acceptance:** Results explorable

---

## Phase 6: Polish & Documentation

**Goal:** Production-ready release

### 6.1 Webhook System

- [ ] **P6.1.1** Create `app/services/webhook_service.py`
  - HMAC-SHA256 signatures
  - Retry logic
  - Failure tracking
  **Acceptance:** Webhooks delivered

- [ ] **P6.1.2** Create `app/api/webhooks.py`
  - CRUD (max 3 per project)
  - POST /webhooks/{id}/test
  **Acceptance:** API works

- [ ] **P6.1.3** Trigger webhooks on events
  - evaluation.completed
  - evaluation.failed
  - generation.completed
  **Acceptance:** Events trigger webhooks

### 6.2 Error Handling

- [ ] **P6.2.1** Create exception hierarchy
  - NotFoundError, ValidationError, ConflictError
  **Acceptance:** Exceptions defined

- [ ] **P6.2.2** Create error response schemas
  - Consistent error format
  **Acceptance:** Errors formatted

- [ ] **P6.2.3** Add exception handlers to FastAPI
  **Acceptance:** Errors return correct status codes

### 6.3 Frontend Polish

- [ ] **P6.3.1** Add loading states to all pages
  **Acceptance:** Loading indicators show

- [ ] **P6.3.2** Add empty states
  - "No projects yet" etc.
  **Acceptance:** Empty states helpful

- [ ] **P6.3.3** Add toast notifications
  - Success/error feedback
  **Acceptance:** Toasts work

- [ ] **P6.3.4** Dashboard implementation
  - Quick stats
  - Recent activity
  - Quick actions
  **Acceptance:** Dashboard complete

### 6.4 Documentation

- [ ] **P6.4.1** Create `docs/api.md`
  - Full API reference
  - Authentication (none for OSS)
  - Examples
  **Acceptance:** API documented

- [ ] **P6.4.2** Create deployment guide
  - Docker deployment
  - Environment configuration
  - Backup/restore
  **Acceptance:** Deployment documented

- [ ] **P6.4.3** Update main README.md
  - Feature overview
  - Quick start
  - Screenshots
  **Acceptance:** README comprehensive

### 6.5 Production Build

- [ ] **P6.5.1** Optimize backend Dockerfile
  - Multi-stage build
  - Minimal image
  **Acceptance:** Image < 500MB

- [ ] **P6.5.2** Create frontend Dockerfile
  - Build static assets
  - Nginx serving
  **Acceptance:** Frontend builds

- [ ] **P6.5.3** Production docker-compose
  - Resource limits
  - Health checks
  - Restart policies
  **Acceptance:** Production ready

### 6.6 End-to-End Testing

- [ ] **P6.6.1** Create E2E test suite
  - Full workflow test
  - API integration tests
  **Acceptance:** E2E passes

- [ ] **P6.6.2** Manual verification checklist
  - All 17 items from plan
  **Acceptance:** All items verified

---

## Task Summary

| Phase | Tasks | Est. Effort |
|-------|-------|-------------|
| Phase 1: Foundation | 32 | Medium |
| Phase 2: Core CRUD | 35 | Medium |
| Phase 3: Evaluation Engine | 24 | High |
| Phase 4: Test Generation | 14 | Medium |
| Phase 5: Results & Traces | 21 | Medium |
| Phase 6: Polish | 18 | Low-Medium |
| **Total** | **144** | |

---

## Notes

- Tasks are roughly ordered by dependency
- Each task should be completable in 1-4 hours
- Test tasks can be done alongside implementation
- Frontend tasks can parallel backend if API contracts defined first
