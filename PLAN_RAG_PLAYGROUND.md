# Implementation Plan: RAG Playground Feature

## Overview

Add an interactive RAG playground feature that allows users to:
1. Query indexed documents manually with a selected RAG system
2. View retrieval details (chunks, scores, traces)
3. Compare multiple RAG systems side-by-side on the same query

---

## Phase 1: Backend API

### 1.1 New API Router: `playground.py`

**Location:** `platform/backend/app/api/playground.py`

**Endpoints:**

```python
# Execute a query against one or more indexes
POST /api/playground/query
Request Body:
{
  "question": str,
  "index_ids": list[int],  # Support multiple indexes for comparison
  "top_k": int = 5         # Number of chunks to retrieve
}

Response:
{
  "results": [
    {
      "index_id": int,
      "index_name": str,
      "rag_type": str,
      "answer": str,
      "retrieved_context": {
        "chunks": list[str],
        "chunk_details": [
          {
            "content": str,
            "document_id": str,
            "chunk_id": str,
            "score": float,
            "rank": int,
            "source": str,
            "metadata": dict
          }
        ]
      },
      "trace": {
        "strategy": str,
        "steps": list[dict],
        "total_duration_ms": float,
        "fusion_details": dict | null
      },
      "metrics": {
        "retrieval_time_ms": float,
        "generation_time_ms": float,
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
      },
      "error": str | null  # If this specific RAG failed
    }
  ],
  "query_id": str  # UUID for optional history tracking
}
```

```python
# Get available indexes for playground (only ready indexes)
GET /api/playground/indexes
Query Params:
  - project_id: int (optional, filter by project)
  - kb_id: int (optional, filter by knowledge base)

Response:
{
  "indexes": [
    {
      "id": int,
      "name": str,
      "rag_type": str,
      "knowledge_base_name": str,
      "project_name": str,
      "document_count": int,
      "chunk_count": int,
      "status": str
    }
  ]
}
```

### 1.2 New Service: `playground_service.py`

**Location:** `platform/backend/app/services/playground_service.py`

**Responsibilities:**
- Execute queries against multiple RAG indexes in parallel (using asyncio.gather)
- Reuse existing `RAGAdapter` for RAG instantiation
- Handle errors gracefully per-index (don't fail entire request if one RAG fails)
- Track token usage and timing metrics

**Key Implementation:**
```python
async def execute_playground_query(
    question: str,
    index_ids: list[int],
    top_k: int,
    db: AsyncSession
) -> PlaygroundQueryResponse:
    # 1. Load indexes and validate they're ready
    # 2. For each index, instantiate RAG via RAGAdapter
    # 3. Execute queries in parallel using asyncio.gather
    # 4. Collect results with error handling per-index
    # 5. Return aggregated response
```

### 1.3 Schema Updates

**Location:** `platform/backend/app/schemas/playground.py`

Define Pydantic models for request/response validation:
- `PlaygroundQueryRequest`
- `PlaygroundQueryResult`
- `PlaygroundQueryResponse`
- `PlaygroundIndexInfo`

---

## Phase 2: Frontend - Playground Page

### 2.1 New Page: `Playground.tsx`

**Location:** `platform/frontend/src/pages/Playground.tsx`

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  RAG Playground                                    [Project ▼]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Select RAG Systems to Compare                            │   │
│  │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │   │
│  │ │ ☑ Index A    │ │ ☑ Index B    │ │ ☐ Index C    │ ... │   │
│  │ │ vector_sem.  │ │ hybrid       │ │ graph_rag    │      │   │
│  │ │ KB: Docs     │ │ KB: Docs     │ │ KB: Other    │      │   │
│  │ └──────────────┘ └──────────────┘ └──────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Ask a question...                              [Ask ▶]  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Top K: [5 ▼]                                                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Results                                                        │
│  ┌────────────────────────┬────────────────────────┐           │
│  │ Index A (vector_sem.)  │ Index B (hybrid)       │           │
│  ├────────────────────────┼────────────────────────┤           │
│  │ Answer:                │ Answer:                │           │
│  │ "The capital of..."    │ "Paris is the..."     │           │
│  │                        │                        │           │
│  │ ⏱ 245ms | 🎯 5 chunks  │ ⏱ 312ms | 🎯 5 chunks │           │
│  │ 💰 0.002 USD           │ 💰 0.003 USD          │           │
│  ├────────────────────────┼────────────────────────┤           │
│  │ [Retrieved Chunks ▼]   │ [Retrieved Chunks ▼]  │           │
│  │ ┌──────────────────┐   │ ┌──────────────────┐  │           │
│  │ │ 1. chunk content │   │ │ 1. chunk content │  │           │
│  │ │    score: 0.92   │   │ │    score: 0.89   │  │           │
│  │ │    source: doc1  │   │ │    source: doc2  │  │           │
│  │ └──────────────────┘   │ └──────────────────┘  │           │
│  │ [View Full Trace]      │ [View Full Trace]     │           │
│  └────────────────────────┴────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### `IndexSelector.tsx`
**Location:** `platform/frontend/src/components/playground/IndexSelector.tsx`

- Multi-select card grid for choosing indexes
- Shows RAG type, KB name, document/chunk counts
- Optional filtering by project/KB
- Visual indication of selected items (max 4 for comparison)

#### `QueryInput.tsx`
**Location:** `platform/frontend/src/components/playground/QueryInput.tsx`

- Text input with submit button
- Top-K selector dropdown
- Loading state during query execution
- Keyboard shortcut (Ctrl+Enter to submit)

#### `ComparisonResults.tsx`
**Location:** `platform/frontend/src/components/playground/ComparisonResults.tsx`

- Responsive grid layout (1-4 columns based on selection count)
- Each result card shows:
  - Index name and RAG type badge
  - Generated answer
  - Quick metrics (time, tokens, cost)
  - Expandable chunks list
  - Button to open full trace viewer modal

#### `ResultCard.tsx`
**Location:** `platform/frontend/src/components/playground/ResultCard.tsx`

- Single RAG result display
- Collapsible sections for answer, chunks, metrics
- Reuses existing `RetrievalTraceViewer` in modal

### 2.3 API Client Updates

**Location:** `platform/frontend/src/api/client.ts`

Add new functions:
```typescript
export async function executePlaygroundQuery(
  question: string,
  indexIds: number[],
  topK: number = 5
): Promise<PlaygroundQueryResponse>

export async function getPlaygroundIndexes(
  projectId?: number,
  kbId?: number
): Promise<PlaygroundIndexInfo[]>
```

Add TypeScript types:
```typescript
interface PlaygroundQueryRequest { ... }
interface PlaygroundQueryResult { ... }
interface PlaygroundQueryResponse { ... }
interface PlaygroundIndexInfo { ... }
```

### 2.4 Navigation Update

**Location:** `platform/frontend/src/App.tsx` (or wherever routes are defined)

- Add new route: `/playground`
- Add navigation link in sidebar/header

---

## Phase 3: Enhancements (Optional)

### 3.1 Query History (Optional)

**Backend:**
- New model `PlaygroundQuery` to store query history
- Endpoint `GET /api/playground/history` to retrieve past queries
- Auto-cleanup of old queries (configurable retention)

**Frontend:**
- Query history sidebar/dropdown
- Click to re-run or view past results

### 3.2 Export Results (Optional)

- Export comparison results as JSON/CSV
- Export for documentation/reporting

### 3.3 Chunk Highlighting (Optional)

- In the answer text, highlight which parts came from which chunks
- Visual connection between answer and source chunks

---

## Implementation Order

### Step 1: Backend Foundation
1. Create `playground.py` router with endpoints
2. Create `playground_service.py` with query execution logic
3. Create `playground.py` schemas
4. Register router in `main.py`
5. Test endpoints via API docs (Swagger)

### Step 2: Frontend Foundation
1. Create `Playground.tsx` page with basic layout
2. Add route and navigation
3. Create `IndexSelector.tsx` component
4. Wire up index fetching

### Step 3: Query Execution
1. Create `QueryInput.tsx` component
2. Implement `executePlaygroundQuery` API call
3. Handle loading/error states

### Step 4: Results Display
1. Create `ComparisonResults.tsx` grid
2. Create `ResultCard.tsx` with answer and metrics
3. Add chunks display (collapsible)
4. Integrate `RetrievalTraceViewer` modal

### Step 5: Polish
1. Responsive design adjustments
2. Error handling improvements
3. Empty states and helpful messages
4. Performance optimizations

---

## File Changes Summary

### New Files
```
platform/backend/app/api/playground.py          # API endpoints
platform/backend/app/services/playground_service.py  # Business logic
platform/backend/app/schemas/playground.py      # Pydantic models

platform/frontend/src/pages/Playground.tsx      # Main page
platform/frontend/src/components/playground/
  ├── IndexSelector.tsx                         # Multi-select indexes
  ├── QueryInput.tsx                            # Question input
  ├── ComparisonResults.tsx                     # Results grid
  └── ResultCard.tsx                            # Single result card
```

### Modified Files
```
platform/backend/app/main.py                    # Register new router
platform/frontend/src/api/client.ts             # Add API functions
platform/frontend/src/App.tsx                   # Add route
platform/frontend/src/components/layout/...     # Add nav link
```

---

## Considerations

### Performance
- Parallel query execution for multiple indexes
- Consider caching RAG instances (they're expensive to initialize)
- Add timeout handling for slow RAG systems

### Error Handling
- Individual RAG failures shouldn't fail entire comparison
- Clear error messages for common issues (index not ready, connection failed)
- Graceful degradation when some results fail

### UX
- Clear visual feedback during query execution
- Show which indexes are being queried with individual progress
- Allow cancellation of in-flight queries
- Disable submit while query is running

### Security
- Validate user has access to requested indexes
- Rate limiting on query endpoint (RAG queries are expensive)
- Input sanitization for questions

---

## Design Decisions (Confirmed)

1. **Query History**: ✅ Persisted - queries will be stored in database for later review
2. **Maximum Comparisons**: ✅ 4 simultaneous comparisons max
3. **Access Control**: ✅ No restrictions - all indexes available to all users
4. **Cost Display**: ✅ Prominently displayed per query result
5. **Response Streaming**: ⏸️ Deferred to v2 - adds significant complexity (SSE/WebSocket, streaming state management, comparing multiple streams). Will implement standard request/response for v1.

---

## Estimated Scope

- **Backend**: ~300-400 lines of new code
- **Frontend**: ~600-800 lines of new code
- **Components reused**: `RetrievalTraceViewer`, existing UI components (cards, badges, buttons)

This plan leverages existing infrastructure (RAGAdapter, RetrievalTraceViewer, artifact storage) to minimize new code while delivering a full-featured playground experience.
