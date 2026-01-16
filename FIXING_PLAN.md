# FIXING PLAN: Immutable Knowledge Base Indexes

## 1. The Core Concept: Index as an Artifact

### The Misconception
Currently, the system treats "Indexing" as a state of the Knowledge Base (e.g., `KB.status = "indexed"`). This implies a KB can only be indexed in one way at a time.

### The Correction
Indexing is a **process** that transforms a **Knowledge Base Version** + **RAG Configuration** into a **Knowledge Base Index (Artifact)**.

*   **Immutable:** Once an index is created, it is never overwritten.
*   **Multiplicity:** You can have 10 different indexes for the same Knowledge Base (e.g., 5 Vector indexes with different chunk sizes, 3 Hybrid indexes, 2 Graph indexes).
*   **Independence:** Indexing with "Config A" creates a brand new artifact. Indexing with "Config B" creates another. They do not touch each other.

## 2. Database Schema Changes

### 2.1. New Model: `KnowledgeBaseIndex`
This is the central object for the new workflow.

```python
class KnowledgeBaseIndex(BaseModel):
    __tablename__ = "knowledge_base_indexes"

    id = Column(UUID, primary_key=True)
    knowledge_base_id = Column(UUID, ForeignKey("knowledge_bases.id"))
    rag_config_id = Column(UUID, ForeignKey("rag_configs.id"))
    
    # The user-friendly name, e.g., "Finance Docs - Hybrid (Chunk 500)"
    name = Column(String) 
    
    # Status of the *artifact creation*
    status = Column(String) # "creating", "ready", "failed"
    
    # Physical storage details
    # For Vector: The collection name (e.g., "idx_abc123")
    # For Graph: The graph name or scope
    physical_id = Column(String, unique=True) 
    
    # Snapshot of configuration used to build this index (Immutable)
    config_snapshot = Column(JSONB)
    
    created_at = Column(DateTime)
```

### 2.2. Changes to `KnowledgeBase`
*   **REMOVE** `index_path`: A KB does not have a single index path anymore.
*   **REMOVE** `status`: The KB itself is just a container for documents. It doesn't have an "indexing" status. (It might have a "processing" status for file parsing, but not for RAG indexing).

### 2.3. Changes to `Evaluation`
*   **ADD** `knowledge_base_index_id`: An evaluation must target a specific Index Artifact.
*   **REMOVE** `knowledge_base_id`: (Optional cleanup) The link to KB is now transitive via the Index.

## 3. Workflow Updates

### 3.1. The "Build Index" Action
Instead of "Indexing the KB", the user performs a **"Build Index"** action.

1.  User goes to **Knowledge Base > Indexes** tab.
2.  Clicks **"New Index"**.
3.  Selects a **RAG Configuration** (e.g., "Vector - High Precision").
4.  **System Action:**
    *   Creates `KnowledgeBaseIndex` record (ID: `idx_001`, Status: `creating`).
    *   Assigns a unique physical ID (e.g., `col_idx_001`).
    *   Spawns background task.
5.  **Background Task:**
    *   Reads documents from KB.
    *   Initializes RAG Implementation using `config_snapshot`.
    *   Writes data to storage `col_idx_001`.
    *   Sets `idx_001.status = "ready"`.

### 3.2. Multiple Indexes Scenario (The Fix)
If the user wants to test chunk sizes:
1.  Build Index -> Select Config "Chunk 500" -> Result: `Index A`.
2.  Build Index -> Select Config "Chunk 1000" -> Result: `Index B`.

Both `Index A` and `Index B` exist simultaneously. They are physically separate collections in the vector database/storage.

### 3.3. Evaluation Workflow
1.  User clicks **"New Evaluation"**.
2.  Selects **Test Set**.
3.  Selects **Knowledge Base Index** (Selecting `Index A` or `Index B`).
    *   *Note:* The RAG Configuration is inherent to the Index. The user doesn't pick "KB + Config" anymore; they pick the "Index" (which encapsulates both).
4.  The system runs the evaluation against the specific artifact.

## 4. Implementation Details

### 4.1. Storage Isolation
To ensure no overwrites, every RAG implementation must support **Explicit Collection Naming**.

*   **Chroma/Qdrant:** The `collection_name` will strictly be the `KnowledgeBaseIndex.physical_id` (e.g., a UUID).
*   **Filesystem:** The directory will be `storage/indexes/{physical_id}`.
*   **Neo4j:** This is trickier. We may need to use separate databases (Enterprise) or prefix labels with the `physical_id` (e.g., `(:Chunk_idx_001)` instead of just `(:Chunk)`). For the Open Source version, Label Prefixing is the viable path.

### 4.2. Migration Steps
1.  **Code:** Create the `KnowledgeBaseIndex` model.
2.  **API:** specific endpoints `/api/v1/indexes`.
3.  **UI:** Update the flow to separate "KB Management" from "Index Management".

## 5. Summary of Fix
By reifying the "Index" into a persistent database object, we eliminate ambiguity. Every "Run" of the indexer produces a unique, addressable artifact that can be evaluated, compared, or deleted independently.