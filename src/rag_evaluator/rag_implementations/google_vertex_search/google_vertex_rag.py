"""RAG implementation backed by Google Vertex AI Search (Discovery Engine) data stores.

Indexing stages local documents to Google Cloud Storage and imports them into a
Vertex AI Search **Data Store** with chunking enabled, letting Google handle
parsing/chunking/embedding. Retrieval queries the data store's default serving
config in ``CHUNKS`` mode. Generation defaults to the framework's configured LLM
(for apples-to-apples comparisons with other RAG types) but can optionally use
Vertex AI Search's grounded Answer API.

An existing data store can be reused for evaluation-only runs by setting
``reuse_existing_data_store=True`` with ``data_store_id`` — indexing becomes a
no-op that just validates the data store exists.
"""

from __future__ import annotations

import time
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.indexing import (
    CheckpointStore,
    SourceDocument,
    discover_source_documents,
    storage_id,
)
from rag_evaluator.common.llm_utils import get_safe_llm_params
from rag_evaluator.common.openai_client import llm_client
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.google_vertex_search import gcs_staging
from rag_evaluator.rag_implementations.google_vertex_search.client import (
    NotFound,
    branch_path,
    collection_path,
    data_store_path,
    discoveryengine,
    get_conversational_search_service_client,
    get_data_store_service_client,
    get_document_service_client,
    get_search_service_client,
    require_google_vertex,
    serving_config_path,
    validate_project_config,
)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class GoogleVertexSearchRAG(BaseRAG):
    """RAG implementation using a Google Vertex AI Search data store."""

    def __init__(
        self,
        data_store_id: str | None = None,
        staging_bucket: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """Initialize the Vertex AI Search RAG.

        Args:
            data_store_id: Vertex AI Search data store ID. Falls back to
                ``config.parameters["data_store_id"]``, then
                ``GOOGLE_VERTEX_DATA_STORE_ID``, then a generated ID.
            staging_bucket: GCS bucket used to stage documents for import. Falls
                back to ``config.parameters["staging_bucket"]``, then
                ``GOOGLE_VERTEX_STAGING_BUCKET``.
            config: Optional RAGConfig for LLM and parameter configuration.
        """
        super().__init__("Google Vertex AI Search", config=config)
        require_google_vertex()

        params = self.config.parameters

        self.project_id = settings.google_vertex_project_id
        self.location = params.get("location", settings.google_vertex_location)
        validate_project_config(self.project_id, self.location)

        self.sa_key_path = settings.google_vertex_sa_key_path

        self.data_store_id = (
            data_store_id
            or params.get("data_store_id")
            or settings.google_vertex_data_store_id
            or storage_id("gvs", self.config.name or "default")
        )
        self.reuse_existing_data_store = bool(params.get("reuse_existing_data_store", False))
        self.staging_bucket = (
            staging_bucket
            or params.get("staging_bucket")
            or settings.google_vertex_staging_bucket
        )

        self.num_previous_chunks = _clamp(int(params.get("num_previous_chunks", 2)), 0, 3)
        self.num_next_chunks = _clamp(int(params.get("num_next_chunks", 2)), 0, 3)
        self.generation_mode = params.get("generation_mode", settings.google_vertex_generation_mode)

        self.openai_client = llm_client(self.config)

        self._data_store_service = get_data_store_service_client(self.location, self.sa_key_path)
        self._document_service = get_document_service_client(self.location, self.sa_key_path)
        self._search_service = get_search_service_client(self.location, self.sa_key_path)

        # Detected from the data store; CHUNKS mode requires chunking enabled at
        # creation time. BYO data stores without chunking fall back to DOCUMENTS.
        self._chunking_enabled = True
        self._data_store_created = False
        self._import_duration_seconds = 0.0
        self._total_documents = 0

        self._retrieval_times: list[float] = []

    def close(self) -> None:
        """Close the OpenAI-compatible client used for framework generation."""
        try:
            if hasattr(self, "openai_client") and self.openai_client:
                self.openai_client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Data store lifecycle
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Validate the configured data store exists without mutating it."""
        self._validate_existing_data_store()

    def _validate_existing_data_store(self) -> None:
        name = data_store_path(self.project_id, self.location, self.data_store_id)
        try:
            data_store = self._data_store_service.get_data_store(name=name)
        except NotFound as e:
            raise ValueError(
                f"Vertex AI Search data store '{self.data_store_id}' was not found in "
                f"project '{self.project_id}' (location '{self.location}')."
            ) from e

        self._chunking_enabled = self._detect_chunking_enabled(data_store)

    @staticmethod
    def _detect_chunking_enabled(data_store: Any) -> bool:
        """Best-effort detection of whether layout chunking is enabled."""
        try:
            processing_config = data_store.document_processing_config
            return "chunking_config" in processing_config
        except (AttributeError, TypeError):
            # Fall back to assuming chunking is enabled if presence checks are
            # unavailable (e.g. mocked clients in tests).
            return True

    def _ensure_data_store(self) -> None:
        """Get or create the data store with chunking enabled."""
        name = data_store_path(self.project_id, self.location, self.data_store_id)
        try:
            data_store = self._data_store_service.get_data_store(name=name)
            self._data_store_created = False
            self._chunking_enabled = self._detect_chunking_enabled(data_store)
            return
        except NotFound:
            pass

        chunking_config = discoveryengine.DocumentProcessingConfig.ChunkingConfig(
            layout_based_chunking_config=(
                discoveryengine.DocumentProcessingConfig.ChunkingConfig.LayoutBasedChunkingConfig(
                    chunk_size=500,
                    include_ancestor_headings=True,
                )
            ),
        )
        document_processing_config = discoveryengine.DocumentProcessingConfig(
            chunking_config=chunking_config,
            default_parsing_config=discoveryengine.DocumentProcessingConfig.ParsingConfig(
                layout_parsing_config=(
                    discoveryengine.DocumentProcessingConfig.ParsingConfig.LayoutParsingConfig()
                ),
            ),
        )
        data_store = discoveryengine.DataStore(
            display_name=self.config.name or self.data_store_id,
            industry_vertical=discoveryengine.IndustryVertical.GENERIC,
            solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
            content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
            document_processing_config=document_processing_config,
        )
        operation = self._data_store_service.create_data_store(
            parent=collection_path(self.project_id, self.location),
            data_store=data_store,
            data_store_id=self.data_store_id,
        )
        operation.result()  # Wait for the create-data-store LRO to finish.
        self._data_store_created = True
        self._chunking_enabled = True

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def prepare_documents(self, documents_path: str) -> None:
        """Stage documents to GCS and import them into the data store."""
        self._prepare_documents(documents_path, checkpoint_store=None)

    def prepare_documents_resumable(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Stage and import documents with durable per-document checkpoints."""
        self._prepare_documents(documents_path, checkpoint_store=checkpoint_store)

    def _prepare_documents(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore | None,
    ) -> None:
        if self.reuse_existing_data_store and self.data_store_id:
            self._validate_existing_data_store()
            self._total_documents = 0
            return

        if not self.staging_bucket:
            raise ValueError(
                "google_vertex_search indexing requires a GCS staging bucket. "
                "Set GOOGLE_VERTEX_STAGING_BUCKET or the 'staging_bucket' parameter."
            )

        self._ensure_data_store()

        sources = discover_source_documents(documents_path)
        total = len(sources)
        storage_client = gcs_staging.get_storage_client(self.sa_key_path)

        gcs_uris: list[str] = []
        for i, source in enumerate(sources):
            if checkpoint_store:
                checkpoint_store.ensure_document(source)
                checkpoint_store.start_document(source.doc_key)

            uri = self._stage_document(storage_client, source, checkpoint_store)
            gcs_uris.append(uri)

            if checkpoint_store:
                checkpoint_store.complete_document(source.doc_key, 1)
            self._report_progress(i + 1, total)
            if checkpoint_store:
                checkpoint_store.update_progress(i + 1, total, {"document": source.relative_path})

        import_start = time.time()
        self._import_documents(gcs_uris)
        self._import_duration_seconds = time.time() - import_start

        self._total_documents = total

    def _stage_document(
        self,
        storage_client: Any,
        source: SourceDocument,
        checkpoint_store: CheckpointStore | None,
    ) -> str:
        """Upload one document to GCS (skipping if already uploaded), return its gs:// URI."""
        blob_name = gcs_staging.gcs_blob_name(self.data_store_id, source.relative_path)
        uri = gcs_staging.gcs_uri(self.staging_bucket, self.data_store_id, source.relative_path)

        upload_storage_id = storage_id("gcs", self.data_store_id, source.doc_key)
        if checkpoint_store:
            checkpoint_store.ensure_chunk(source.doc_key, source.checksum, upload_storage_id, 0)
            completed = checkpoint_store.completed_chunks(source.doc_key)
            already_uploaded = upload_storage_id in completed
        else:
            already_uploaded = False

        if already_uploaded:
            already_uploaded = gcs_staging.blob_exists(storage_client, self.staging_bucket, blob_name)

        if not already_uploaded:
            if checkpoint_store:
                checkpoint_store.start_chunk(upload_storage_id)
            gcs_staging.upload_file(storage_client, self.staging_bucket, source.source_path, blob_name)
            if checkpoint_store:
                checkpoint_store.complete_chunk(upload_storage_id)

        return uri

    def _import_documents(self, gcs_uris: list[str]) -> None:
        """Import staged documents via a long-running operation (LRO), polled to completion."""
        if not gcs_uris:
            return

        request = discoveryengine.ImportDocumentsRequest(
            parent=branch_path(self.project_id, self.location, self.data_store_id),
            gcs_source=discoveryengine.GcsSource(
                input_uris=gcs_uris,
                data_schema="content",
            ),
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )
        operation = self._document_service.import_documents(request=request)
        operation.result()  # Block until the import LRO completes.

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve chunks (or document snippets) from the Vertex AI Search data store."""
        start_time = time.time()

        serving_config = serving_config_path(self.project_id, self.location, self.data_store_id)
        content_search_spec = self._content_search_spec()

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=question,
            page_size=top_k,
            content_search_spec=content_search_spec,
        )

        query_start = time.time()
        response = self._search_service.search(request)
        query_time = time.time() - query_start

        chunk_details = [
            self._result_to_chunk(i, result) for i, result in enumerate(response.results)
        ]

        retrieval_time = time.time() - start_time
        with self._metrics_lock:
            self._retrieval_times.append(retrieval_time)

        trace = RetrievalTrace(strategy="vector", total_duration_ms=retrieval_time * 1000)
        trace.add_step(
            step_type="vertex_search",
            input_data={
                "query": question,
                "data_store_id": self.data_store_id,
                "top_k": top_k,
                "search_result_mode": "CHUNKS" if self._chunking_enabled else "DOCUMENTS",
            },
            output_refs=[c.chunk_id for c in chunk_details],
            duration_ms=query_time * 1000,
        )
        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=[c.content for c in chunk_details],
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=retrieval_time,
        )

    def _content_search_spec(self) -> Any:
        if self._chunking_enabled:
            return discoveryengine.SearchRequest.ContentSearchSpec(
                search_result_mode=(
                    discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
                ),
                chunk_spec=discoveryengine.SearchRequest.ContentSearchSpec.ChunkSpec(
                    num_previous_chunks=self.num_previous_chunks,
                    num_next_chunks=self.num_next_chunks,
                ),
            )
        return discoveryengine.SearchRequest.ContentSearchSpec(
            search_result_mode=(
                discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS
            ),
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
            ),
        )

    def _result_to_chunk(self, rank: int, result: Any) -> RetrievedChunk:
        if self._chunking_enabled and getattr(result, "chunk", None) and result.chunk.content:
            chunk = result.chunk
            doc_metadata = chunk.document_metadata
            source = doc_metadata.uri if doc_metadata else "unknown"
            title = doc_metadata.title if doc_metadata else ""
            return RetrievedChunk(
                content=chunk.content,
                document_id=source,
                chunk_id=chunk.id or f"chunk_{rank}",
                score=float(chunk.relevance_score or 0.0),
                rank=rank,
                source=source,
                metadata={"title": title} if title else {},
            )

        # DOCUMENTS fallback (BYO data stores without chunking enabled).
        document = result.document
        struct_data = dict(document.derived_struct_data or {})
        snippets = struct_data.get("snippets") or []
        content = snippets[0].get("snippet", "") if snippets else ""
        source = struct_data.get("link", "unknown")
        return RetrievedChunk(
            content=content,
            document_id=source,
            chunk_id=document.id or f"doc_{rank}",
            score=0.0,
            rank=rank,
            source=source,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer:
        """Generate an answer, using the framework LLM or Google's grounded Answer API."""
        start_time = time.time()

        if self.generation_mode == "google_grounded":
            answer = self._generate_grounded(question)
        else:
            answer = self._generate_only(question, context.chunks)

        generation_time = time.time() - start_time
        return GeneratedAnswer(
            text=answer,
            generation_time=generation_time,
            prompt_tokens=self._token_usage.prompt_tokens,
            completion_tokens=self._token_usage.completion_tokens,
        )

    def _generate_only(self, question: str, context_chunks: list[str]) -> str:
        context_text = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))

        prompt = f"""Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        model = self.config.llm_model or settings.openai_model
        completion_params = get_safe_llm_params(
            model,
            temperature=0.0,
            reasoning_effort=self.config.llm_reasoning_effort,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response = self.openai_client.chat.completions.create(**completion_params)

        if response.usage:
            self._token_usage.add_prompt_tokens(response.usage.prompt_tokens)
            self._token_usage.add_completion_tokens(response.usage.completion_tokens)

        return response.choices[0].message.content or "No answer generated"

    def _generate_grounded(self, question: str) -> str:
        """Generate an answer using Vertex AI Search's grounded Answer API."""
        client = get_conversational_search_service_client(self.location, self.sa_key_path)
        serving_config = (
            f"{data_store_path(self.project_id, self.location, self.data_store_id)}"
            f"/servingConfigs/default_serving_config"
        )
        request = discoveryengine.AnswerQueryRequest(
            serving_config=serving_config,
            query=discoveryengine.Query(text=question),
        )
        response = client.answer_query(request)
        return response.answer.answer_text or "No answer generated"

    # ------------------------------------------------------------------
    # Query / metrics
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve from the data store and generate an answer."""
        self.reset_token_usage()
        start_time = time.time()

        context = self.retrieve(question, top_k)
        answer = self.generate(question, context)

        total_time = time.time() - start_time

        return {
            "answer": answer.text,
            "context": context.chunks,
            "metadata": {
                "retrieval_time": context.retrieval_time,
                "generation_time": answer.generation_time,
                "chunks_retrieved": len(context.chunks),
                "sources": [c.source for c in context.chunk_details],
                "data_store_id": self.data_store_id,
                "generation_mode": self.generation_mode,
                "token_usage": self._token_usage.to_dict(),
                "total_time": total_time,
            },
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return retrieval/import metrics and data store info."""
        avg_retrieval_time = (
            sum(self._retrieval_times) / len(self._retrieval_times)
            if self._retrieval_times
            else 0.0
        )

        return {
            "avg_retrieval_time": avg_retrieval_time,
            "total_queries": len(self._retrieval_times),
            "total_documents": self._total_documents,
            "data_store_id": self.data_store_id,
            "location": self.location,
            "reused_existing_data_store": self.reuse_existing_data_store,
            "data_store_created": self._data_store_created,
            "chunking_enabled": self._chunking_enabled,
            "import_duration_seconds": self._import_duration_seconds,
            "generation_mode": self.generation_mode,
            "token_usage": self.get_token_usage().to_dict(),
        }
