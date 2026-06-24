"""Hybrid search RAG implementation combining semantic and keyword search using Qdrant."""

import time
from pathlib import Path
from typing import Any
from uuid import UUID

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.document_loaders import create_loader
from rag_evaluator.common.indexing import (
    CheckpointStore,
    SourceDocument,
    discover_source_documents,
    stable_hash,
)
from rag_evaluator.common.openai_client import embedding_client, llm_client
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
from rag_evaluator.config import settings


class HybridSearchRAG(BaseRAG):
    """RAG implementation using Qdrant hybrid search (dense + sparse vectors)."""

    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """Initialize hybrid search RAG with Qdrant.

        Args:
            collection_name: Name of the Qdrant collection (defaults to settings)
            qdrant_url: Qdrant server URL (defaults to settings)
            config: Optional RAGConfig for LLM and embedding configuration
        """
        super().__init__("Hybrid Search (Semantic + Keyword)", config=config)

        # Initialize Qdrant client — raise on version incompatibility instead of silently warning
        import warnings

        self.qdrant_url = qdrant_url or settings.qdrant_url
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.client = QdrantClient(url=self.qdrant_url)
        for w in caught_warnings:
            if "incompatible" in str(w.message).lower():
                raise ConnectionError(
                    f"Qdrant version mismatch: {w.message}. "
                    "Update qdrant-client in pyproject.toml or the server image in docker-compose.yml "
                    "so that minor versions differ by at most 1."
                )

        # Initialize OpenAI-compatible clients (generation and dense embeddings
        # are independent endpoints resolved from the config).
        self.openai_client = llm_client(self.config)
        self.embedding_client = embedding_client(self.config)

        # Initialize FastEmbed for sparse embeddings (SPLADE)
        self.sparse_model = SparseTextEmbedding(
            model_name=self.config.parameters.get("sparse_model_name", settings.sparse_model_name),
        )

        # Text splitter for chunking (smaller chunks for hybrid search)
        chunk_size = self.config.parameters.get("chunk_size", settings.hybrid_chunk_size)
        chunk_overlap = self.config.parameters.get("chunk_overlap", settings.hybrid_chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Collection setup
        self.collection_name = collection_name or settings.qdrant_collection_name
        self._ensure_collection()

        # Metrics tracking
        self._retrieval_times: list[float] = []
        self._total_chunks = 0
        self._retrieval_times = []

    def close(self) -> None:
        """Close Qdrant and OpenAI clients."""
        try:
            if hasattr(self, "client") and self.client:
                self.client.close()
                self.client = None  # type: ignore[assignment]
            if hasattr(self, "openai_client") and self.openai_client:
                self.openai_client.close()
            if hasattr(self, "embedding_client") and self.embedding_client:
                self.embedding_client.close()
        except Exception:
            pass

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist with both dense and sparse vectors."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=1536,  # text-embedding-3-small dimension
                            distance=models.Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(),
                    },
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Qdrant at {self.qdrant_url}. "
                f"Make sure Qdrant is running (docker compose up -d). Error: {e}"
            ) from e

    def _get_dense_embedding(self, text: str) -> list[float]:
        """Get dense embedding from OpenAI.

        Args:
            text: Text to embed

        Returns:
            Dense embedding vector (1536 dimensions)
        """
        model = self.config.embedding_model or settings.embedding_model
        response = self.embedding_client.embeddings.create(
            model=model,
            input=text,
        )
        # Track embedding tokens
        if hasattr(response, "usage") and response.usage:
            self._token_usage.add_embedding_tokens(response.usage.total_tokens)
        return response.data[0].embedding  # type: ignore[no-any-return]

    def _get_sparse_embedding(self, text: str) -> models.SparseVector:
        """Get sparse embedding from FastEmbed SPLADE model.

        Args:
            text: Text to embed

        Returns:
            Sparse vector with indices and values
        """
        # FastEmbed returns a generator, get first result
        embeddings = list(self.sparse_model.embed([text]))
        sparse_emb = embeddings[0]

        return models.SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

    def _point_id(self, doc_key: str, chunk_hash: str) -> str:
        """Return a deterministic UUID string accepted by Qdrant."""
        return str(UUID(hex=stable_hash(self.collection_name, doc_key, chunk_hash, length=32)))

    def _process_batch(
        self,
        batch_chunks: list[LangChainDocument],
        point_ids: list[str],
        chunk_indices: list[int],
    ) -> list[models.PointStruct]:
        """Process a batch of chunks: generate embeddings and create points.

        Args:
            batch_chunks: List of document chunks
            point_ids: Deterministic point IDs for each chunk
            chunk_indices: Stable chunk indices for metadata

        Returns:
            List of Qdrant points
        """
        texts = [chunk.page_content for chunk in batch_chunks]

        # Batch dense embeddings (OpenAI-compatible)
        model = self.config.embedding_model or settings.embedding_model
        dense_response = self.embedding_client.embeddings.create(
            model=model,
            input=texts,
        )
        # Track embedding tokens
        if hasattr(dense_response, "usage") and dense_response.usage:
            self._token_usage.add_embedding_tokens(dense_response.usage.total_tokens)
        dense_embeddings = [data.embedding for data in dense_response.data]

        # Batch sparse embeddings (FastEmbed)
        # FastEmbed returns a generator, so we convert to list
        sparse_embeddings = list(self.sparse_model.embed(texts, batch_size=len(texts)))

        points = []
        for i, (text, dense, sparse) in enumerate(zip(texts, dense_embeddings, sparse_embeddings)):
            chunk = batch_chunks[i]
            chunk_index = chunk_indices[i]

            sparse_vec = models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            )

            point = models.PointStruct(
                id=point_ids[i],
                vector={
                    "dense": dense,
                    "sparse": sparse_vec,
                },
                payload={
                    "text": text,
                    "source": chunk.metadata.get("source", "unknown"),
                    "doc_key": chunk.metadata.get("doc_key"),
                    "checksum": chunk.metadata.get("checksum"),
                    "chunk_index": chunk_index,
                },
            )
            points.append(point)

        return points

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents with both dense and sparse vectors.

        Args:
            documents_path: Path to the directory containing documents
        """
        self._prepare_documents(documents_path, checkpoint_store=None)

    def prepare_documents_resumable(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Prepare and index documents with durable checkpoints."""
        self._prepare_documents(documents_path, checkpoint_store=checkpoint_store)

    def _load_document(self, source: SourceDocument) -> LangChainDocument | None:
        try:
            loader = create_loader(source.source_path)
            doc = loader.load(source.source_path)
            return LangChainDocument(
                page_content=doc.content,
                metadata={
                    "source": doc.source,
                    "doc_key": source.doc_key,
                    "checksum": source.checksum,
                    **doc.metadata,
                },
            )
        except Exception as e:
            print(f"Warning: Failed to load {Path(source.source_path).name}: {e}")
            return None

    def _existing_point_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=False,
                with_vectors=False,
            )
            return {str(record.id) for record in records}
        except Exception:
            return set()

    def _prepare_documents(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore | None,
    ) -> None:
        sources = discover_source_documents(documents_path)
        document_chunks: list[tuple[SourceDocument, list[LangChainDocument]]] = []

        for source in sources:
            checkpoint = checkpoint_store.ensure_document(source) if checkpoint_store else None
            if checkpoint and checkpoint.status == "completed":
                assert checkpoint_store is not None
                completed = checkpoint_store.completed_chunks(source.doc_key)
                existing = self._existing_point_ids(list(completed))
                if len(existing) == len(completed) and checkpoint.chunk_count:
                    document_chunks.append((source, []))
                    continue

            document = self._load_document(source)
            if document is None:
                continue
            chunks = self.text_splitter.split_documents([document])
            document_chunks.append((source, chunks))
            print(f"Loaded: {Path(source.source_path).name}")

        total_chunks = sum(len(chunks) for _, chunks in document_chunks)
        if total_chunks == 0:
            completed_total = sum(
                checkpoint_store.ensure_document(source).chunk_count
                for source, chunks in document_chunks
                if checkpoint_store and not chunks
            )
            if completed_total == 0:
                raise ValueError(f"No documents found in {documents_path}")
            self._total_chunks = completed_total
            return

        print(f"Loaded {len(sources)} documents, split into {total_chunks} chunks")

        batch_size = settings.hybrid_indexing_batch_size
        processed = 0
        global_chunk_index = 0

        for source, chunks in document_chunks:
            if checkpoint_store and not chunks:
                processed += checkpoint_store.ensure_document(source).chunk_count
                continue

            if checkpoint_store:
                checkpoint_store.start_document(source.doc_key)
                completed_chunks = checkpoint_store.completed_chunks(source.doc_key)
                existing_ids = self._existing_point_ids(list(completed_chunks))
                for missing_id in set(completed_chunks) - existing_ids:
                    checkpoint_store.mark_chunk_pending(
                        missing_id,
                        "Completed checkpoint was missing from Qdrant storage",
                    )
            else:
                completed_chunks = {}
                existing_ids = set()

            pending_batch: list[LangChainDocument] = []
            pending_ids: list[str] = []
            pending_indices: list[int] = []

            try:
                for local_index, chunk in enumerate(chunks):
                    chunk_hash = stable_hash(source.checksum, str(local_index), chunk.page_content)
                    point_id = self._point_id(source.doc_key, chunk_hash)
                    global_chunk_index += 1

                    if checkpoint_store:
                        checkpoint_store.ensure_chunk(
                            source.doc_key,
                            chunk_hash,
                            point_id,
                            local_index,
                        )
                        if point_id in completed_chunks and point_id in existing_ids:
                            processed += 1
                            continue
                        checkpoint_store.start_chunk(point_id)

                    pending_batch.append(chunk)
                    pending_ids.append(point_id)
                    pending_indices.append(global_chunk_index - 1)

                    if len(pending_batch) >= batch_size:
                        processed += self._flush_batch(
                            pending_batch,
                            pending_ids,
                            pending_indices,
                            checkpoint_store,
                            processed,
                            total_chunks,
                            source.relative_path,
                        )
                        pending_batch = []
                        pending_ids = []
                        pending_indices = []

                if pending_batch:
                    processed += self._flush_batch(
                        pending_batch,
                        pending_ids,
                        pending_indices,
                        checkpoint_store,
                        processed,
                        total_chunks,
                        source.relative_path,
                    )

                if checkpoint_store:
                    checkpoint_store.complete_document(source.doc_key, len(chunks))
            except Exception as e:
                if checkpoint_store:
                    checkpoint_store.fail_document(source.doc_key, str(e))
                raise

        self._total_chunks = total_chunks
        print(f"Successfully indexed {total_chunks} chunks in Qdrant (hybrid mode)")

    def _flush_batch(
        self,
        batch: list[LangChainDocument],
        point_ids: list[str],
        chunk_indices: list[int],
        checkpoint_store: CheckpointStore | None,
        processed_before: int,
        total_chunks: int,
        relative_path: str,
    ) -> int:
        before_tokens = self._token_usage.embedding_tokens
        points = self._process_batch(batch, point_ids, chunk_indices)
        token_delta = self._token_usage.embedding_tokens - before_tokens

        self.client.upsert(collection_name=self.collection_name, points=points)

        per_point_tokens = token_delta // len(point_ids) if point_ids else 0
        for point_id in point_ids:
            if checkpoint_store:
                checkpoint_store.complete_chunk(point_id, per_point_tokens)

        processed = processed_before + len(point_ids)
        print(f"Processed and uploaded {processed}/{total_chunks} chunks")
        self._report_progress(processed, total_chunks)
        if checkpoint_store:
            checkpoint_store.update_progress(
                processed,
                total_chunks,
                {"document": relative_path},
            )
        return len(point_ids)

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve context using hybrid search (dense + sparse with RRF fusion).

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            RetrievedContext with chunks and trace information
        """
        start_time = time.time()

        # Get both embeddings for the question
        dense_start = time.time()
        dense_vec = self._get_dense_embedding(question)
        dense_time = time.time() - dense_start

        sparse_start = time.time()
        sparse_vec = self._get_sparse_embedding(question)
        sparse_time = time.time() - sparse_start

        # Hybrid search with RRF fusion
        # Prefetch more candidates from each search, then fuse
        prefetch_limit = top_k * 4  # Prefetch more for better fusion

        query_start = time.time()
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        query_time = time.time() - query_start

        # Extract retrieved chunks
        retrieved_chunks = []
        chunk_details = []

        for i, point in enumerate(results.points):
            if point.payload:
                text = point.payload.get("text", "")
                source = point.payload.get("source", "unknown")
                chunk_idx = point.payload.get("chunk_index", -1)

                retrieved_chunks.append(text)
                chunk_details.append(
                    RetrievedChunk(
                        content=text,
                        document_id=source,
                        chunk_id=f"chunk_{chunk_idx}",
                        score=point.score if point.score else 0.0,
                        rank=i,
                        source=source,
                        metadata={
                            "chunk_index": chunk_idx,
                            "fusion_score": point.score,
                        },
                    )
                )

        retrieval_time = time.time() - start_time
        with self._metrics_lock:
            self._retrieval_times.append(retrieval_time)

        # Build trace
        trace = RetrievalTrace(
            strategy="hybrid",
            total_duration_ms=retrieval_time * 1000,
            fusion_details={
                "method": "RRF",
                "prefetch_limit": prefetch_limit,
                "k": 60,  # RRF default k
            },
        )
        trace.add_step(
            step_type="dense_embedding",
            input_data={"query": question},
            output_refs=["dense_vector"],
            duration_ms=dense_time * 1000,
        )
        trace.add_step(
            step_type="sparse_embedding",
            input_data={"query": question, "model": "SPLADE"},
            output_refs=["sparse_vector"],
            duration_ms=sparse_time * 1000,
        )
        trace.add_step(
            step_type="hybrid_search",
            input_data={
                "top_k": top_k,
                "collection": self.collection_name,
                "fusion": "RRF",
            },
            output_refs=[c.chunk_id for c in chunk_details],
            duration_ms=query_time * 1000,
            metadata={
                "dense_prefetch": prefetch_limit,
                "sparse_prefetch": prefetch_limit,
            },
        )
        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=retrieved_chunks,
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=retrieval_time,
        )

    def _retrieve_only(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Perform retrieval without generation.

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with context and metadata
        """
        context = self.retrieve(question, top_k)
        return {
            "context": context.chunks,
            "metadata": {
                "sources": [c.source for c in context.chunk_details],
                "scores": [c.score for c in context.chunk_details],
                "fusion_method": "RRF",
            },
        }

    def _generate_only(self, question: str, context_chunks: list[str]) -> str:
        """Generate answer from context without retrieval.

        Args:
            question: The question to answer
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer text
        """
        # Generate answer using LLM with retrieved context
        context_text = "\n\n".join([f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Call OpenAI API
        from rag_evaluator.common.llm_utils import get_safe_llm_params

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

        # Track token usage
        if response.usage:
            self._token_usage.add_prompt_tokens(response.usage.prompt_tokens)
            self._token_usage.add_completion_tokens(response.usage.completion_tokens)

        return response.choices[0].message.content or "No answer generated"

    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer:
        """Generate answer from retrieved context.

        Args:
            question: The question to answer
            context: Previously retrieved context

        Returns:
            GeneratedAnswer with text and token usage
        """
        start_time = time.time()

        answer = self._generate_only(question, context.chunks)

        generation_time = time.time() - start_time

        return GeneratedAnswer(
            text=answer,
            generation_time=generation_time,
            prompt_tokens=self._token_usage.prompt_tokens,
            completion_tokens=self._token_usage.completion_tokens,
        )

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using hybrid search (dense + sparse with RRF fusion).

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing answer, context, and metadata
        """
        # Reset token usage for this query
        self.reset_token_usage()

        # Start timing
        start_time = time.time()

        # Retrieve context
        context = self.retrieve(question, top_k)

        # Generate answer
        answer = self._generate_only(question, context.chunks)

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "context": context.chunks,
            "metadata": {
                "retrieval_time": context.retrieval_time,
                "chunks_retrieved": len(context.chunks),
                "sources": [c.source for c in context.chunk_details],
                "fusion_method": "RRF",
                "token_usage": self._token_usage.to_dict(),
                "total_time": total_time,
            },
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        avg_retrieval_time = (
            sum(self._retrieval_times) / len(self._retrieval_times)
            if self._retrieval_times
            else 0.0
        )

        # Get collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
        except Exception:
            total_points = self._total_chunks

        return {
            "avg_retrieval_time": avg_retrieval_time,
            "total_chunks": total_points,
            "total_queries": len(self._retrieval_times),
            "collection_name": self.collection_name,
            "fusion_method": "RRF",
            "chunk_size": self.config.parameters.get("chunk_size", settings.hybrid_chunk_size),
            "chunk_overlap": self.config.parameters.get(
                "chunk_overlap", settings.hybrid_chunk_overlap
            ),
            "token_usage": self.get_token_usage().to_dict(),
        }
