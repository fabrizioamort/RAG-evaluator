"""ChromaDB-based semantic search RAG implementation."""

import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.document_loaders import create_loader
from rag_evaluator.common.indexing import (
    CheckpointStore,
    SourceDocument,
    discover_source_documents,
    stable_hash,
    storage_id,
)
from rag_evaluator.common.openai_client import (
    embedding_client,
    llm_client,
    resolve_embedding_model,
    resolve_llm_model,
)
from rag_evaluator.common.provider_interfaces import (
    GeneratedAnswer,
    RetrievalTrace,
    RetrievedChunk,
    RetrievedContext,
)
from rag_evaluator.config import settings


class ChromaSemanticRAG(BaseRAG):
    """RAG implementation using ChromaDB for semantic vector search."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """Initialize ChromaDB semantic RAG.

        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_directory: Optional custom persistence directory (defaults to settings)
            config: Optional RAGConfig for LLM and embedding configuration
        """
        super().__init__("ChromaDB Semantic Search", config=config)

        # Initialize ChromaDB client
        persist_path = persist_directory or settings.chroma_persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Create or get collection
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize OpenAI-compatible clients (generation and embeddings are
        # independent endpoints resolved from the config).
        self.openai_client = llm_client(self.config)
        self.embedding_client = embedding_client(self.config)

        # Text splitter for chunking documents
        chunk_size = self.config.parameters.get("chunk_size", 1000)
        chunk_overlap = self.config.parameters.get("chunk_overlap", 200)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Track metrics
        self._retrieval_times: list[float] = []
        self._total_chunks = 0
        self._retrieval_times = []

    def close(self) -> None:
        """Close ChromaDB and OpenAI clients."""
        try:
            # ChromaDB doesn't have a public close() for PersistentClient in all versions,
            # but we can try to persistence if applicable or just nullify reference
            # In newer versions, PersistentClient handles teardown better
            self.client = None  # type: ignore[assignment]
            if hasattr(self, "openai_client"):
                self.openai_client.close()
            if hasattr(self, "embedding_client"):
                self.embedding_client.close()
        except Exception:
            pass

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        model = resolve_embedding_model(
            self.config, self.config.embedding_model or settings.embedding_model
        )
        response = self.embedding_client.embeddings.create(model=model, input=text)
        # Track embedding tokens
        if hasattr(response, "usage") and response.usage:
            self._token_usage.add_embedding_tokens(response.usage.total_tokens)
        return response.data[0].embedding  # type: ignore[no-any-return]

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents in ChromaDB.

        Args:
            documents_path: Path to the directory containing documents
        """
        self._prepare_documents(documents_path, checkpoint_store=None)

    def prepare_documents_resumable(
        self,
        documents_path: str,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Prepare and index documents in ChromaDB with durable checkpoints."""
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

    def _collection_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=ids, include=[])
            found = result.get("ids", []) if isinstance(result, dict) else []
            return {str(item) for item in found}
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
                existing = self._collection_existing_ids(list(completed))
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

        processed = 0
        global_chunk_index = 0
        for source, chunks in document_chunks:
            if checkpoint_store and not chunks:
                processed += checkpoint_store.ensure_document(source).chunk_count
                continue

            if checkpoint_store:
                checkpoint_store.start_document(source.doc_key)
                completed_chunks = checkpoint_store.completed_chunks(source.doc_key)
                existing_ids = self._collection_existing_ids(list(completed_chunks))
                for missing_id in set(completed_chunks) - existing_ids:
                    checkpoint_store.mark_chunk_pending(
                        missing_id,
                        "Completed checkpoint was missing from Chroma storage",
                    )
            else:
                completed_chunks = {}
                existing_ids = set()

            try:
                for local_index, chunk in enumerate(chunks):
                    chunk_hash = stable_hash(source.checksum, str(local_index), chunk.page_content)
                    chunk_id = storage_id("chroma", self.collection_name, source.doc_key, chunk_hash)
                    global_chunk_index += 1

                    if checkpoint_store:
                        checkpoint_store.ensure_chunk(
                            source.doc_key,
                            chunk_hash,
                            chunk_id,
                            local_index,
                        )
                        if chunk_id in completed_chunks and chunk_id in existing_ids:
                            processed += 1
                            continue
                        checkpoint_store.start_chunk(chunk_id)

                    metadata = {
                        "source": chunk.metadata.get("source", "unknown"),
                        "doc_key": source.doc_key,
                        "checksum": source.checksum,
                        "chunk_index": global_chunk_index - 1,
                        "local_chunk_index": local_index,
                    }

                    before_tokens = self._token_usage.embedding_tokens
                    embedding = self._get_embedding(chunk.page_content)
                    token_delta = self._token_usage.embedding_tokens - before_tokens

                    self.collection.upsert(
                        ids=[chunk_id],
                        documents=[chunk.page_content],
                        metadatas=[metadata],  # type: ignore[arg-type]
                        embeddings=[embedding],  # type: ignore[arg-type]
                    )

                    if checkpoint_store:
                        checkpoint_store.complete_chunk(chunk_id, token_delta)

                    processed += 1
                    self._report_progress(processed, total_chunks)
                    if checkpoint_store:
                        checkpoint_store.update_progress(
                            processed,
                            total_chunks,
                            {"document": source.relative_path},
                        )

                if checkpoint_store:
                    checkpoint_store.complete_document(source.doc_key, len(chunks))
            except Exception as e:
                if checkpoint_store:
                    checkpoint_store.fail_document(source.doc_key, str(e))
                raise

        self._total_chunks = total_chunks
        print(f"Successfully indexed {total_chunks} chunks in ChromaDB")

    def _retrieve_only(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Perform retrieval without generation.

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with context and metadata
        """
        # Get embedding for the question
        question_embedding = self._get_embedding(question)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],  # type: ignore[arg-type]
            n_results=top_k,
        )

        # Extract retrieved chunks
        retrieved_chunks = results["documents"][0] if results["documents"] else []
        retrieved_metadata = results["metadatas"][0] if results["metadatas"] else []
        distances_list = results.get("distances")
        distances = distances_list[0] if distances_list else []

        return {
            "context": retrieved_chunks,
            "metadata": {
                "sources": [meta.get("source", "unknown") for meta in retrieved_metadata],
                "chunk_indices": [meta.get("chunk_index", -1) for meta in retrieved_metadata],
                "distances": distances,
            },
        }

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve context for a question.

        Args:
            question: The question to retrieve context for
            top_k: Number of top documents to retrieve

        Returns:
            RetrievedContext with chunks and trace information
        """
        start_time = time.time()

        # Get embedding for the question
        question_embedding = self._get_embedding(question)
        embedding_time = time.time() - start_time

        # Query ChromaDB
        query_start = time.time()
        results = self.collection.query(
            query_embeddings=[question_embedding],  # type: ignore[arg-type]
            n_results=top_k,
        )
        query_time = time.time() - query_start

        # Extract retrieved chunks
        retrieved_chunks = results["documents"][0] if results["documents"] else []
        retrieved_metadata = results["metadatas"][0] if results["metadatas"] else []
        distances_list = results.get("distances")
        distances = distances_list[0] if distances_list else []

        retrieval_time = time.time() - start_time
        with self._metrics_lock:
            self._retrieval_times.append(retrieval_time)

        # Build chunk details
        chunk_details = []
        for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_metadata)):
            source = str(meta.get("source", "unknown"))
            distance = distances[i] if i < len(distances) else 0.0
            # Convert distance to similarity score (cosine distance -> similarity)
            score = 1.0 - distance if distance else 1.0

            chunk_details.append(
                RetrievedChunk(
                    content=chunk,
                    document_id=source,
                    chunk_id=f"chunk_{meta.get('chunk_index', i)}",
                    score=score,
                    rank=i,
                    source=source,
                    metadata={"distance": distance, **meta},
                )
            )

        # Build trace
        trace = RetrievalTrace(
            strategy="vector",
            total_duration_ms=retrieval_time * 1000,
        )
        trace.add_step(
            step_type="embedding",
            input_data={"query": question},
            output_refs=["query_embedding"],
            duration_ms=embedding_time * 1000,
        )
        trace.add_step(
            step_type="vector_search",
            input_data={"top_k": top_k, "collection": self.collection_name},
            output_refs=[c.chunk_id for c in chunk_details],
            duration_ms=query_time * 1000,
            metadata={"method": "cosine_similarity"},
        )
        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=retrieved_chunks,
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=retrieval_time,
        )

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

        prompt = f"""Answer the question using only the provided context. The context may state general rules or principles; apply them to the specific situation described in the question. For yes/no questions, give the direct conclusion first, then the supporting rule from the context. Only if the context contains no rule or information relevant to the question, reply exactly: "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Call OpenAI API
        from rag_evaluator.common.llm_utils import get_safe_llm_params

        model = resolve_llm_model(self.config, self.config.llm_model or settings.openai_model)
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
        """Query using semantic similarity search.

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

        # Get collection count
        collection_count = self.collection.count()

        return {
            "avg_retrieval_time": avg_retrieval_time,
            "total_chunks": collection_count,
            "total_queries": len(self._retrieval_times),
            "collection_name": self.collection_name,
            "token_usage": self.get_token_usage().to_dict(),
        }
