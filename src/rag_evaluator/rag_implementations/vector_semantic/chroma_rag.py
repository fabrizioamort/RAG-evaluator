"""ChromaDB-based semantic search RAG implementation."""

import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from rag_evaluator.common.document_loaders import create_loader
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

        # Initialize OpenAI client with timeout
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )

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
        except Exception:
            pass

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        model = self.config.embedding_model or settings.embedding_model
        response = self.openai_client.embeddings.create(model=model, input=text)
        # Track embedding tokens
        if hasattr(response, "usage") and response.usage:
            self._token_usage.add_embedding_tokens(response.usage.total_tokens)
        return response.data[0].embedding  # type: ignore[no-any-return]

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents in ChromaDB.

        Args:
            documents_path: Path to the directory containing documents
        """
        docs_path = Path(documents_path)

        if not docs_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")

        # Validate extensions
        valid_extensions = {".txt", ".pdf", ".docx"}

        langchain_documents = []

        # Walk through directory
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in valid_extensions and file_path.is_file():
                try:
                    loader = create_loader(str(file_path))
                    doc = loader.load(str(file_path))

                    # Convert to LangChain document
                    lc_doc = LangChainDocument(
                        page_content=doc.content, metadata={"source": doc.source, **doc.metadata}
                    )
                    langchain_documents.append(lc_doc)
                    print(f"Loaded: {file_path.name}")

                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")

        if not langchain_documents:
            raise ValueError(f"No documents found in {documents_path}")

        documents = langchain_documents

        if not documents:
            raise ValueError(f"No documents found in {documents_path}")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks")

        # Prepare data for ChromaDB
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []
        chunk_embeddings: list[list[float]] = []

        total_chunks = len(chunks)

        # Process chunks in batches for efficiency
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk.page_content)

            # Store metadata
            metadata = {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i,
            }
            chunk_metadatas.append(metadata)

            # Get embedding
            embedding = self._get_embedding(chunk.page_content)
            chunk_embeddings.append(embedding)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
                self._report_progress(i + 1, total_chunks)

        # Add to ChromaDB collection
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas,  # type: ignore[arg-type]
            embeddings=chunk_embeddings,  # type: ignore[arg-type]
        )

        self._total_chunks = len(chunks)
        print(f"Successfully indexed {len(chunks)} chunks in ChromaDB")

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

        prompt = f"""Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Call OpenAI API
        model = self.config.llm_model or settings.openai_model
        completion_params: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        # Only add temperature for models that support it (not gpt-5-nano)
        if "nano" not in model.lower():
            completion_params["temperature"] = 0.0

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
