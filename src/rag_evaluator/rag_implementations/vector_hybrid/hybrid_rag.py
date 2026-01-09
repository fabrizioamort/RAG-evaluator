"""Hybrid search RAG implementation combining semantic and keyword search using Qdrant."""

import time
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.common.document_loaders import create_loader
from rag_evaluator.config import settings


class HybridSearchRAG(BaseRAG):
    """RAG implementation using Qdrant hybrid search (dense + sparse vectors)."""

    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
    ) -> None:
        """Initialize hybrid search RAG with Qdrant.

        Args:
            collection_name: Name of the Qdrant collection (defaults to settings)
            qdrant_url: Qdrant server URL (defaults to settings)
        """
        super().__init__("Hybrid Search (Semantic + Keyword)")

        # Initialize Qdrant client
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.client = QdrantClient(url=self.qdrant_url)

        # Initialize OpenAI client for dense embeddings
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout,
        )

        # Initialize FastEmbed for sparse embeddings (SPLADE)
        self.sparse_model = SparseTextEmbedding(
            model_name=settings.sparse_model_name,
        )

        # Text splitter for chunking (smaller chunks for hybrid search)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.hybrid_chunk_size,
            chunk_overlap=settings.hybrid_chunk_overlap,
            length_function=len,
        )

        # Collection setup
        self.collection_name = collection_name or settings.qdrant_collection_name
        self._ensure_collection()

        # Metrics tracking
        self._retrieval_times: list[float] = []
        self._total_chunks = 0

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
        response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
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

    def _process_batch(
        self, batch_chunks: list[LangChainDocument], start_index: int
    ) -> list[models.PointStruct]:
        """Process a batch of chunks: generate embeddings and create points.

        Args:
            batch_chunks: List of document chunks
            start_index: Starting ID for points

        Returns:
            List of Qdrant points
        """
        texts = [chunk.page_content for chunk in batch_chunks]

        # Batch dense embeddings (OpenAI)
        dense_response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        dense_embeddings = [data.embedding for data in dense_response.data]

        # Batch sparse embeddings (FastEmbed)
        # FastEmbed returns a generator, so we convert to list
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for i, (text, dense, sparse) in enumerate(zip(texts, dense_embeddings, sparse_embeddings)):
            idx = start_index + i
            chunk = batch_chunks[i]

            sparse_vec = models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            )

            point = models.PointStruct(
                id=idx,
                vector={
                    "dense": dense,
                    "sparse": sparse_vec,
                },
                payload={
                    "text": text,
                    "source": chunk.metadata.get("source", "unknown"),
                    "chunk_index": idx,
                },
            )
            points.append(point)

        return points

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents with both dense and sparse vectors.

        Args:
            documents_path: Path to the directory containing documents
        """
        docs_path = Path(documents_path)

        if not docs_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")

        # Validate extensions
        valid_extensions = {".txt", ".pdf", ".docx"}

        langchain_documents = []

        # Walk through directory and load documents
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in valid_extensions and file_path.is_file():
                try:
                    loader = create_loader(str(file_path))
                    doc = loader.load(str(file_path))

                    # Convert to LangChain document
                    lc_doc = LangChainDocument(
                        page_content=doc.content,
                        metadata={"source": doc.source, **doc.metadata},
                    )
                    langchain_documents.append(lc_doc)
                    print(f"Loaded: {file_path.name}")

                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")

        if not langchain_documents:
            raise ValueError(f"No documents found in {documents_path}")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(langchain_documents)

        print(f"Loaded {len(langchain_documents)} documents, split into {len(chunks)} chunks")

        # Delete existing points to avoid duplicates
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[]),
                ),
            )
        except UnexpectedResponse:
            # Collection might be empty, which is fine
            pass

        # Process in batches
        batch_size = 100
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            try:
                # Process batch (generate embeddings)
                points = self._process_batch(batch, start_index=i)

                # Upload batch to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

                print(
                    f"Processed and uploaded chunks {i + 1}-{min(i + batch_size, total_chunks)}/{total_chunks}"
                )

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                raise

        self._total_chunks = total_chunks
        print(f"Successfully indexed {total_chunks} chunks in Qdrant (hybrid mode)")

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using hybrid search (dense + sparse with RRF fusion).

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing answer, context, and metadata
        """
        # Start timing
        start_time = time.time()

        # Get both embeddings for the question
        dense_vec = self._get_dense_embedding(question)
        sparse_vec = self._get_sparse_embedding(question)

        # Hybrid search with RRF fusion
        # Prefetch more candidates from each search, then fuse
        prefetch_limit = top_k * 4  # Prefetch more for better fusion

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

        # Extract retrieved chunks
        retrieved_chunks = []
        retrieved_metadata = []

        for point in results.points:
            if point.payload:
                retrieved_chunks.append(point.payload.get("text", ""))
                retrieved_metadata.append(
                    {
                        "source": point.payload.get("source", "unknown"),
                        "chunk_index": point.payload.get("chunk_index", -1),
                        "score": point.score,
                    }
                )

        retrieval_time = time.time() - start_time
        self._retrieval_times.append(retrieval_time)

        # Generate answer using LLM with retrieved context
        context_text = "\n\n".join(
            [f"[{i + 1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)]
        )

        prompt = f"""Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

        # Call OpenAI API
        completion_params: dict[str, Any] = {
            "model": settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        # Only add temperature for models that support it (not gpt-5-nano)
        if "nano" not in settings.openai_model.lower():
            completion_params["temperature"] = 0.0

        response = self.openai_client.chat.completions.create(**completion_params)

        answer = response.choices[0].message.content or "No answer generated"

        return {
            "answer": answer,
            "context": retrieved_chunks,
            "metadata": {
                "retrieval_time": retrieval_time,
                "chunks_retrieved": len(retrieved_chunks),
                "sources": [meta.get("source", "unknown") for meta in retrieved_metadata],
                "fusion_method": "RRF",
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
            "chunk_size": settings.hybrid_chunk_size,
            "chunk_overlap": settings.hybrid_chunk_overlap,
        }
