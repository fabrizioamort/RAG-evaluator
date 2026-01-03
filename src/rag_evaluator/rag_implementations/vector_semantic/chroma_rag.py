"""ChromaDB-based semantic search RAG implementation."""

import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from rag_evaluator.common.base_rag import BaseRAG
from rag_evaluator.common.document_loaders import create_loader
from rag_evaluator.config import settings


class ChromaSemanticRAG(BaseRAG):
    """RAG implementation using ChromaDB for semantic vector search."""

    def __init__(
        self, collection_name: str = "rag_documents", persist_directory: str | None = None
    ) -> None:
        """Initialize ChromaDB semantic RAG.

        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_directory: Optional custom persistence directory (defaults to settings)
        """
        super().__init__("ChromaDB Semantic Search")

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
            api_key=settings.openai_api_key, timeout=settings.openai_timeout
        )

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Track metrics
        self._retrieval_times: list[float] = []
        self._total_chunks = 0

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(model=settings.embedding_model, input=text)
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
                        page_content=doc.content,
                        metadata={
                            "source": doc.source,
                            **doc.metadata
                        }
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

        # Add to ChromaDB collection
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas,  # type: ignore[arg-type]
            embeddings=chunk_embeddings,  # type: ignore[arg-type]
        )

        self._total_chunks = len(chunks)
        print(f"Successfully indexed {len(chunks)} chunks in ChromaDB")

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using semantic similarity search.

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing answer, context, and metadata
        """
        # Start timing
        start_time = time.time()

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
        # Note: Some models like gpt-5-nano don't support temperature parameter
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
        }
