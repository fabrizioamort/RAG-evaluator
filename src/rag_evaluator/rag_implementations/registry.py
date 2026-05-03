"""Registry mapping RAG type keys to implementation classes and parameter schemas."""

import importlib
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG

# Maps RAG type key -> fully-qualified class path
_RAG_CLASS_PATHS: dict[str, str] = {
    "vector_semantic": "rag_evaluator.rag_implementations.vector_semantic.chroma_rag.ChromaSemanticRAG",
    "vector_hybrid": "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.HybridSearchRAG",
    "graph_rag": "rag_evaluator.rag_implementations.graph_rag.neo4j_rag.Neo4jGraphRAG",
    "filesystem_rag": "rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag.FilesystemRAG",
}

# Human-readable metadata for each type
RAG_TYPES: dict[str, dict[str, str]] = {
    "vector_semantic": {
        "name": "Vector Semantic Search",
        "description": "ChromaDB-based semantic vector search using embeddings",
    },
    "vector_hybrid": {
        "name": "Hybrid Search",
        "description": "Qdrant-based hybrid search combining dense and sparse vectors with RRF fusion",
    },
    "graph_rag": {
        "name": "Graph RAG",
        "description": "Neo4j-based graph RAG with entity relationships and vector search",
    },
    "filesystem_rag": {
        "name": "Filesystem RAG",
        "description": "LLM-guided agent that navigates a prepared filesystem structure",
    },
}

# Parameter schemas for each type (used by the platform UI)
RAG_TYPE_PARAMETERS: dict[str, dict[str, Any]] = {
    "vector_semantic": {
        "properties": {
            "chunk_size": {"type": "integer", "default": 1000, "description": "Size of text chunks"},
            "chunk_overlap": {"type": "integer", "default": 200, "description": "Overlap between chunks"},
            "collection_name": {"type": "string", "default": "rag_documents", "description": "ChromaDB collection name"},
        },
    },
    "vector_hybrid": {
        "properties": {
            "chunk_size": {"type": "integer", "default": 500, "description": "Size of text chunks"},
            "chunk_overlap": {"type": "integer", "default": 50, "description": "Overlap between chunks"},
            "collection_name": {"type": "string", "default": "hybrid_rag", "description": "Qdrant collection name"},
        },
    },
    "graph_rag": {
        "properties": {
            "vector_index_name": {"type": "string", "default": "chunk_embeddings", "description": "Neo4j vector index name"},
        },
    },
    "filesystem_rag": {
        "properties": {
            "word_threshold": {"type": "integer", "default": 1000, "description": "Word count threshold"},
            "max_iterations": {"type": "integer", "default": 10, "description": "Max ReAct loop iterations"},
            "max_tool_calls": {"type": "integer", "default": 20, "description": "Max tool calls per query"},
            "max_file_reads": {"type": "integer", "default": 10, "description": "Max file reads per query"},
        },
    },
}


def get_rag_class(rag_type: str) -> type[BaseRAG]:
    """Return the RAG implementation class for the given type key.

    Raises:
        ValueError: If rag_type is not in the registry.
        ImportError: If the class cannot be imported.
    """
    if rag_type not in _RAG_CLASS_PATHS:
        raise ValueError(f"Unknown RAG type: {rag_type}. Supported: {list(_RAG_CLASS_PATHS)}")
    module_path, class_name = _RAG_CLASS_PATHS[rag_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[return-value]
