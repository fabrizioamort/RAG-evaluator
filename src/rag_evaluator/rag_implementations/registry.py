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
    "rlm_rag": "rag_evaluator.rag_implementations.rlm_rag.rlm_rag.RLMFilesystemRAG",
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
    "rlm_rag": {
        "name": "RLM-RAG",
        "description": "Recursive language-model RAG that explores a prepared filesystem with Python tools",
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
    "rlm_rag": {
        "properties": {
            "security_mode": {
                "type": "string",
                "default": "lite",
                "enum": ["lite", "full"],
                "description": "Security mode: lite for trusted in-process execution, full for subprocess isolation",
            },
            "orchestrator_model": {
                "type": "string",
                "default": "RAG config llm_model",
                "description": "Model used for main reasoning and code generation",
            },
            "worker_model": {
                "type": "string",
                "default": "gpt-5-nano",
                "description": "Model used for summaries, topics, and sub-LLM calls",
            },
            "max_repl_steps": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum Python exploration steps per query",
            },
            "repl_timeout": {
                "type": "number",
                "default": 5.0,
                "minimum": 0.1,
                "maximum": 60,
                "description": "Timeout in seconds for each REPL step",
            },
            "max_file_reads": {
                "type": "integer",
                "default": 12,
                "description": "Maximum file reads per query",
            },
            "max_read_bytes": {
                "type": "integer",
                "default": 50000,
                "description": "Maximum bytes returned by a file read",
            },
            "max_read_lines": {
                "type": "integer",
                "default": 1000,
                "description": "Maximum lines returned by a file read",
            },
            "max_sub_calls": {
                "type": "integer",
                "default": 8,
                "description": "Maximum recursive worker-model calls per query",
            },
            "max_recursion_depth": {
                "type": "integer",
                "default": 2,
                "description": "Maximum nested sub-LLM call depth",
            },
            "small_corpus_threshold": {
                "type": "integer",
                "default": 10,
                "description": "Use simple-context fallback at or below this document count",
            },
            "chunk_size": {
                "type": "integer",
                "default": 1000,
                "description": "Preparation chunk size",
            },
            "chunk_overlap": {
                "type": "integer",
                "default": 200,
                "description": "Preparation chunk overlap",
            },
            "use_llm_summaries": {
                "type": "boolean",
                "default": True,
                "description": "Generate LLM summaries during preparation",
            },
            "use_llm_topics": {
                "type": "boolean",
                "default": True,
                "description": "Extract LLM topics during preparation",
            },
            "max_topics_per_doc": {
                "type": "integer",
                "default": 5,
                "description": "Maximum topics extracted per document",
            },
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
