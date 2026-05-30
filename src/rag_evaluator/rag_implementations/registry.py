"""Registry mapping RAG type keys to implementation classes and parameter schemas."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from copy import deepcopy
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

# Top-level fields that are frozen into an index and cannot be overridden when
# querying an existing index.
BUILD_TIME_TOP_LEVEL_FIELDS = {
    "rag_type",
    "llm_provider",
    "llm_base_url",
    "embedding_model",
    "storage_path",
}

# Parameter schemas for each type (used by the platform UI and validation).
# Every exposed parameter must declare a lifecycle phase:
# - build: affects stored artifacts or physical storage and is frozen in indexes
# - query: can be changed when querying/evaluating an existing ready index
RAG_TYPE_PARAMETERS: dict[str, dict[str, Any]] = {
    "vector_semantic": {
        "properties": {
            "chunk_size": {
                "type": "integer",
                "phase": "build",
                "default": 1000,
                "minimum": 1,
                "description": "Size of text chunks",
            },
            "chunk_overlap": {
                "type": "integer",
                "phase": "build",
                "default": 200,
                "minimum": 0,
                "description": "Overlap between chunks",
            },
            "collection_name": {
                "type": "string",
                "phase": "build",
                "default": "rag_documents",
                "description": "ChromaDB collection name",
                "platform_managed": True,
            },
            "persist_directory": {
                "type": "string",
                "phase": "build",
                "description": "ChromaDB persistence directory",
                "platform_managed": True,
            },
        },
    },
    "vector_hybrid": {
        "properties": {
            "chunk_size": {
                "type": "integer",
                "phase": "build",
                "default": 500,
                "minimum": 1,
                "description": "Size of text chunks",
            },
            "chunk_overlap": {
                "type": "integer",
                "phase": "build",
                "default": 50,
                "minimum": 0,
                "description": "Overlap between chunks",
            },
            "collection_name": {
                "type": "string",
                "phase": "build",
                "default": "hybrid_rag",
                "description": "Qdrant collection name",
                "platform_managed": True,
            },
            "qdrant_url": {
                "type": "string",
                "phase": "build",
                "description": "Qdrant server URL",
                "platform_managed": True,
            },
            "sparse_model_name": {
                "type": "string",
                "phase": "build",
                "default": "prithivida/Splade_PP_en_v1",
                "description": "Sparse embedding model used for hybrid indexing",
            },
        },
    },
    "graph_rag": {
        "properties": {
            "neo4j_uri": {
                "type": "string",
                "phase": "build",
                "description": "Neo4j connection URI",
                "platform_managed": True,
            },
            "neo4j_username": {
                "type": "string",
                "phase": "build",
                "description": "Neo4j username",
                "platform_managed": True,
            },
            "neo4j_password": {
                "type": "string",
                "phase": "build",
                "description": "Neo4j password",
                "platform_managed": True,
            },
            "vector_index_name": {
                "type": "string",
                "phase": "build",
                "default": "chunk_embeddings",
                "description": "Neo4j vector index name",
            },
            "extraction_model": {
                "type": "string",
                "phase": "build",
                "default": "RAG config llm_model",
                "description": "Model used for graph extraction during indexing",
            },
        },
    },
    "filesystem_rag": {
        "properties": {
            "prepared_path": {
                "type": "string",
                "phase": "build",
                "description": "Prepared filesystem output path",
                "platform_managed": True,
            },
            "word_threshold": {
                "type": "integer",
                "phase": "build",
                "default": 1000,
                "minimum": 1,
                "description": "Word count threshold",
            },
            "max_iterations": {
                "type": "integer",
                "phase": "query",
                "default": 10,
                "minimum": 1,
                "description": "Max ReAct loop iterations",
            },
            "max_tool_calls": {
                "type": "integer",
                "phase": "query",
                "default": 20,
                "minimum": 1,
                "description": "Max tool calls per query",
            },
            "max_file_reads": {
                "type": "integer",
                "phase": "query",
                "default": 10,
                "minimum": 1,
                "description": "Max file reads per query",
            },
        },
    },
    "rlm_rag": {
        "properties": {
            "security_mode": {
                "type": "string",
                "phase": "query",
                "default": "lite",
                "enum": ["lite", "full"],
                "description": "Security mode: lite for trusted in-process execution, full for subprocess isolation",
            },
            "orchestrator_model": {
                "type": "string",
                "phase": "query",
                "default": "RAG config llm_model",
                "description": "Model used for main reasoning and code generation",
            },
            "worker_model": {
                "type": "string",
                "phase": "build",
                "default": "gpt-5-nano",
                "description": "Model used for summaries, topics, and sub-LLM calls",
            },
            "max_repl_steps": {
                "type": "integer",
                "phase": "query",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum Python exploration steps per query",
            },
            "repl_timeout": {
                "type": "number",
                "phase": "query",
                "default": 5.0,
                "minimum": 0.1,
                "maximum": 60,
                "description": "Timeout in seconds for each REPL step",
            },
            "max_file_reads": {
                "type": "integer",
                "phase": "query",
                "default": 12,
                "minimum": 1,
                "description": "Maximum file reads per query",
            },
            "max_read_bytes": {
                "type": "integer",
                "phase": "query",
                "default": 50000,
                "minimum": 1000,
                "description": "Maximum bytes returned by a file read",
            },
            "max_read_lines": {
                "type": "integer",
                "phase": "query",
                "default": 1000,
                "minimum": 1,
                "description": "Maximum lines returned by a file read",
            },
            "max_sub_calls": {
                "type": "integer",
                "phase": "query",
                "default": 8,
                "minimum": 0,
                "description": "Maximum recursive worker-model calls per query",
            },
            "max_recursion_depth": {
                "type": "integer",
                "phase": "query",
                "default": 2,
                "minimum": 0,
                "description": "Maximum nested sub-LLM call depth",
            },
            "small_corpus_threshold": {
                "type": "integer",
                "phase": "query",
                "default": 10,
                "minimum": 1,
                "description": "Use simple-context fallback at or below this document count",
            },
            "chunk_size": {
                "type": "integer",
                "phase": "build",
                "default": 1000,
                "minimum": 1,
                "description": "Preparation chunk size",
            },
            "chunk_overlap": {
                "type": "integer",
                "phase": "build",
                "default": 200,
                "minimum": 0,
                "description": "Preparation chunk overlap",
            },
            "use_llm_summaries": {
                "type": "boolean",
                "phase": "build",
                "default": True,
                "description": "Generate LLM summaries during preparation",
            },
            "use_llm_topics": {
                "type": "boolean",
                "phase": "build",
                "default": True,
                "description": "Extract LLM topics during preparation",
            },
            "max_topics_per_doc": {
                "type": "integer",
                "phase": "build",
                "default": 5,
                "minimum": 1,
                "description": "Maximum topics extracted per document",
            },
            "prepared_path": {
                "type": "string",
                "phase": "build",
                "description": "Prepared RLM filesystem path",
                "platform_managed": True,
            },
        },
    },
}


def get_parameter_schema(rag_type: str) -> dict[str, Any]:
    """Return the parameter schema for a RAG type."""
    if rag_type not in RAG_TYPE_PARAMETERS:
        raise ValueError(f"Unknown RAG type: {rag_type}")
    return deepcopy(RAG_TYPE_PARAMETERS[rag_type])


def _properties(rag_type: str) -> dict[str, dict[str, Any]]:
    return get_parameter_schema(rag_type).get("properties", {})


def build_param_names(rag_type: str) -> set[str]:
    """Return parameter names frozen at index build time."""
    return {name for name, schema in _properties(rag_type).items() if schema["phase"] == "build"}


def query_param_names(rag_type: str) -> set[str]:
    """Return parameter names that can be overridden at query time."""
    return {name for name, schema in _properties(rag_type).items() if schema["phase"] == "query"}


def split_parameters(
    rag_type: str, parameters: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a RAG parameter dict into build-time and query-time partitions."""
    params = dict(parameters or {})
    properties = _properties(rag_type)
    build_params: dict[str, Any] = {}
    query_params: dict[str, Any] = {}

    for name, value in params.items():
        schema = properties.get(name)
        if not schema:
            # Preserve unknown legacy values in the build partition so older
            # snapshots remain reproducible, but do not make them query-overridable.
            build_params[name] = value
        elif schema["phase"] == "query":
            query_params[name] = value
        else:
            build_params[name] = value

    return build_params, query_params


def _as_override_dict(overrides: Any) -> dict[str, Any]:
    if overrides is None:
        return {}
    if hasattr(overrides, "model_dump"):
        return overrides.model_dump(exclude_none=True)
    if isinstance(overrides, Mapping):
        return dict(overrides)
    raise ValueError("query_overrides must be an object")


def _validate_param_value(name: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected_type = schema.get("type")

    if expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"Query override `{name}` must be an integer.")
    if expected_type == "number" and not isinstance(value, int | float):
        raise ValueError(f"Query override `{name}` must be a number.")
    if expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"Query override `{name}` must be a string.")
    if expected_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"Query override `{name}` must be a boolean.")

    enum = schema.get("enum")
    if enum and value not in enum:
        raise ValueError(f"Query override `{name}` must be one of: {', '.join(enum)}.")

    minimum = schema.get("minimum")
    if minimum is not None and value < minimum:
        raise ValueError(f"Query override `{name}` must be >= {minimum}.")

    maximum = schema.get("maximum")
    if maximum is not None and value > maximum:
        raise ValueError(f"Query override `{name}` must be <= {maximum}.")


def _build_override_error(name: str) -> ValueError:
    return ValueError(
        f"Cannot override build-time parameter `{name}` for an existing index. "
        "Changing it requires creating a new index."
    )


def validate_query_overrides(rag_type: str, overrides: Any) -> dict[str, Any]:
    """Validate and normalize query overrides for an existing index.

    Returns a dict with optional ``llm_model`` and ``top_k`` keys plus a
    ``parameters`` dict containing only allowed query-phase RAG parameters.
    """
    data = _as_override_dict(overrides)
    normalized: dict[str, Any] = {}
    allowed_top_level = {"llm_model", "top_k", "parameters"}

    for key in data:
        if key in allowed_top_level:
            continue
        if key in BUILD_TIME_TOP_LEVEL_FIELDS:
            raise _build_override_error(key)
        raise ValueError(f"Unknown query override `{key}`.")

    llm_model = data.get("llm_model")
    if llm_model is not None:
        if not isinstance(llm_model, str) or not llm_model.strip():
            raise ValueError("Query override `llm_model` must be a non-empty string.")
        normalized["llm_model"] = llm_model.strip()

    top_k = data.get("top_k")
    if top_k is not None:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Query override `top_k` must be an integer >= 1.")
        normalized["top_k"] = top_k

    raw_parameters = data.get("parameters") or {}
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("Query override `parameters` must be an object.")

    properties = _properties(rag_type)
    parameters: dict[str, Any] = {}
    for name, value in raw_parameters.items():
        schema = properties.get(name)
        if not schema:
            if name in BUILD_TIME_TOP_LEVEL_FIELDS:
                raise _build_override_error(name)
            raise ValueError(f"Unknown RAG parameter override `{name}` for `{rag_type}`.")
        if schema["phase"] != "query":
            raise _build_override_error(name)
        _validate_param_value(name, value, schema)
        parameters[name] = value

    normalized["parameters"] = parameters
    return normalized


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
