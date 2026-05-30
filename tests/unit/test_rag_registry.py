import pytest

from rag_evaluator.rag_implementations.registry import (
    RAG_TYPES,
    build_param_names,
    get_parameter_schema,
    get_rag_class,
    query_param_names,
    validate_query_overrides,
)


def test_all_types_present():
    assert set(RAG_TYPES) == {
        "vector_semantic",
        "vector_hybrid",
        "graph_rag",
        "filesystem_rag",
        "rlm_rag",
    }


def test_get_rag_class_returns_type():
    from rag_evaluator.common.base_rag import BaseRAG
    cls = get_rag_class("vector_semantic")
    assert issubclass(cls, BaseRAG)


def test_get_rag_class_returns_rlm_type():
    from rag_evaluator.common.base_rag import BaseRAG

    cls = get_rag_class("rlm_rag")
    assert issubclass(cls, BaseRAG)


def test_get_rag_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown RAG type"):
        get_rag_class("nonexistent")


def test_every_exposed_parameter_has_phase():
    for rag_type in RAG_TYPES:
        schema = get_parameter_schema(rag_type)
        for name, definition in schema["properties"].items():
            assert definition.get("phase") in {"build", "query"}, (rag_type, name)


def test_build_query_partitions_are_exhaustive_and_disjoint():
    for rag_type in RAG_TYPES:
        params = set(get_parameter_schema(rag_type)["properties"])
        build = build_param_names(rag_type)
        query = query_param_names(rag_type)

        assert build | query == params
        assert build.isdisjoint(query)


def test_validate_query_overrides_accepts_query_parameter():
    overrides = validate_query_overrides(
        "rlm_rag",
        {
            "llm_model": "gpt-5-mini",
            "top_k": 8,
            "parameters": {"orchestrator_model": "gpt-5-mini", "max_repl_steps": 20},
        },
    )

    assert overrides["llm_model"] == "gpt-5-mini"
    assert overrides["top_k"] == 8
    assert overrides["parameters"]["max_repl_steps"] == 20


def test_validate_query_overrides_rejects_build_parameter():
    with pytest.raises(ValueError, match="Cannot override build-time parameter `chunk_size`"):
        validate_query_overrides("rlm_rag", {"parameters": {"chunk_size": 1200}})


def test_validate_query_overrides_rejects_top_level_embedding_model():
    with pytest.raises(ValueError, match="Cannot override build-time parameter `embedding_model`"):
        validate_query_overrides("vector_semantic", {"embedding_model": "text-embedding-3-large"})
