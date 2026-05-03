from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_rag_class


def test_all_types_present():
    assert set(RAG_TYPES) == {"vector_semantic", "vector_hybrid", "graph_rag", "filesystem_rag"}


def test_get_rag_class_returns_type():
    from rag_evaluator.common.base_rag import BaseRAG
    cls = get_rag_class("vector_semantic")
    assert issubclass(cls, BaseRAG)


def test_get_rag_class_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown RAG type"):
        get_rag_class("nonexistent")
