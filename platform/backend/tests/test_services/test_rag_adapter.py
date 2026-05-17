"""Tests for Neo4j parameter resolution in RAGAdapterService."""

from types import SimpleNamespace
from uuid import uuid4

import app.services.rag_adapter as rag_adapter_module
from app.services.rag_adapter import RAGAdapterService


def _graph_config_model(parameters: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        project_id=uuid4(),
        name="Graph Config",
        rag_type="graph_rag",
        parameters=parameters,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_base_url=None,
    )


def test_create_rag_from_config_graph_uses_env_for_blank_connection_values(
    monkeypatch,
) -> None:
    """Blank or whitespace Neo4j values should resolve from backend settings."""
    service = RAGAdapterService()
    captured_kwargs: dict[str, object] = {}

    class FakeGraphRAG:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(service, "_get_rag_class", lambda _: FakeGraphRAG)
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_URI", "bolt://env:7687")
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_USERNAME", "env-user")
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_PASSWORD", "env-pass")

    model = _graph_config_model(
        {
            "neo4j_uri": "  ",
            "neo4j_username": "",
            "neo4j_password": "   ",
        }
    )

    service.create_rag_from_config(model)

    assert captured_kwargs["neo4j_uri"] == "bolt://env:7687"
    assert captured_kwargs["neo4j_username"] == "env-user"
    assert captured_kwargs["neo4j_password"] == "env-pass"


def test_create_rag_for_index_graph_uses_env_for_blank_connection_values(
    monkeypatch,
) -> None:
    """Index snapshot blank Neo4j values should resolve from backend settings."""
    service = RAGAdapterService()
    captured_kwargs: dict[str, object] = {}

    class FakeGraphRAG:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(service, "_get_rag_class", lambda _: FakeGraphRAG)
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_URI", "bolt://env:7687")
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_USERNAME", "env-user")
    monkeypatch.setattr(rag_adapter_module.settings, "NEO4J_PASSWORD", "env-pass")

    index = SimpleNamespace(
        id=uuid4(),
        name="Index 1",
        physical_id="idx_test_123",
        config_snapshot={
            "rag_type": "graph_rag",
            "parameters": {
                "neo4j_uri": "",
                "neo4j_username": "  ",
                "neo4j_password": "",
            },
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_base_url": None,
        },
    )

    service.create_rag_for_index(index)

    assert captured_kwargs["neo4j_uri"] == "bolt://env:7687"
    assert captured_kwargs["neo4j_username"] == "env-user"
    assert captured_kwargs["neo4j_password"] == "env-pass"

