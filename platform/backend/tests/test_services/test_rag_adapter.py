"""Tests for Neo4j parameter resolution in RAGAdapterService."""

from pathlib import Path
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


def _rlm_config_model(parameters: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        project_id=uuid4(),
        name="RLM Config",
        rag_type="rlm_rag",
        parameters=parameters,
        llm_provider="openai",
        llm_model="gpt-main",
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


def test_create_rag_from_config_rlm_builds_config_and_prepared_path(
    monkeypatch,
    tmp_path,
) -> None:
    """RLM configs should receive typed RLMConfig and platform-managed path."""
    service = RAGAdapterService()
    captured_kwargs: dict[str, object] = {}

    class FakeRLMRAG:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    index_path = tmp_path / "idx_123"
    monkeypatch.setattr(service, "_get_rag_class", lambda _: FakeRLMRAG)

    model = _rlm_config_model(
        {
            "security_mode": "full",
            "worker_model": "gpt-worker",
            "orchestrator_model": None,
            "max_repl_steps": 9,
        }
    )

    service.create_rag_from_config(model, index_path=str(index_path))

    rlm_config = captured_kwargs["rlm_config"]
    assert rlm_config.security_mode == "full"
    assert rlm_config.orchestrator_model == "gpt-main"
    assert rlm_config.worker_model == "gpt-worker"
    assert rlm_config.max_repl_steps == 9
    assert captured_kwargs["prepared_path"] == str(index_path / "rlm_rag")


def test_create_rag_for_index_rlm_uses_physical_id_storage(
    monkeypatch,
) -> None:
    """RLM index snapshots should be isolated under storage/indexes/<physical_id>/rlm_rag."""
    service = RAGAdapterService()
    captured_kwargs: dict[str, object] = {}

    class FakeRLMRAG:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(service, "_get_rag_class", lambda _: FakeRLMRAG)
    monkeypatch.setattr(rag_adapter_module.settings, "STORAGE_PATH", "storage")

    index = SimpleNamespace(
        id=uuid4(),
        name="Index 1",
        physical_id="idx_rlm_123",
        config_snapshot={
            "rag_type": "rlm_rag",
            "parameters": {
                "worker_model": "gpt-worker",
                "max_file_reads": 6,
            },
            "llm_provider": "openai",
            "llm_model": "gpt-main",
            "llm_base_url": None,
        },
    )

    service.create_rag_for_index(index)

    rlm_config = captured_kwargs["rlm_config"]
    assert rlm_config.orchestrator_model == "gpt-main"
    assert rlm_config.worker_model == "gpt-worker"
    assert rlm_config.max_file_reads == 6
    assert captured_kwargs["prepared_path"] == str(
        Path("storage") / "indexes" / "idx_rlm_123" / "rlm_rag"
    )


def test_build_effective_config_applies_query_overrides() -> None:
    service = RAGAdapterService()
    index = SimpleNamespace(
        id=uuid4(),
        name="Index 1",
        physical_id="idx_effective",
        embedding_model="text-embedding-3-small",
        config_snapshot={
            "rag_type": "rlm_rag",
            "parameters": {
                "worker_model": "gpt-worker",
                "orchestrator_model": "gpt-main",
                "max_repl_steps": 15,
                "chunk_size": 1000,
            },
            "llm_provider": "openai",
            "llm_model": "gpt-main",
            "llm_base_url": None,
            "embedding_model": "text-embedding-3-small",
        },
    )

    effective = service.build_effective_config(
        index,
        {
            "llm_model": "gpt-override",
            "top_k": 9,
            "parameters": {"orchestrator_model": "gpt-override", "max_repl_steps": 20},
        },
    )

    assert effective.top_k == 9
    assert effective.generation_model == "gpt-override"
    assert effective.effective_config_snapshot["llm_model"] == "gpt-override"
    assert effective.effective_config_snapshot["parameters"]["worker_model"] == "gpt-worker"
    assert effective.effective_config_snapshot["parameters"]["max_repl_steps"] == 20


def test_build_effective_config_rejects_build_override() -> None:
    service = RAGAdapterService()
    index = SimpleNamespace(
        id=uuid4(),
        name="Index 1",
        physical_id="idx_effective",
        embedding_model="text-embedding-3-small",
        config_snapshot={
            "rag_type": "vector_semantic",
            "parameters": {"chunk_size": 1000},
            "llm_provider": "openai",
            "llm_model": "gpt-main",
            "llm_base_url": None,
            "embedding_model": "text-embedding-3-small",
        },
    )

    import pytest

    with pytest.raises(ValueError, match="Cannot override build-time parameter `chunk_size`"):
        service.build_effective_config(index, {"parameters": {"chunk_size": 500}})


def test_load_rag_for_index_query_calls_load_not_prepare(monkeypatch) -> None:
    service = RAGAdapterService()
    calls: dict[str, int] = {"load": 0, "prepare": 0}

    class FakeRAG:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def load_index(self) -> None:
            calls["load"] += 1

        def prepare_documents(self, documents_path: str) -> None:
            calls["prepare"] += 1

    monkeypatch.setattr(service, "_get_rag_class", lambda _: FakeRAG)
    index = SimpleNamespace(
        id=uuid4(),
        name="Index 1",
        physical_id="idx_load",
        embedding_model="text-embedding-3-small",
        config_snapshot={
            "rag_type": "vector_semantic",
            "parameters": {"chunk_size": 1000},
            "llm_provider": "openai",
            "llm_model": "gpt-main",
            "llm_base_url": None,
            "embedding_model": "text-embedding-3-small",
        },
    )

    service.load_rag_for_index_query(index, {"top_k": 7})

    assert calls == {"load": 1, "prepare": 0}
