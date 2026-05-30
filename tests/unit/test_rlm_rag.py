from pathlib import Path
from types import SimpleNamespace

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.rag_implementations.rlm_rag import (
    RLMConfig,
    RLMFilesystemRAG,
    rlm_config_from_rag_config,
)
from rag_evaluator.rag_implementations.rlm_rag.preparation import ManifestManager


def test_rlm_config_from_rag_config_defaults_orchestrator_to_llm_model() -> None:
    config = RAGConfig(
        name="RLM",
        llm_model="gpt-main",
        parameters={
            "orchestrator_model": None,
            "worker_model": "gpt-worker",
            "prepared_path": "ignored-by-rlm-config",
        },
    )

    rlm_config = rlm_config_from_rag_config(config)

    assert rlm_config.orchestrator_model == "gpt-main"
    assert rlm_config.worker_model == "gpt-worker"


def test_manifest_invalidates_when_preparation_config_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    (source / "doc.md").write_text("# Title\n\nContent", encoding="utf-8")

    manager = ManifestManager(prepared)
    manager.update(source, RLMConfig(use_llm_summaries=False, use_llm_topics=False))

    assert manager.is_valid(
        source,
        RLMConfig(use_llm_summaries=False, use_llm_topics=False),
    )
    assert not manager.is_valid(
        source,
        RLMConfig(use_llm_summaries=False, use_llm_topics=False, chunk_size=1200),
    )


def test_prepare_documents_uses_configured_prepared_path(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    prepared = tmp_path / "custom_prepared"
    source.mkdir()
    (source / "doc.md").write_text("# Doc\n\nThis is local content.", encoding="utf-8")

    rag = RLMFilesystemRAG(
        rlm_config=RLMConfig(use_llm_summaries=False, use_llm_topics=False),
        prepared_path=prepared,
    )

    metrics = rag.prepare_documents(str(source))

    assert Path(metrics["prepared_path"]) == prepared.resolve()
    assert (prepared / "_meta" / "catalog.json").exists()
    assert not (tmp_path / "raw_prepared").exists()


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Mocked answer from prepared context.")
                )
            ],
        )


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_simple_context_query_uses_prepared_corpus_and_configured_model(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    prepared = tmp_path / "prepared"
    source.mkdir()
    (source / "paris.md").write_text("Paris is the capital of France.", encoding="utf-8")

    rag = RLMFilesystemRAG(
        rlm_config=RLMConfig(
            orchestrator_model="gpt-test",
            use_llm_summaries=False,
            use_llm_topics=False,
        ),
        prepared_path=prepared,
    )
    rag.prepare_documents(str(source))

    fake_client = _FakeOpenAI()
    assert rag._simple_rag is not None
    rag._simple_rag._client = fake_client

    result = rag.query("What is the capital of France?")

    assert result["answer"] == "Mocked answer from prepared context."
    assert result["metadata"]["mode"] == "simple_context"
    assert result["metadata"]["security_mode"] == "lite"
    assert "Paris is the capital" in result["context"][0]
    assert fake_client.chat.completions.calls[0]["model"] == "gpt-test"
    assert result["metadata"]["token_usage"]["prompt_tokens"] == 11
