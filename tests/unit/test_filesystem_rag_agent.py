"""Tests for filesystem RAG query routing and prefetch behavior."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.agent import agent as agent_module
from rag_evaluator.rag_implementations.filesystem_rag.agent.agent import (
    AgentResponse,
    FilesystemRAGAgent,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
    format_tool_result,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.router import (
    QueryRouter,
    SearchMode,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import FilesystemRAGTools
from rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag import FilesystemRAG

LEGAL_VARE_QUESTION = (
    "Josh is 21 years old. He witnessed a murder that was clearly perpetrated by the "
    "accused. The prosecution, however, needs to determine whether the accused committed "
    "the acts voluntarily, thus satisfying the elements of intentional or reckless murder. "
    "In court, the judge plays a recording of Josh giving his eyewitness testimony. "
    "What is this evidentiary process called?"
)

SALLY_VIEW_QUESTION = (
    "Sally is accused of cultivating narcotic plants in her backyard. One of the "
    'elements of this charge is that "the accused intentionally cultivated or '
    'attempted to cultivate a particular substance." To establish whether this '
    "is the case, the judge believes it would be valuable to visit Sally's "
    "backyard and have the jury examine it for themselves. What is the name of "
    "the legal procedure whereby the court travels to a location relevant to the "
    "charge?"
)


def _write_prepared_fixture(root: Path) -> None:
    """Create a small prepared filesystem fixture."""
    (root / "_meta").mkdir(parents=True)
    (root / "_index" / "topics").mkdir(parents=True)
    (root / "_index" / "entities").mkdir(parents=True)
    (root / "_index" / "questions").mkdir(parents=True)
    (root / "_summaries").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)

    (root / "_meta" / "corpus_overview.md").write_text("Legal bench subset", encoding="utf-8")
    (root / "_meta" / "navigation_guide.md").write_text("Use indexes", encoding="utf-8")
    (root / "_index" / "topics" / "_topic_map.md").write_text(
        "General legal topics", encoding="utf-8"
    )
    (root / "_index" / "entities" / "_entity_registry.md").write_text(
        "No entities", encoding="utf-8"
    )

    (root / "_index" / "questions" / "question_seeds.md").write_text(
        "\n".join(
            [
                "# Question Seeds",
                '- "What is Use of Circumstantial Evidence?" -> doc_013',
                '- "What is The VARE Procedure?" -> doc_029',
                '- "What is 2.1 Views?" -> doc_045',
                '- "What is view?" -> doc_045',
                (
                    "- \"What does the section on a prosecution witness's "
                    "evidence-in-chief by way of an audio or audiovisual recording "
                    'cover?" -> doc_029'
                ),
            ]
        ),
        encoding="utf-8",
    )
    (root / "_summaries" / "doc_013_summary.md").write_text(
        "# Summary: Use of Circumstantial Evidence\n"
        "The prosecution may use circumstantial evidence to prove elements.",
        encoding="utf-8",
    )
    (root / "_summaries" / "doc_029_summary.md").write_text(
        "# Summary: The VARE Procedure\n"
        "A prosecution witness's evidence-in-chief may be given by way of an audio "
        "or audiovisual recording in which the witness answers questions put by a "
        'prescribed person. The recording is called a "VARE".',
        encoding="utf-8",
    )
    (root / "_summaries" / "doc_045_summary.md").write_text(
        "# Summary: 2.1 Views\n"
        'Under s 53, the court may order a "demonstration, experiment or '
        'inspection" (collectively, a "view"). An inspection involves the court '
        "travelling to view a location or an object that could not be brought "
        "into the courtroom.",
        encoding="utf-8",
    )
    (root / "documents" / "doc_045.md").write_text(
        "# 2.1 Views\n\n"
        "## What is a view?\n\n"
        "## 1. Under s 53 of the Evidence Act 2008, the court may order a "
        '"demonstration, experiment or inspection" (collectively, a "view").\n\n'
        "## 2. These terms are not defined in the Evidence Act 2008. Based on "
        "the common law and the conventional meaning of the terms:\n\n"
        "- An inspection involves the court travelling to view a location or an "
        "object that could not be brought into the courtroom;\n\n"
        "- A demonstration builds on an inspection by allowing a witness to "
        "explain the incident in question.\n",
        encoding="utf-8",
    )


def test_called_process_question_routes_as_known_item() -> None:
    result = QueryRouter().route(LEGAL_VARE_QUESTION)

    assert result.mode == SearchMode.KNOWN_ITEM
    assert result.confidence >= 0.9


def test_prefetch_promotes_vare_candidate_for_recorded_witness_question() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(str(prepared_path), client=object())  # type: ignore[arg-type]

        prefetch = agent._build_prefetch_context(LEGAL_VARE_QUESTION)

        assert prefetch["candidates"][0]["doc_id"] == "doc_029"
        assert any("VARE" in chunk for chunk in prefetch["chunks"])
        assert any(
            "recording" in candidate["matched_terms"] for candidate in prefetch["candidates"]
        )


def test_prefetch_includes_full_document_excerpt_for_view_question() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(str(prepared_path), client=object())  # type: ignore[arg-type]

        prefetch = agent._build_prefetch_context(SALLY_VIEW_QUESTION)

        assert prefetch["candidates"][0]["doc_id"] == "doc_045"
        assert "documents/doc_045.md" in prefetch["sources"]
        assert any(
            "Candidate Full Text Excerpt: doc_045" in chunk and 'collectively, a "view"' in chunk
            for chunk in prefetch["chunks"]
        )


def test_agent_retries_transient_gateway_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(agent_module, "_LLM_RETRY_BASE_DELAY_SECONDS", 0)

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: Any) -> object:
            self.calls += 1
            if self.calls == 1:
                raise Exception("BadGatewayError: Upstream error from Morph: undefined")
            return recovered_response

    class FakeChat:
        def __init__(self, completions: FakeCompletions) -> None:
            self.completions = completions

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = FakeChat(completions)

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        recovered_response = object()
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent._call_llm([{"role": "user", "content": "What is VARE?"}])

        assert response is recovered_response
        assert completions.calls == 2


def test_unusable_answer_reason_classification() -> None:
    assert agent_module.unusable_answer_reason("") == "empty"
    assert agent_module.unusable_answer_reason("   \n") == "empty"
    assert agent_module.unusable_answer_reason("抱歉，我无法协助处理该请求。") == "non_english"
    assert (
        agent_module.unusable_answer_reason("I'm sorry, but I can't assist with that.") == "refusal"
    )
    assert agent_module.unusable_answer_reason("Yes.") is None
    assert (
        agent_module.unusable_answer_reason(
            "The procedure is called a view under s 53 of the Evidence Act 2008."
        )
        is None
    )


def test_agent_retries_unusable_final_answer() -> None:
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content
            self.tool_calls = None

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> object:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse("抱歉，我无法协助处理该请求。")
            assert "tools" not in kwargs
            return FakeResponse("The recording procedure is called a VARE.")

    class FakeChat:
        def __init__(self, completions: FakeCompletions) -> None:
            self.completions = completions

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = FakeChat(completions)

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "The recording procedure is called a VARE."
        assert response.metadata["answer_retries"] == 1
        assert response.metadata["answer_retry_reason"] == "non_english"
        assert completions.calls == 2


def test_system_prompt_contains_answer_contract_and_sensitive_framing() -> None:
    from rag_evaluator.rag_implementations.filesystem_rag.agent.prompts import (
        format_system_prompt,
    )

    prompt = format_system_prompt(strategy_hint="", initial_context="ctx")

    assert "Answer Contract" in prompt
    assert "first sentence must directly state the conclusion" in prompt.lower()
    assert "sexual offences" in prompt
    assert "Never refuse" in prompt


def test_format_tool_result_read_file_is_plain_text_with_large_budget() -> None:
    content = "The object's condition may have changed since the offence.\n" * 100
    result = {"content": content, "total_lines": 100, "is_partial": False, "headers": []}

    formatted = format_tool_result("read_file", result)

    assert formatted.startswith("[full read; file has 100 lines]")
    assert "condition may have changed" in formatted
    assert "\\n" not in formatted
    assert len(formatted) > 2000


def test_format_tool_result_read_file_truncates_with_reread_hint() -> None:
    result = {"content": "x" * 30_000, "total_lines": 400, "is_partial": True, "headers": []}

    formatted = format_tool_result("read_file", result)

    assert len(formatted) < 30_000
    assert formatted.startswith("[partial read; file has 400 lines]")
    assert "start_line/end_line" in formatted


def test_format_tool_result_navigation_tools_keep_tight_budget() -> None:
    entries = [{"name": f"doc_{i:03d}.md", "type": "file", "size": 1234} for i in range(200)]

    formatted = format_tool_result("list_directory", entries)

    assert formatted.endswith("[truncated]")
    assert len(formatted) <= 2000 + len("\n... [truncated]")


def test_read_file_rejects_oversized_full_read() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        (prepared_path / "_index" / "questions").mkdir(parents=True)
        question_seeds = prepared_path / "_index" / "questions" / "question_seeds.md"
        question_seeds.write_text("# Question Seeds\n" + ("seed -> doc_001\n" * 10_000))
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.read_file("_index/questions/question_seeds.md")

        assert result["is_partial"] is True
        assert result["truncated"] is True
        assert "too large for a full read" in result["content"]
        assert "seed -> doc_001" not in result["content"]


def test_agent_excludes_question_seed_reads_from_evidence_context() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(str(prepared_path), client=object())  # type: ignore[arg-type]

        chunk = agent._context_chunk_from_tool_result(
            "_index/questions/question_seeds.md",
            {"content": "# Question Seeds\nseed -> doc_001"},
        )

        assert chunk is None


def test_query_with_trace_uses_single_agent_call() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        rag = FilesystemRAG(prepared_path=str(prepared_path))

        class FakeAgent:
            calls = 0

            def query(self, question: str) -> AgentResponse:
                self.calls += 1
                return AgentResponse(
                    answer="This is the VARE procedure.",
                    context=["VARE candidate context"],
                    metadata={
                        "query_time": 0.01,
                        "files_read": [],
                        "context_sources": ["_summaries/doc_029_summary.md"],
                        "tool_calls": 0,
                        "reasoning_trace": [],
                        "search_mode": "known_item",
                        "iterations": 1,
                        "routing_confidence": 0.95,
                        "prefetch_terms": ["recording", "procedure"],
                        "prefetch_candidates": [
                            {
                                "doc_id": "doc_029",
                                "score": 10.0,
                                "matched_terms": ["recording", "procedure"],
                            }
                        ],
                    },
                )

        fake_agent = FakeAgent()
        rag._agent = fake_agent  # type: ignore[assignment]

        result = rag.query_with_trace(LEGAL_VARE_QUESTION)

        assert fake_agent.calls == 1
        assert result["answer"] == "This is the VARE procedure."
        assert result["context"] == ["VARE candidate context"]

        trace_steps: list[dict[str, Any]] = result["retrieval_trace"]["steps"]
        assert [step["type"] for step in trace_steps] == ["query_routing", "lexical_prefetch"]
        assert trace_steps[1]["output_refs"] == ["doc_029"]
