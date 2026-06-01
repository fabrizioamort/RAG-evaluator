"""Tests for filesystem RAG query routing and prefetch behavior."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.agent.agent import (
    AgentResponse,
    FilesystemRAGAgent,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.router import (
    QueryRouter,
    SearchMode,
)
from rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag import FilesystemRAG

LEGAL_VARE_QUESTION = (
    "Josh is 21 years old. He witnessed a murder that was clearly perpetrated by the "
    "accused. The prosecution, however, needs to determine whether the accused committed "
    "the acts voluntarily, thus satisfying the elements of intentional or reckless murder. "
    "In court, the judge plays a recording of Josh giving his eyewitness testimony. "
    "What is this evidentiary process called?"
)


def _write_prepared_fixture(root: Path) -> None:
    """Create a small prepared filesystem fixture."""
    (root / "_meta").mkdir(parents=True)
    (root / "_index" / "topics").mkdir(parents=True)
    (root / "_index" / "entities").mkdir(parents=True)
    (root / "_index" / "questions").mkdir(parents=True)
    (root / "_summaries").mkdir(parents=True)

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
                '# Question Seeds',
                '- "What is Use of Circumstantial Evidence?" -> doc_013',
                '- "What is The VARE Procedure?" -> doc_029',
                (
                    '- "What does the section on a prosecution witness\'s '
                    'evidence-in-chief by way of an audio or audiovisual recording '
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
        assert any("recording" in candidate["matched_terms"] for candidate in prefetch["candidates"])


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
