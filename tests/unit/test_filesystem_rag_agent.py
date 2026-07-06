"""Tests for filesystem RAG query routing and prefetch behavior."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.common.provider_interfaces import RetrievalTrace, RetrievedContext
from rag_evaluator.rag_implementations.filesystem_rag.agent import agent as agent_module
from rag_evaluator.rag_implementations.filesystem_rag.agent.agent import (
    AgentResponse,
    FilesystemRAGAgent,
)
from rag_evaluator.rag_implementations.filesystem_rag.agent.prefetch import (
    build_prefetch_context,
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
from rag_evaluator.rag_implementations.filesystem_rag.passage_index import (
    build_bm25_passage_index,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
    DocumentAnalysis,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
)

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
    (root / "_index" / "passages").mkdir(parents=True)
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
    doc_013_content = (
        "# Use of Circumstantial Evidence\n\n"
        "The prosecution may use circumstantial evidence to prove elements of an "
        "offence when the facts support the inference."
    )
    doc_029_content = (
        "# The VARE Procedure\n\n"
        "A prosecution witness's evidence-in-chief may be given by way of an audio "
        "or audiovisual recording in which the witness answers questions put by a "
        'prescribed person. The recording is called a "VARE".'
    )
    doc_045_content = (
        "# 2.1 Views\n\n"
        "## What is a view?\n\n"
        "## 1. Under s 53 of the Evidence Act 2008, the court may order a "
        '"demonstration, experiment or inspection" (collectively, a "view").\n\n'
        "## 2. These terms are not defined in the Evidence Act 2008. Based on "
        "the common law and the conventional meaning of the terms:\n\n"
        "- An inspection involves the court travelling to view a location or an "
        "object that could not be brought into the courtroom;\n\n"
        "- A demonstration builds on an inspection by allowing a witness to "
        "explain the incident in question.\n"
    )
    (root / "documents" / "doc_013.md").write_text(doc_013_content, encoding="utf-8")
    (root / "documents" / "doc_029.md").write_text(doc_029_content, encoding="utf-8")
    (root / "documents" / "doc_045.md").write_text(
        doc_045_content,
        encoding="utf-8",
    )

    build_bm25_passage_index(
        [
            (
                _processed_doc("doc_013", "Use of Circumstantial Evidence", doc_013_content),
                _analysis(
                    "The prosecution may use circumstantial evidence to prove elements.",
                    ["What is Use of Circumstantial Evidence?"],
                ),
            ),
            (
                _processed_doc("doc_029", "The VARE Procedure", doc_029_content),
                _analysis(
                    "A witness's evidence-in-chief may be given by audio or audiovisual "
                    'recording. The recording is called a "VARE".',
                    [
                        "What is the evidentiary process for recorded witness testimony?",
                        "What is The VARE Procedure?",
                    ],
                ),
            ),
            (
                _processed_doc("doc_045", "2.1 Views", doc_045_content),
                _analysis(
                    'The court may order a "demonstration, experiment or inspection" '
                    'collectively called a "view"; an inspection involves travelling '
                    "to view a location or object.",
                    [
                        "What is the legal procedure where the court travels to a location?",
                        "What is a view?",
                    ],
                ),
            ),
        ],
        root,
    )


def _processed_doc(doc_id: str, title: str, content: str) -> ProcessedDocument:
    return ProcessedDocument(
        id=doc_id,
        original_path=f"{doc_id}.txt",
        original_format="txt",
        markdown_content=content,
        title=title,
        word_count=len(content.split()),
        char_count=len(content),
        line_count=len(content.splitlines()),
        sections=[],
    )


def _analysis(summary: str, question_seeds: list[str]) -> DocumentAnalysis:
    return DocumentAnalysis(
        summary=summary,
        topics=[],
        topic_scores={"general": 1.0},
        entities={},
        temporal_markers=[],
        question_seeds=question_seeds,
        key_sections=[],
        related_topics=[],
        analysis_method="heuristic",
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
            "Candidate BM25 Passage: doc_045" in chunk and 'collectively, a "view"' in chunk
            for chunk in prefetch["chunks"]
        )


def test_prefetch_merges_question_seed_navigation_hints() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(str(prepared_path), client=object())  # type: ignore[arg-type]

        prefetch = agent._build_prefetch_context(LEGAL_VARE_QUESTION)

        vare_candidate = next(
            candidate for candidate in prefetch["candidates"] if candidate["doc_id"] == "doc_029"
        )
        assert "question_seed" in vare_candidate["prefetch_sources"]
        assert prefetch["question_seed_hints"][0]["doc_id"] == "doc_029"


def test_prefetch_caps_candidates_per_section_family() -> None:
    class FakeTools:
        prepared_path = Path("__missing__")

        def search_passages(self, _query: str, top_k: int = 5) -> dict[str, Any]:
            results = [
                _prefetch_candidate("1.5-c1-s1", 90),
                _prefetch_candidate("1.5-c2-s1", 80),
                _prefetch_candidate("1.5-c3-s1", 70),
                _prefetch_candidate("2.3-c1-s1", 60),
            ]
            return {"query_terms": ["juror"], "results": results[:top_k]}

    prefetch = build_prefetch_context(  # type: ignore[arg-type]
        FakeTools(),
        "Should a juror exposed to news stories be excused?",
        max_candidates=4,
    )

    selected_ids = [candidate["doc_id"] for candidate in prefetch["candidates"]]
    assert selected_ids.count("1.5-c1-s1") == 1
    assert "1.5-c2-s1" in selected_ids
    assert "1.5-c3-s1" not in selected_ids
    assert "2.3-c1-s1" in selected_ids


def _prefetch_candidate(doc_id: str, score: float) -> dict[str, Any]:
    return {
        "passage_id": f"{doc_id}#L1-L3",
        "doc_id": doc_id,
        "score": score,
        "matched_terms": ["juror"],
        "title": doc_id,
        "section_title": doc_id,
        "source": f"documents/{doc_id}.md",
        "summary_source": f"_summaries/{doc_id}_summary.md",
        "start_line": 1,
        "end_line": 3,
        "snippet": "snippet",
        "read_hint": {"path": f"documents/{doc_id}.md", "start_line": 1, "end_line": 3},
    }


def test_search_passages_tool_returns_ranked_snippet_and_read_hint() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.search_passages(
            "What is the legal procedure where the court travels to a location?",
            top_k=2,
        )

        assert result["results"][0]["doc_id"] == "doc_045"
        assert 'collectively, a "view"' in result["results"][0]["snippet"]
        assert result["results"][0]["read_hint"] == {
            "path": "documents/doc_045.md",
            "start_line": 1,
            "end_line": len((prepared_path / "documents" / "doc_045.md").read_text().splitlines()),
        }


def test_search_passages_tool_reports_missing_index() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.search_passages("VARE")

        assert result["results"] == []
        assert "BM25 passage index not found" in result["error"]


def test_grep_search_ranks_and_reports_truncation_for_all_terms() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        docs_dir = prepared_path / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "doc_001.md").write_text(
            "silence only\nanother silence mention\n",
            encoding="utf-8",
        )
        (docs_dir / "doc_002.md").write_text(
            "\n".join(
                [
                    "silence appears here",
                    "nearby filler",
                    "admission appears nearby",
                    "the decisive line joins silence and admission",
                    "another admission line",
                ]
            ),
            encoding="utf-8",
        )
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.grep_search(
            "silence admission",
            path="documents",
            max_results=1,
            context_lines=2,
            match_all_terms=True,
        )

        assert result["files_searched"] == 2
        assert result["files_with_matches"] == 1
        assert result["total_matches"] == 4
        assert result["returned_matches"] == 1
        assert result["truncated"] is True
        assert result["results"][0]["file"].replace("\\", "/") == "documents/doc_002.md"
        assert "silence and admission" in result["results"][0]["content"]
        assert result["files"][0]["file"].replace("\\", "/") == "documents/doc_002.md"


def test_grep_search_falls_back_to_and_mode_on_zero_hits() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        docs_dir = prepared_path / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "doc_001.md").write_text(
            "silence appears here\nan accusation follows\nan admission concludes\n",
            encoding="utf-8",
        )
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.grep_search("silence accusation admission", path="documents")

        assert result["fallback"] == "match_all_terms"
        assert result["pattern"] == "silence accusation admission"
        assert result["match_all_terms"] is True
        assert result["total_matches"] == 3
        assert result["files"][0]["file"].replace("\\", "/") == "documents/doc_001.md"


def test_grep_search_does_not_fall_back_for_regex_patterns() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        docs_dir = prepared_path / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "doc_001.md").write_text("foo here\nbar there\n", encoding="utf-8")
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.grep_search("baz|qux", path="documents")

        assert result["total_matches"] == 0
        assert "fallback" not in result


def test_grep_search_does_not_set_fallback_on_direct_hits() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        docs_dir = prepared_path / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "doc_001.md").write_text(
            "the silence accusation admission line\n", encoding="utf-8"
        )
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.grep_search("silence accusation admission", path="documents")

        assert result["total_matches"] == 1
        assert "fallback" not in result


def _write_section_sibling_fixture(prepared_path: Path) -> None:
    docs = prepared_path / "documents"
    docs.mkdir(parents=True)
    (docs / "1.2-c1-s1.md").write_text("# Provocation Intro\n\nBody.", encoding="utf-8")
    (docs / "1.2-c2-s1.md").write_text(
        "# 428C6B2E Passage 0956  1 2 C2 S1\n\n## Elements of Provocation\n\nBody.",
        encoding="utf-8",
    )
    (docs / "1.2.1-c1-s1.md").write_text("# Charge: Provocation\n\nBody.", encoding="utf-8")
    (docs / "1.2-c1-s1.meta.json").write_text("{}", encoding="utf-8")
    summaries = prepared_path / "_summaries"
    summaries.mkdir()
    (summaries / "1.2-c1-s1_summary.md").write_text("# Summary", encoding="utf-8")
    topics = prepared_path / "_index" / "topics"
    topics.mkdir(parents=True)
    (topics / "_topic_map.md").write_text("# Topics", encoding="utf-8")


def test_read_file_returns_section_siblings_with_informative_titles() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_section_sibling_fixture(prepared_path)
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.read_file("documents/1.2-c1-s1.md")

        assert result["section_id"] == "1.2"
        assert result["section_siblings"] == [
            {"file": "documents/1.2-c2-s1.md", "title": "Elements of Provocation"}
        ]

        reverse = tools.read_file("documents/1.2-c2-s1.md")
        assert reverse["section_siblings"] == [
            {"file": "documents/1.2-c1-s1.md", "title": "Provocation Intro"}
        ]


def test_read_file_omits_section_siblings_for_non_document_reads() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_section_sibling_fixture(prepared_path)
        tools = FilesystemRAGTools(str(prepared_path))

        assert "section_siblings" not in tools.read_file("_summaries/1.2-c1-s1_summary.md")
        assert "section_siblings" not in tools.read_file("_index/topics/_topic_map.md")
        assert "section_siblings" not in tools.read_file("documents/1.2-c1-s1.meta.json")


def test_format_tool_result_preserves_sibling_block_when_truncating() -> None:
    result = {
        "content": "x" * 30_000,
        "total_lines": 400,
        "is_partial": True,
        "headers": [],
        "section_id": "8.12",
        "section_siblings": [
            {"file": "documents/8.12-c3-s1.md", "title": '"Heat of passion"'},
        ],
    }

    formatted = format_tool_result("read_file", result)

    assert "start_line/end_line" in formatted
    assert formatted.endswith(
        "[Other chunks in section 8.12 — read any whose title matches the question:]\n"
        '  documents/8.12-c3-s1.md — "Heat of passion"'
    )
    assert len(formatted) < 30_000


def test_bm25_passage_index_persists_section_metadata() -> None:
    content = (
        "# Legal Notes\n\n"
        "Intro paragraph for the corpus.\n\n"
        "## VARE\n"
        "The recording is called VARE.\n\n"
        "## Views\n"
        "The court travels to inspect a location."
    )
    doc = ProcessedDocument(
        id="doc_001",
        original_path="legal_notes.txt",
        original_format="txt",
        markdown_content=content,
        title="Legal Notes",
        word_count=len(content.split()),
        char_count=len(content),
        line_count=len(content.splitlines()),
        sections=[
            {"title": "Legal Notes", "start_line": 1, "end_line": 4, "level": 1},
            {"title": "VARE", "start_line": 5, "end_line": 7, "level": 2},
            {"title": "Views", "start_line": 8, "end_line": 9, "level": 2},
        ],
    )

    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)
        build_bm25_passage_index(
            [(doc, _analysis("Legal evidence notes.", ["What is VARE?"]))],
            output_path,
        )

        payload = json.loads((output_path / "_index" / "passages" / "bm25.json").read_text())
        vare_passage = next(p for p in payload["passages"] if p["section_title"] == "VARE")

        assert payload["kind"] == "bm25_passage"
        assert vare_passage["passage_id"] == "doc_001#L5-L7"
        assert vare_passage["source"] == "documents/doc_001.md"
        assert vare_passage["start_line"] == 5
        assert vare_passage["end_line"] == 7


def test_search_passages_dedupes_results_by_document() -> None:
    multi_content = (
        "# Provocation Notes\n\n"
        "Intro about provocation doctrine and heat of passion generally.\n\n"
        "## Elements\n"
        "The provocation defence requires a wrongful act and loss of self "
        "control in the heat of passion.\n\n"
        "## History\n"
        "Provocation history covers heat of passion rulings from older courts.\n\n"
        "## Charge\n"
        "The provocation charge mentions heat of passion directions for juries."
    )
    multi_doc = ProcessedDocument(
        id="doc_multi",
        original_path="multi.txt",
        original_format="txt",
        markdown_content=multi_content,
        title="Provocation Notes",
        word_count=len(multi_content.split()),
        char_count=len(multi_content),
        line_count=len(multi_content.splitlines()),
        sections=[
            {"title": "Provocation Notes", "start_line": 1, "end_line": 4, "level": 1},
            {"title": "Elements", "start_line": 5, "end_line": 8, "level": 2},
            {"title": "History", "start_line": 9, "end_line": 11, "level": 2},
            {"title": "Charge", "start_line": 12, "end_line": 14, "level": 2},
        ],
    )
    single_doc = _processed_doc(
        "doc_single",
        "Sudden Fight",
        "# Sudden Fight\n\n"
        "A separate doctrine where a sudden fight reduces murder culpability; "
        "provocation is mentioned once for contrast.",
    )

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        (prepared_path / "documents").mkdir(parents=True)
        build_bm25_passage_index(
            [
                (multi_doc, _analysis("Provocation and heat of passion notes.", [])),
                (single_doc, _analysis("Sudden fight doctrine.", [])),
            ],
            prepared_path,
        )
        tools = FilesystemRAGTools(str(prepared_path))

        result = tools.search_passages("provocation heat of passion", top_k=2)

        doc_ids = [item["doc_id"] for item in result["results"]]
        assert doc_ids == ["doc_multi", "doc_single"]
        assert result["results"][0]["other_matching_sections"] >= 1


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
    assert (
        agent_module.unusable_answer_reason(
            '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="read_file">'
            '<｜｜DSML｜｜parameter name="path">documents/7.3.2-c4-s1.md'
        )
        == "tool_markup"
    )
    assert (
        agent_module.unusable_answer_reason(
            "I verified this with the read_file tool against the charge book."
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
        assert response.metadata["llm_request_params"] == {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "reasoning_effort": None,
        }


class FakeFunctionCall:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = call_id
        self.function = FakeFunctionCall(name, arguments)


class FakeToolMessage:
    def __init__(self, tool_calls: list[FakeToolCall]) -> None:
        self.content = None
        self.tool_calls = tool_calls

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ],
        }


class FakeFinalMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class FakeChoice:
    def __init__(self, message: FakeToolMessage | FakeFinalMessage) -> None:
        self.message = message


class FakeResponse:
    def __init__(
        self,
        message: FakeToolMessage | FakeFinalMessage,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.choices = [FakeChoice(message)]
        self.usage = usage


def test_agent_recovers_from_leaked_tool_markup_by_reissuing_tool_call() -> None:
    markup = (
        '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="read_file">'
        '<｜｜DSML｜｜parameter name="path">documents/doc_029.md'
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(FakeFinalMessage(markup))
            if self.calls == 2:
                assert any(
                    message.get("role") == "user"
                    and "raw tool-call markup" in message.get("content", "")
                    for message in kwargs["messages"]
                )
                return FakeResponse(
                    FakeToolMessage(
                        [
                            FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"}),
                            FakeToolCall("call_2", "read_file", {"path": "documents/doc_013.md"}),
                        ]
                    )
                )
            return FakeResponse(FakeFinalMessage("The recording procedure is called a VARE."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "The recording procedure is called a VARE."
        assert response.metadata["markup_recovery_used"] is True
        assert response.metadata["answer_retries"] == 0
        assert "documents/doc_029.md" in response.metadata["files_read"]
        assert completions.calls == 3


def test_agent_nudges_once_for_verification_when_fewer_than_two_docs_read() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )
            if self.calls == 2:
                return FakeResponse(FakeFinalMessage("Early answer from one document."))
            if self.calls == 3:
                assert any(
                    message.get("role") == "user"
                    and "read at least one more relevant document chunk" in message.get("content", "")
                    for message in kwargs["messages"]
                )
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_2", "read_file", {"path": "documents/doc_013.md"})]
                    )
                )
            return FakeResponse(FakeFinalMessage("Verified answer from two documents."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Verified answer from two documents."
        assert response.metadata["evidence_nudge_used"] is True
        assert response.metadata["answer_retries"] == 0
        assert completions.calls == 4


def test_agent_skips_nudge_when_two_distinct_documents_read() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [
                            FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"}),
                            FakeToolCall("call_2", "read_file", {"path": "documents/doc_013.md"}),
                        ]
                    )
                )
            return FakeResponse(FakeFinalMessage("Answer from two documents."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Answer from two documents."
        assert response.metadata["evidence_nudge_used"] is False
        assert completions.calls == 2


def test_agent_accepts_answer_after_single_nudge_without_new_reads() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )
            return FakeResponse(FakeFinalMessage("The answer stands as given."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "The answer stands as given."
        assert response.metadata["evidence_nudge_used"] is True
        assert completions.calls == 3


def test_agent_short_circuits_when_tool_call_limit_reached() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                assert "tools" in kwargs
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )

            assert "tools" not in kwargs
            assert any(
                message.get("role") == "tool"
                and "Maximum tool calls reached" in message.get("content", "")
                for message in kwargs["messages"]
            )
            return FakeResponse(FakeFinalMessage("The best available answer."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_iterations=5,
            max_tool_calls=0,
            client=FakeClient(completions),  # type: ignore[arg-type]
        )

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "The best available answer."
        assert response.metadata["limit_reached"] == "tool_calls"
        assert response.metadata["max_iterations_reached"] is False
        assert response.metadata["iterations"] == 1
        assert completions.calls == 2


def test_agent_short_circuits_when_file_read_limit_reached() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )

            assert "tools" not in kwargs
            assert any(
                message.get("role") == "tool"
                and "Maximum file reads reached" in message.get("content", "")
                for message in kwargs["messages"]
            )
            return FakeResponse(FakeFinalMessage("The best available answer."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_iterations=5,
            max_file_reads=0,
            client=FakeClient(completions),  # type: ignore[arg-type]
        )

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "The best available answer."
        assert response.metadata["limit_reached"] == "file_reads"
        assert response.metadata["max_iterations_reached"] is False
        assert response.metadata["iterations"] == 1
        assert completions.calls == 2


def test_agent_compacts_old_tool_messages_but_preserves_pairing() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )
            if self.calls == 2:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_2", "read_file", {"path": "documents/doc_013.md"})]
                    )
                )
            if self.calls == 3:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_3", "search_passages", {"query": "circumstantial"})]
                    )
                )

            assistant_tool_messages = [
                message
                for message in kwargs["messages"]
                if message.get("role") == "assistant" and message.get("tool_calls")
            ]
            tool_messages = [
                message for message in kwargs["messages"] if message.get("role") == "tool"
            ]
            assert len(assistant_tool_messages) == 3
            assert len(tool_messages) == 3
            assert tool_messages[0]["tool_call_id"] == "call_1"
            assert "result elided - read_file(path=\"documents/doc_029.md\")" in tool_messages[
                0
            ]["content"]
            assert "Use of Circumstantial Evidence" in tool_messages[1]["content"]
            assert "circumstantial" in tool_messages[2]["content"]
            return FakeResponse(FakeFinalMessage("Answer from compacted history."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]
        agent._build_prefetch_context = lambda _question: {  # type: ignore[method-assign]
            "chunks": [],
            "sources": [],
            "terms": [],
            "candidates": [],
        }

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Answer from compacted history."
        assert completions.calls == 4


def test_agent_keeps_recent_tool_messages_uncompacted() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_1", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )
            if self.calls == 2:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_2", "read_file", {"path": "documents/doc_013.md"})]
                    )
                )

            tool_messages = [
                message for message in kwargs["messages"] if message.get("role") == "tool"
            ]
            assert len(tool_messages) == 2
            assert "result elided" not in tool_messages[0]["content"]
            assert "The VARE Procedure" in tool_messages[0]["content"]
            assert "Use of Circumstantial Evidence" in tool_messages[1]["content"]
            return FakeResponse(FakeFinalMessage("Answer from recent history."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(str(prepared_path), client=FakeClient(completions))  # type: ignore[arg-type]
        agent._build_prefetch_context = lambda _question: {  # type: ignore[method-assign]
            "chunks": [],
            "sources": [],
            "terms": [],
            "candidates": [],
        }

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Answer from recent history."
        assert completions.calls == 3


def test_agent_returns_cached_reference_for_repeated_identical_tool_call() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(
                    FakeToolMessage(
                        [
                            FakeToolCall(
                                "call_1",
                                "read_file",
                                {"path": "documents/doc_029.md"},
                            ),
                            FakeToolCall(
                                "call_2",
                                "read_file",
                                {"path": "documents/doc_013.md"},
                            ),
                        ]
                    )
                )
            if self.calls == 2:
                return FakeResponse(
                    FakeToolMessage(
                        [FakeToolCall("call_3", "read_file", {"path": "documents/doc_029.md"})]
                    )
                )

            tool_messages = [
                message for message in kwargs["messages"] if message.get("role") == "tool"
            ]
            assert "cached result elided" in tool_messages[-1]["content"]
            assert "documents/doc_029.md" in tool_messages[-1]["content"]
            assert "The VARE Procedure" not in tool_messages[-1]["content"]
            return FakeResponse(FakeFinalMessage("Answer with cached repeat."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_file_reads=2,
            client=FakeClient(completions),  # type: ignore[arg-type]
        )
        agent._build_prefetch_context = lambda _question: {  # type: ignore[method-assign]
            "chunks": [],
            "sources": [],
            "terms": [],
            "candidates": [],
        }

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Answer with cached repeat."
        assert response.metadata["tool_calls"] == 3
        assert response.metadata["files_read"].count("documents/doc_029.md") == 1
        assert response.metadata["context_sources"].count("documents/doc_029.md") == 1
        assert completions.calls == 3


def test_agent_budget_nudge_fires_once_at_sixty_percent_budget() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls <= 3:
                return FakeResponse(
                    FakeToolMessage(
                        [
                            FakeToolCall(
                                f"call_{self.calls}",
                                "search_passages",
                                {"query": f"query {self.calls}"},
                            )
                        ]
                    )
                )

            nudges = [
                message
                for message in kwargs["messages"]
                if message.get("role") == "system"
                and "used most of your iteration/tool budget" in message.get("content", "")
            ]
            assert len(nudges) == 1
            return FakeResponse(FakeFinalMessage("Budget-aware answer."))

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = type("FakeChat", (), {"completions": completions})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        completions = FakeCompletions()
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_iterations=5,
            max_file_reads=0,
            client=FakeClient(completions),  # type: ignore[arg-type]
        )
        agent._build_prefetch_context = lambda _question: {  # type: ignore[method-assign]
            "chunks": [],
            "sources": [],
            "terms": [],
            "candidates": [],
        }

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.answer == "Budget-aware answer."
        assert response.metadata["budget_nudge_used"] is True
        assert completions.calls == 4


def test_agent_does_not_report_prefetch_candidates_as_evidence_context() -> None:
    class FakeCompletions:
        def create(self, **_kwargs: Any) -> FakeResponse:
            return FakeResponse(FakeFinalMessage("The answer comes after prefetch only."))

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("FakeChat", (), {"completions": FakeCompletions()})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_iterations=1,
            client=FakeClient(),  # type: ignore[arg-type]
        )

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.context == []
        assert response.metadata["context_sources"] == []
        assert response.metadata["prefetch_candidates"]


def test_agent_records_provider_token_usage() -> None:
    class FakeCompletions:
        def create(self, **_kwargs: Any) -> FakeResponse:
            return FakeResponse(
                FakeFinalMessage("The recording procedure is called a VARE."),
                usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            )

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("FakeChat", (), {"completions": FakeCompletions()})()

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(
            str(prepared_path),
            max_iterations=1,
            client=FakeClient(),  # type: ignore[arg-type]
        )

        response = agent.query(LEGAL_VARE_QUESTION)

        assert response.metadata["token_usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }


def test_filesystem_rag_tracks_measured_agent_tokens_not_estimates() -> None:
    rag = FilesystemRAG()
    response = AgentResponse(
        answer="This answer is deliberately long enough to make estimates obvious.",
        context=[],
        metadata={
            "query_time": 0.01,
            "tool_calls": 0,
            "files_read": [],
            "search_mode": "known_item",
            "iterations": 4,
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
    )

    rag._track_agent_response("A much longer question than seven tokens would imply", response)

    assert rag._token_usage.to_dict() == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "embedding_tokens": 0,
        "total_tokens": 10,
    }


def test_filesystem_rag_resolves_passage_id_filename_sources_via_metadata() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        docs_dir = prepared_path / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "1.5-c6-s1.md").write_text("# Passage", encoding="utf-8")
        (docs_dir / "1.5-c6-s1.meta.json").write_text(
            json.dumps(
                {
                    "id": "1.5-c6-s1",
                    "original_file": "raw/fd237f78_passage_0035__1_5-c6-s1.txt",
                }
            ),
            encoding="utf-8",
        )
        rag = FilesystemRAG(prepared_path=str(prepared_path))

        resolved = rag._resolve_source_to_passage("documents/1.5-c6-s1.md")

        assert resolved.replace("\\", "/") == "raw/fd237f78_passage_0035__1_5-c6-s1.txt"


def test_query_with_trace_filters_navigation_noise_from_reported_sources() -> None:
    class FakeAgent:
        def query(self, _question: str) -> AgentResponse:
            return AgentResponse(
                answer="Answer.",
                context=["chunk"],
                metadata={
                    "query_time": 0.01,
                    "tool_calls": 3,
                    "files_read": [
                        "_index/topics/alibi.md",
                        "_index/questions/question_seeds.md",
                        "documents/4.6-c4-s1.md",
                        "documents/4.6-c4-s1.meta.json",
                    ],
                    "context_sources": [
                        "_index/topics/alibi.md",
                        "documents/4.6-c4-s1.md",
                    ],
                    "search_mode": "known_item",
                    "iterations": 2,
                },
            )

    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        (prepared_path / "documents").mkdir(parents=True)
        rag = FilesystemRAG(prepared_path=str(prepared_path))
        rag._agent = FakeAgent()  # type: ignore[assignment]

        result = rag.query_with_trace("What reasoning applies to an alibi?")

        assert result["metadata"]["files_read"] == ["documents/4.6-c4-s1.md"]
        assert result["metadata"]["context_sources"] == ["documents/4.6-c4-s1.md"]


def test_reportable_sources_falls_back_when_filter_empties_the_list() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        (prepared_path / "documents").mkdir(parents=True)
        rag = FilesystemRAG(prepared_path=str(prepared_path))

        sources = ["documents/doc_029.md", "_index/topics/_topic_map.md"]

        assert rag._reportable_sources(sources) == sources


def test_generate_uses_single_plain_completion_without_agent_navigation() -> None:
    class FakeGenerateAgent:
        def __init__(self) -> None:
            self.calls = 0
            self.query_calls = 0
            self.usage: dict[str, int] = {}
            self.last_kwargs: dict[str, Any] = {}

        def reset_llm_usage(self) -> None:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _create_chat_completion_with_retries(self, kwargs: dict[str, Any]) -> FakeResponse:
            self.calls += 1
            self.last_kwargs = kwargs
            self.usage = {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17}
            return FakeResponse(
                FakeFinalMessage("The answer comes from the supplied context."),
                usage=self.usage,
            )

        def get_llm_usage(self) -> dict[str, int]:
            return self.usage

        def query(self, _question: str) -> AgentResponse:
            self.query_calls += 1
            raise AssertionError("generate() must not re-enter agent navigation")

    rag = FilesystemRAG()
    fake_agent = FakeGenerateAgent()
    rag._agent = fake_agent  # type: ignore[assignment]
    context = RetrievedContext(
        chunks=["The supplied context says the answer is VARE."],
        chunk_details=[],
        trace=RetrievalTrace(strategy="agentic"),
        retrieval_time=0.0,
    )

    generated = rag.generate("What is the procedure called?", context)

    assert generated.text == "The answer comes from the supplied context."
    assert generated.prompt_tokens == 13
    assert generated.completion_tokens == 4
    assert fake_agent.calls == 1
    assert fake_agent.query_calls == 0
    assert "tools" not in fake_agent.last_kwargs
    assert "supplied context" in fake_agent.last_kwargs["messages"][1]["content"]


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


def test_agent_excludes_bm25_index_reads_from_evidence_context() -> None:
    with TemporaryDirectory() as tmp_dir:
        prepared_path = Path(tmp_dir)
        _write_prepared_fixture(prepared_path)
        agent = FilesystemRAGAgent(str(prepared_path), client=object())  # type: ignore[arg-type]

        chunk = agent._context_chunk_from_tool_result(
            "_index/passages/bm25.json",
            {"content": '{"passages": []}'},
        )

        assert chunk is None


def test_filesystem_rag_passes_configured_preparation_options(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> dict[str, Any]:
            return {"metrics": {}, "documents_processed": 0}

    config = RAGConfig(
        name="Filesystem RAG",
        parameters={
            "filesystem_force_analysis_method": "llm",
            "filesystem_word_threshold": 9999,
            "filesystem_use_llm_synthesis": True,
        },
    )
    rag = FilesystemRAG(config=config)
    monkeypatch.setattr(
        "rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag.PreparationPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(rag, "_initialize_agent", lambda: None)

    rag.prepare_documents("data/raw")

    assert captured["word_threshold"] == 9999
    assert captured["force_analysis_method"] == "llm"
    assert captured["use_llm_synthesis"] is True


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
