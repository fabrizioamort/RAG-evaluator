from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_script_module() -> Any:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "legal_rag_retrieval_check.py"
    spec = importlib.util.spec_from_file_location("legal_rag_retrieval_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_tiny_bm25_index(prepared_path: Path) -> None:
    documents = prepared_path / "documents"
    documents.mkdir(parents=True)
    (prepared_path / "_index" / "questions").mkdir(parents=True)
    (prepared_path / "_index" / "questions" / "question_seeds.md").write_text(
        '# Question Seeds\n- "Should Ted\'s friend be excused as a juror?" -> 1.2-c2-s2\n',
        encoding="utf-8",
    )
    (documents / "1.2-c2-s2.md").write_text(
        "Ted is a juror. The court may excuse a potential juror for impartiality.",
        encoding="utf-8",
    )
    (documents / "7.5-c1-s1.md").write_text(
        "Recent possession evidence may support an inference.",
        encoding="utf-8",
    )
    payload = {
        "version": 1,
        "kind": "bm25_passage",
        "parameters": {"k1": 1.5, "b": 0.75},
        "passage_count": 2,
        "avg_doc_length": 5.0,
        "document_frequencies": {
            "ted": 1,
            "juror": 1,
            "excuse": 1,
            "impartiality": 1,
            "recent": 1,
            "possession": 1,
            "evidence": 1,
            "inference": 1,
        },
        "passages": [
            {
                "passage_id": "1.2-c2-s2#L1-L1",
                "doc_id": "1.2-c2-s2",
                "title": "Excusing Jurors",
                "section_title": "Excusing Jurors",
                "source": "documents/1.2-c2-s2.md",
                "summary_source": "_summaries/1.2-c2-s2_summary.md",
                "start_line": 1,
                "end_line": 1,
                "length": 5,
                "term_counts": {"ted": 1, "juror": 2, "excuse": 1, "impartiality": 1},
            },
            {
                "passage_id": "7.5-c1-s1#L1-L1",
                "doc_id": "7.5-c1-s1",
                "title": "Recent Possession",
                "section_title": "Recent Possession",
                "source": "documents/7.5-c1-s1.md",
                "summary_source": "_summaries/7.5-c1-s1_summary.md",
                "start_line": 1,
                "end_line": 1,
                "length": 5,
                "term_counts": {"recent": 1, "possession": 1, "evidence": 1, "inference": 1},
            },
        ],
    }
    index_path = prepared_path / "_index" / "passages" / "bm25.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps(payload), encoding="utf-8")


def test_locked_cases_cover_phase_0_subset() -> None:
    module = load_script_module()

    cases = module.load_cases(None)

    assert len(cases) == 10
    assert {case.gold_passage_id for case in cases} >= {
        "3.6-c2-s1",
        "1.5-c8-s1",
        "1.5-c7-s2",
        "2.3.3-c2-s1",
        "1.5-c5-s1",
    }


def test_result_matches_gold_from_supported_identifiers() -> None:
    module = load_script_module()

    assert module.result_matches_gold({"doc_id": "1.5-c8-s1"}, "1.5-c8-s1")
    assert module.result_matches_gold({"source": "documents/1.5-c8-s1.md"}, "1.5-c8-s1")
    assert module.result_matches_gold({"passage_id": "1.5-c8-s1#L1-L12"}, "1.5-c8-s1")
    assert not module.result_matches_gold({"doc_id": "1.5-c8-s2"}, "1.5-c8-s1")


def test_evaluate_case_reports_gold_rank() -> None:
    module = load_script_module()

    with tempfile.TemporaryDirectory() as temp_dir:
        prepared_path = Path(temp_dir)
        write_tiny_bm25_index(prepared_path)
        index = module.BM25PassageIndex.load(prepared_path)
        case = module.RetrievalCase(
            case_id="legal_001",
            name="Bob & Ted",
            question="Should Ted's friend serving as a juror be excused for impartiality?",
            gold_passage_id="1.2-c2-s2",
        )

        row = module.evaluate_case(index, case, top_k=2)

        assert row["rank"] == 1
        assert row["in_top_5"] is True
        assert row["in_top_k"] is True
        assert row["top_doc_id"] == "1.2-c2-s2"


def test_evaluate_prefetch_case_reports_gold_rank() -> None:
    module = load_script_module()

    with tempfile.TemporaryDirectory() as temp_dir:
        prepared_path = Path(temp_dir)
        write_tiny_bm25_index(prepared_path)
        tools = module.FilesystemRAGTools(str(prepared_path))
        case = module.RetrievalCase(
            case_id="legal_001",
            name="Bob & Ted",
            question="Should Ted's friend serving as a juror be excused for impartiality?",
            gold_passage_id="1.2-c2-s2",
        )

        row = module.evaluate_prefetch_case(tools, case, top_k=2)

        assert row["rank"] == 1
        assert row["in_top_5"] is True
        assert row["in_top_k"] is True
        assert row["top_doc_id"] == "1.2-c2-s2"
