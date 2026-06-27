"""Unit tests for the pure Legal RAG Bench export formatting helpers."""

import json

from app.services.evaluation_exporter import (
    HEADLINE_COLUMNS,
    ExportMember,
    build_markdown_report,
    build_question_record,
    headline_rows,
    per_question_jsonl,
    taxonomy_columns,
    taxonomy_rows,
    to_csv,
)


def _members() -> list[ExportMember]:
    return [
        ExportMember(
            label="Chroma semantic",
            rag_type="vector_semantic",
            pass_rate=0.61,
            performance_metrics={"avg_latency_seconds": 1.23},
            legal_rag_bench={
                "retrieval": {"hit_at_k_rate": 0.52},
                "judge": {"correct_rate": 0.61, "grounded_rate": 0.88},
                "taxonomy": {"success": 61, "retrieval_error": 30, "reasoning_error": 9},
            },
            manifest={
                "generation_model": "gpt-4o-mini",
                "eval_judge_model": "gpt-4o",
                "effective_config_snapshot": {"top_k": 5},
            },
        ),
        ExportMember(
            label="Filesystem RAG",
            rag_type="filesystem_rag",
            pass_rate=0.55,
            performance_metrics={"avg_latency_seconds": 4.1},
            legal_rag_bench={
                "retrieval": {"gold_accessed_rate": 0.70},
                "judge": {"correct_rate": 0.55, "grounded_rate": 0.80},
                "taxonomy": {"success": 55},
            },
        ),
    ]


def test_headline_rows_pick_retrieval_metric_per_system():
    rows = headline_rows(_members())
    assert rows[0]["Retrieval metric"] == "hit@5"
    assert rows[0]["Retrieval score"] == "52.0%"
    assert rows[0]["Retrieval mode"] == "dense vector"
    assert rows[1]["Retrieval metric"] == "gold_accessed"
    assert rows[1]["Retrieval score"] == "70.0%"
    assert rows[1]["Retrieval mode"] == "agentic file search"
    assert rows[0]["Avg latency"] == "1.23s"


def test_headline_csv_has_header_and_rows():
    csv_text = to_csv(headline_rows(_members()), HEADLINE_COLUMNS)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("System,Retrieval mode")
    assert len(lines) == 3


def test_taxonomy_rows_default_missing_to_zero():
    rows = taxonomy_rows(_members())
    cols = taxonomy_columns()
    assert "Success" in cols
    assert rows[1]["Retrieval error"] == "0"
    assert rows[0]["Success"] == "61"


def test_markdown_report_contains_tables_and_manifest():
    report = build_markdown_report(_members(), title="My Comparison")
    assert "# My Comparison" in report
    assert "## Headline metrics" in report
    assert "## Taxonomy" in report
    assert "gpt-4o-mini" in report
    assert '"top_k": 5' in report


def test_per_question_jsonl_is_one_object_per_line():
    record = build_question_record(
        member_label="Chroma semantic",
        evaluation_id="e1",
        rag_type="vector_semantic",
        rag_config_name="chroma",
        question="q",
        expected_answer="a",
        generated_answer="g",
        scores={"faithfulness": 0.9},
        latency_seconds=1.0,
        legal_rag_bench={
            "retrieval": {"hit_at_k": True},
            "judge": {"correct": True, "grounded": True},
            "taxonomy": "success",
        },
    )
    text = per_question_jsonl([record, record])
    lines = text.strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["system"] == "Chroma semantic"
    assert parsed["legal_taxonomy"] == "success"
    assert parsed["legal_judge"]["correct"] is True
