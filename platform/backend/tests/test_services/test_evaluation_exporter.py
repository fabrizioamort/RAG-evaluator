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
            summary_metrics={"g_eval_avg": 0.83},
            performance_metrics={"avg_latency_seconds": 1.23},
            legal_rag_bench={
                "retrieval": {"hit_at_k_rate": 0.52},
                "judge": {"correct_rate": 0.61, "grounded_rate": 0.88},
                "taxonomy": {"success": 61, "retrieval_error": 30, "reasoning_error": 9},
                "success_signals": {
                    "g_eval_pass_rate": 0.6,
                    "taxonomy_success_rate": 0.61,
                    "alternate_evidence_supported_rate": 0.12,
                },
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
            summary_metrics={"g_eval_avg": 0.9},
            performance_metrics={"avg_latency_seconds": 4.1},
            legal_rag_bench={
                "retrieval": {"gold_accessed_rate": 0.70},
                "judge": {"correct_rate": 0.55, "grounded_rate": 0.80},
                "taxonomy": {"success": 55},
                "success_signals": {
                    "g_eval_pass_rate": 0.0,
                    "taxonomy_success_rate": 0.55,
                    "alternate_evidence_supported_rate": 0.25,
                },
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
    assert rows[0]["G-Eval"] == "60.0%"
    assert rows[1]["G-Eval"] == "0.0%"
    assert rows[0]["Judge correct"] == "61.0%"
    assert rows[0]["Judge grounded"] == "88.0%"
    assert rows[0]["Taxonomy success"] == "61.0%"
    assert rows[0]["Alt evidence"] == "12.0%"


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
        scores={"faithfulness": 0.9, "g_eval": 0.72},
        latency_seconds=1.0,
        legal_rag_bench={
            "retrieval": {
                "hit_at_k": True,
                "gold_accessed": False,
                "gold_access_rank": None,
                "relevant_passage_id": "gold-id",
                "retrieved_passage_ids": ["alt-id"],
            },
            "judge": {"correct": True, "grounded": True, "reasoning": "supported"},
            "taxonomy": "success",
            "success_signals": {
                "g_eval_pass": True,
                "supported_by_retrieved_context": True,
                "alternate_evidence_supported": True,
                "correct_without_gold": True,
            },
        },
    )
    text = per_question_jsonl([record, record])
    lines = text.strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["system"] == "Chroma semantic"
    assert parsed["generated_answer"] == "g"
    assert parsed["g_eval_score"] == 0.72
    assert parsed["g_eval_pass"] is True
    assert parsed["legal_taxonomy"] == "success"
    assert parsed["judge_correct"] is True
    assert parsed["judge_grounded"] is True
    assert parsed["judge_reason"] == "supported"
    assert parsed["gold_accessed"] is False
    assert parsed["relevant_passage_id"] == "gold-id"
    assert parsed["retrieved_passage_ids"] == ["alt-id"]
    assert parsed["alternate_evidence_supported"] is True
    assert parsed["correct_without_gold"] is True
    assert parsed["legal_judge"]["correct"] is True
