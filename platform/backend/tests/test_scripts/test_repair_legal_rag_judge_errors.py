"""Tests for Legal RAG repair/rejudge script helpers."""

import uuid

from app.models.test_case import TestCase
from scripts.repair_legal_rag_judge_errors import _matches_source_qa_filter, _needs_rejudge


def test_needs_rejudge_repair_mode_only_selects_errors() -> None:
    scored = {
        "legal_rag_bench": {
            "judge": {"correct": True, "grounded": True},
            "taxonomy": "success",
        }
    }
    failed = {
        "legal_rag_bench": {
            "judge": {"correct": None, "grounded": None, "parse_error": "judge_exception"},
            "taxonomy": "judge_error",
        }
    }

    assert _needs_rejudge(scored, rejudge_all=False) is False
    assert _needs_rejudge(failed, rejudge_all=False) is True


def test_needs_rejudge_all_mode_selects_scored_legal_rows() -> None:
    scored = {
        "legal_rag_bench": {
            "judge": {"correct": True, "grounded": True},
            "taxonomy": "success",
        }
    }

    assert _needs_rejudge(scored, rejudge_all=True) is True
    assert _needs_rejudge({"metric_results": []}, rejudge_all=True) is False


def test_matches_source_qa_filter_accepts_selected_case() -> None:
    test_case = TestCase(
        test_set_id=uuid.uuid4(),
        question="Question?",
        expected_answer="Answer.",
        metadata_={"source_qa_id": 10},
    )

    assert _matches_source_qa_filter(test_case, {10}) is True
    assert _matches_source_qa_filter(test_case, {2, 5, 8}) is False
    assert _matches_source_qa_filter(test_case, None) is True


def test_matches_source_qa_filter_rejects_missing_metadata() -> None:
    test_case = TestCase(
        test_set_id=uuid.uuid4(),
        question="Question?",
        expected_answer="Answer.",
        metadata_={},
    )

    assert _matches_source_qa_filter(test_case, {10}) is False
