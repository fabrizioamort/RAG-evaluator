"""Tests for Legal RAG repair/rejudge script helpers."""

from scripts.repair_legal_rag_judge_errors import _needs_rejudge


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
