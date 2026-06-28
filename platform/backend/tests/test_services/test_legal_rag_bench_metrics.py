"""Tests for Legal RAG Bench retrieval and taxonomy helpers."""

from app.services.legal_rag_bench_metrics import (
    compute_legal_rag_retrieval_metrics,
    derive_taxonomy,
    extract_relevant_passage_id,
    summarize_legal_rag_metrics,
)


def test_extract_relevant_passage_id_prefers_metadata() -> None:
    passage_id = extract_relevant_passage_id(
        {
            "benchmark": {
                "relevant_passage_id": "nested-id",
            },
            "relevant_passage_id": "top-level-id",
        },
        ground_truth_context=["Passage ID: context-id\nText"],
    )

    assert passage_id == "top-level-id"


def test_extract_relevant_passage_id_falls_back_to_annotated_context() -> None:
    passage_id = extract_relevant_passage_id(
        {},
        ground_truth_context=[
            "Passage ID: 1.5-c6-s1\nTitle: Irrelevance of Sentence\n\nCounsel should not...",
        ],
    )

    assert passage_id == "1.5-c6-s1"


def test_retrieval_metrics_match_annotated_retrieved_context_ids() -> None:
    result = compute_legal_rag_retrieval_metrics(
        rag_type="vector_semantic",
        top_k=5,
        relevant_passage_id="1.5-c6-s1",
        response={
            "context": [
                "Passage ID: 1.2-c1-s1\nTitle: Jury Empanelment\n\n...",
                "Passage ID: 1.5-c6-s1\nTitle: Irrelevance of Sentence\n\n...",
            ]
        },
    )

    assert result is not None
    assert result["hit_at_k"] is True
    assert result["hit_at_5"] is True
    assert result["gold_accessed"] is True
    assert result["gold_access_rank"] == 2


def test_retrieval_metrics_match_trace_source_filename_ids() -> None:
    result = compute_legal_rag_retrieval_metrics(
        rag_type="vector_semantic",
        top_k=5,
        relevant_passage_id="1.5-c6-s1",
        response={
            "retrieval_trace": {
                "retrieved_chunks": [
                    {
                        "source": "storage/raw/fd237f78_passage_0035__1_5-c6-s1.txt",
                        "metadata": {"doc_key": "fd237f78_passage_0035__1_5-c6-s1.txt"},
                    }
                ]
            }
        },
    )

    assert result is not None
    assert result["hit_at_k"] is True
    assert result["gold_accessed"] is True
    assert result["gold_access_rank"] == 1


def test_retrieval_metrics_rank_uses_one_id_per_chunk() -> None:
    # Each retrieved chunk exposes BOTH a real source path and a synthetic
    # doc_key. Only one id per chunk may count toward rank, otherwise the gold
    # passage (here the 5th of 5 retrieved) is pushed past top_k and hit@k is
    # wrongly scored a miss.
    chunks = [
        {
            "source": f"storage/raw/aa{i}_passage_000{i}__{pid}.txt",
            "metadata": {"doc_key": f"doc_aa{i}deadbeef"},
        }
        for i, pid in enumerate(
            ["1.1-c1-s1", "2.2-c1-s1", "3.3-c1-s1", "4.4-c1-s1", "1.5-c6-s1"],
            start=1,
        )
    ]
    result = compute_legal_rag_retrieval_metrics(
        rag_type="vector_semantic",
        top_k=5,
        relevant_passage_id="1.5-c6-s1",
        response={"retrieval_trace": {"retrieved_chunks": chunks}},
    )

    assert result is not None
    assert result["gold_access_rank"] == 5
    assert result["hit_at_k"] is True
    assert result["hit_at_5"] is True


def test_grounded_incorrect_without_retrieval_gets_taxonomy_bucket() -> None:
    taxonomy = derive_taxonomy(
        retrieval_metrics=None,
        judge_result={"correct": False, "grounded": True},
    )

    assert taxonomy == "grounded_but_incorrect"


def test_abstention_taxonomy_takes_precedence_over_ungrounded() -> None:
    taxonomy = derive_taxonomy(
        retrieval_metrics={"hit_at_k": True},
        judge_result={"correct": False, "grounded": False, "abstention": True},
    )

    assert taxonomy == "abstention"


def test_ungrounded_without_abstention_stays_hallucination() -> None:
    taxonomy = derive_taxonomy(
        retrieval_metrics=None,
        judge_result={"correct": False, "grounded": False},
    )

    assert taxonomy == "hallucination_or_ungrounded"


def test_summary_reports_abstention_rate_and_counts() -> None:
    summary = summarize_legal_rag_metrics(
        [
            {
                "retrieval": None,
                "judge": {"correct": False, "grounded": False, "abstention": True},
                "taxonomy": "abstention",
            },
            {
                "retrieval": None,
                "judge": {"correct": True, "grounded": True},
                "taxonomy": "success",
            },
        ]
    )

    assert summary is not None
    assert summary["judge"]["abstention_rate"] == 0.5
    assert summary["taxonomy"] == {"abstention": 1, "success": 1}


def test_summary_taxonomy_counts_all_classified_judge_results() -> None:
    summary = summarize_legal_rag_metrics(
        [
            {
                "retrieval": None,
                "judge": {"correct": False, "grounded": True},
                "taxonomy": "grounded_but_incorrect",
            },
            {
                "retrieval": None,
                "judge": {"correct": True, "grounded": True},
                "taxonomy": "success",
            },
            {
                "retrieval": None,
                "judge": {"correct": False, "grounded": False},
                "taxonomy": "hallucination_or_ungrounded",
            },
        ]
    )

    assert summary is not None
    assert summary["count"] == 3
    assert summary["taxonomy"] == {
        "grounded_but_incorrect": 1,
        "success": 1,
        "hallucination_or_ungrounded": 1,
    }
