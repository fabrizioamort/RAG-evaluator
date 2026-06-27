"""Legal RAG Bench retrieval metrics and taxonomy helpers."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

LEGAL_RAG_RETRIEVAL_METRIC = "legal_rag_retrieval"
LEGAL_RAG_BINARY_JUDGE_METRIC = "legal_rag_binary_judge"

_PASSAGE_ID_RE = re.compile(r"passage\s+id\s*:\s*([^\n\r]+)", re.IGNORECASE)
_PATH_ID_SUFFIXES = {
    ".html",
    ".htm",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".txt",
}


def is_legal_rag_metric_enabled(metric_config: dict[str, Any] | None) -> bool:
    metrics = set((metric_config or {}).get("metrics", []))
    return bool(metrics & {LEGAL_RAG_RETRIEVAL_METRIC, LEGAL_RAG_BINARY_JUDGE_METRIC})


def is_legal_rag_judge_enabled(metric_config: dict[str, Any] | None) -> bool:
    return LEGAL_RAG_BINARY_JUDGE_METRIC in set((metric_config or {}).get("metrics", []))


def extract_relevant_passage_id(
    test_case_metadata: dict[str, Any] | None,
    ground_truth_context: list[str] | None = None,
) -> str | None:
    metadata = test_case_metadata if isinstance(test_case_metadata, dict) else {}
    for key in ("relevant_passage_id", "gold_passage_id", "passage_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    benchmark_metadata = metadata.get("benchmark")
    if isinstance(benchmark_metadata, dict):
        value = benchmark_metadata.get("relevant_passage_id")
        if value is not None and str(value).strip():
            return str(value).strip()

    for context in ground_truth_context or []:
        if not isinstance(context, str):
            continue
        match = _PASSAGE_ID_RE.search(context)
        if match:
            return _clean_extracted_identifier(match.group(1))

    return None


def compute_legal_rag_retrieval_metrics(
    *,
    rag_type: str,
    top_k: int,
    relevant_passage_id: str | None,
    response: dict[str, Any],
) -> dict[str, Any] | None:
    if not relevant_passage_id:
        return None

    retrieved_passage_ids = extract_retrieved_passage_ids(response)
    normalized_gold = _normalize_identifier(relevant_passage_id)
    accessed_positions = [
        index
        for index, passage_id in enumerate(retrieved_passage_ids, start=1)
        if _normalize_identifier(passage_id) == normalized_gold
    ]
    gold_access_rank = accessed_positions[0] if accessed_positions else None
    is_filesystem = rag_type == "filesystem_rag"
    hit_at_k = (
        None
        if is_filesystem
        else gold_access_rank is not None and gold_access_rank <= top_k
    )
    gold_accessed = gold_access_rank is not None

    return {
        "benchmark": "legal_rag_bench",
        "relevant_passage_id": relevant_passage_id,
        "retrieved_passage_ids": retrieved_passage_ids,
        "retrieval_metric": "gold_accessed" if is_filesystem else f"hit@{top_k}",
        "hit_at_k": hit_at_k,
        "hit_at_5": None if is_filesystem else gold_access_rank is not None and gold_access_rank <= 5,
        "gold_accessed": gold_accessed,
        "gold_access_rank": gold_access_rank,
        "top_k": top_k,
    }


def extract_retrieved_passage_ids(response: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "sources",
            "context_sources",
            "files_read",
            "retrieved_sources",
            "source_documents",
        ):
            candidates.extend(_extract_ids_from_value(metadata.get(key)))

    for key in ("sources", "context_sources", "retrieved_sources", "retrieval_trace"):
        candidates.extend(_extract_ids_from_value(response.get(key)))

    candidates.extend(_extract_ids_from_value(response.get("context")))
    return _dedupe_preserving_order(candidates)


def derive_taxonomy(
    *,
    retrieval_metrics: dict[str, Any] | None,
    judge_result: dict[str, Any] | None,
) -> str | None:
    if not judge_result:
        return None

    correct = judge_result.get("correct")
    grounded = judge_result.get("grounded")

    # A refusal / non-answer is an abstention, not a hallucination. The judge's
    # deterministic non-answer override forces grounded=False, so this check
    # must come before the grounded=False bucket below.
    if judge_result.get("abstention"):
        return "abstention"

    if correct is True and grounded is True:
        return "success"
    if grounded is False:
        return "hallucination_or_ungrounded"

    retrieval_hit = None
    if retrieval_metrics:
        retrieval_hit = retrieval_metrics.get("hit_at_k")
        if retrieval_hit is None:
            retrieval_hit = retrieval_metrics.get("gold_accessed")

    if grounded is True and correct is False and retrieval_hit is False:
        return "retrieval_error"
    if grounded is True and correct is False and retrieval_hit is True:
        return "reasoning_error"
    if grounded is True and correct is False:
        return "grounded_but_incorrect"
    return None


def summarize_legal_rag_metrics(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None

    retrieval_results = [r for r in results if r.get("retrieval")]
    judge_results = [r for r in results if r.get("judge")]
    taxonomy_results = [r for r in results if r.get("taxonomy")]

    summary: dict[str, Any] = {"count": len(results)}
    if retrieval_results:
        hit_values = [
            r["retrieval"].get("hit_at_k")
            for r in retrieval_results
            if r["retrieval"].get("hit_at_k") is not None
        ]
        gold_values = [
            r["retrieval"].get("gold_accessed")
            for r in retrieval_results
            if r["retrieval"].get("gold_accessed") is not None
        ]
        summary["retrieval"] = {
            "count": len(retrieval_results),
            "hit_at_k_rate": _rate(hit_values),
            "gold_accessed_rate": _rate(gold_values),
        }

    if judge_results:
        correct_values = [r["judge"].get("correct") for r in judge_results]
        grounded_values = [r["judge"].get("grounded") for r in judge_results]
        abstention_values = [bool(r["judge"].get("abstention")) for r in judge_results]
        summary["judge"] = {
            "count": len(judge_results),
            "correct_rate": _rate(correct_values),
            "grounded_rate": _rate(grounded_values),
            "abstention_rate": _rate(abstention_values),
        }

    if taxonomy_results:
        counts: dict[str, int] = {}
        for result in taxonomy_results:
            taxonomy = str(result["taxonomy"])
            counts[taxonomy] = counts.get(taxonomy, 0) + 1
        summary["taxonomy"] = counts

    return summary


def _extract_ids_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _ids_from_string(value)
    if isinstance(value, dict):
        ids: list[str] = []
        for key in (
            "passage_id",
            "relevant_passage_id",
            "id",
            "doc_id",
            "doc_key",
            "source",
            "source_path",
            "filename",
            "path",
        ):
            ids.extend(_extract_ids_from_value(value.get(key)))
        for key in (
            "metadata",
            "payload",
            "document",
            "source_document",
            "retrieved_chunks",
            "chunks",
            "chunk_details",
            "results",
        ):
            ids.extend(_extract_ids_from_value(value.get(key)))
        return ids
    if isinstance(value, list | tuple | set):
        ids: list[str] = []
        for item in value:
            ids.extend(_extract_ids_from_value(item))
        return ids
    return _ids_from_string(str(value))


def _ids_from_string(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    ids = [_clean_extracted_identifier(match.group(1)) for match in _PASSAGE_ID_RE.finditer(text)]
    if ids:
        return ids

    path_text = text.replace("\\", "/")
    return [_identifier_from_path_like_value(path_text)]


def _clean_extracted_identifier(value: str) -> str:
    return re.split(r"\s+title\s*:", value.strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _normalize_identifier(value: str) -> str:
    identifier = _identifier_from_path_like_value(value)
    identifier = re.sub(r"(?<=\d)_(?=\d)", ".", identifier)
    return identifier.casefold()


def _identifier_from_path_like_value(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePath(normalized)
    if "/" in normalized or path.suffix.casefold() in _PATH_ID_SUFFIXES:
        normalized = path.stem or normalized
    if "__" in normalized:
        normalized = normalized.rsplit("__", maxsplit=1)[1]
    return normalized


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = _normalize_identifier(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _rate(values: list[Any]) -> float | None:
    booleans = [value for value in values if isinstance(value, bool)]
    if not booleans:
        return None
    return sum(1 for value in booleans if value) / len(booleans)
