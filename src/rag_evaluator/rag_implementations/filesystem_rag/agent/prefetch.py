"""Deterministic prefetch helpers for the Filesystem RAG agent."""

from __future__ import annotations

from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import (
    FilesystemRAGTools,
)


def build_prefetch_context(
    tools: FilesystemRAGTools,
    question: str,
    max_candidates: int = 3,
) -> dict[str, Any]:
    """Build deterministic BM25 candidate context before LLM navigation."""
    search_result = tools.search_passages(question, top_k=max_candidates)
    terms = search_result.get("query_terms", [])
    ranked = search_result.get("results", [])
    chunks: list[str] = []
    sources: list[str] = []

    for candidate in ranked:
        source = str(candidate.get("source") or "")
        doc_id = str(candidate.get("doc_id") or "")
        snippet = str(candidate.get("snippet") or "")
        read_hint = candidate.get("read_hint") or {}

        chunks.append(
            "\n".join(
                [
                    f"# Candidate BM25 Passage: {candidate.get('passage_id', doc_id)}",
                    f"Source: {source}",
                    f"Title: {candidate.get('title', '')}",
                    f"Section: {candidate.get('section_title', '')}",
                    f"Score: {candidate.get('score', 0)}",
                    f"Lines: {candidate.get('start_line')}-{candidate.get('end_line')}",
                    f"Matched terms: {', '.join(candidate.get('matched_terms', []))}",
                    "Read hint: "
                    f"read_file(path={read_hint.get('path')!r}, "
                    f"start_line={read_hint.get('start_line')}, "
                    f"end_line={read_hint.get('end_line')})",
                    "",
                    snippet,
                ]
            )
        )
        if source:
            sources.append(source)

    return {
        "terms": terms,
        "candidates": ranked,
        "chunks": chunks,
        "sources": sources,
    }
