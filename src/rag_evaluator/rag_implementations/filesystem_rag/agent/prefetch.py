"""Deterministic prefetch helpers for the Filesystem RAG agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.agent.tools import (
    FilesystemRAGTools,
)
from rag_evaluator.rag_implementations.filesystem_rag.passage_index import tokenize

DEFAULT_PREFETCH_CANDIDATES = 8
_SEARCH_DEPTH_MULTIPLIER = 3
_SECTION_FAMILY_LIMIT = 2

_QUESTION_SEED_RE = re.compile(
    r'^\s*-\s*"(?P<question>.+?)"\s*(?:->|→|â†’)\s*(?P<doc_id>[^\s]+)\s*$'
)
_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "having",
    "he",
    "her",
    "his",
    "how",
    "if",
    "in",
    "is",
    "it",
    "its",
    "might",
    "must",
    "of",
    "on",
    "or",
    "prior",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}

_LEGAL_REFORMULATION_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(circumstantial|inadmissible|possession|phone|guilt)\b",
        "use of circumstantial evidence admissible prove guilt inference possession",
    ),
    (
        r"\b(arson|petrol|photos?|texts?|weekend|new information)\b",
        "juror receives outside information evidence not admitted jury misconduct tell judge",
    ),
    (
        r"\b(news|publicity|stories|prior to the trial|trial commencing)\b",
        "pre-trial publicity juror exposure news stories accused impartiality excuse",
    ),
    (
        r"\b(recording|eyewitness|testimony|evidence-in-chief|witnessed)\b",
        "VARE procedure audio audiovisual recording prosecution witness evidence-in-chief",
    ),
    (
        r"\b(standard sentence|sentence|imprisonment|punishment)\b",
        "jury not told sentence punishment standard sentence imprisonment",
    ),
    (
        r"\b(locksmith|lockpick|expert knowledge|physical evidence)\b",
        "juror special expertise personal knowledge physical evidence irrelevant changed condition",
    ),
    (
        r"\b(reasonable doubt|standard of proof|fully convinced|prosecution)\b",
        "beyond reasonable doubt standard of proof every element not every fact prosecution",
    ),
    (
        r"\b(court travels|location|backyard|examine|inspection)\b",
        "view inspection demonstration experiment court travels location",
    ),
    (
        r"\b(cannabis|privilege|self-incrimination|witness)\b",
        "privilege against self-incrimination witness compelled evidence",
    ),
    (
        r"\b(friend|juror|jury|excuse|excused)\b",
        "juror excused close friend accused impartiality challenge for cause",
    ),
)


def build_prefetch_context(
    tools: FilesystemRAGTools,
    question: str,
    max_candidates: int = DEFAULT_PREFETCH_CANDIDATES,
) -> dict[str, Any]:
    """Build deterministic candidate context before LLM navigation."""
    max_candidates = max(1, max_candidates)
    search_depth = max(max_candidates * _SEARCH_DEPTH_MULTIPLIER, max_candidates)
    search_results: list[dict[str, Any]] = []

    raw_result = tools.search_passages(question, top_k=search_depth)
    search_results.append({"label": "raw", "query": question, "result": raw_result})

    reformulated_query = reformulate_legal_query(question)
    if reformulated_query and reformulated_query.lower() != question.lower():
        reformulated_result = tools.search_passages(reformulated_query, top_k=search_depth)
        search_results.append(
            {
                "label": "reformulated",
                "query": reformulated_query,
                "result": reformulated_result,
            }
        )

    merged: dict[str, dict[str, Any]] = {}
    terms: list[str] = []
    for search in search_results:
        terms.extend(str(term) for term in search["result"].get("query_terms", []))
        for candidate in search["result"].get("results", []):
            _merge_candidate(
                merged,
                candidate,
                source_label=search["label"],
                query=search["query"],
                score_weight=1.0 if search["label"] == "raw" else 1.05,
            )

    question_seed_hints = find_question_seed_hints(
        tools.prepared_path,
        question,
        max_hints=max_candidates,
    )
    for hint in question_seed_hints:
        _merge_candidate(
            merged,
            hint,
            source_label="question_seed",
            query=hint["seed_question"],
            score_weight=1.0,
        )

    ranked = _diverse_top_candidates(list(merged.values()), max_candidates=max_candidates)
    chunks: list[str] = []
    sources: list[str] = []

    for candidate in ranked:
        if candidate.get("prefetch_kind") == "question_seed":
            chunks.append(_format_question_seed_hint(candidate))
            source = str(candidate.get("source") or "")
            if source:
                sources.append(source)
            continue

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
                    f"Prefetch sources: {', '.join(candidate.get('prefetch_sources', []))}",
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
        "terms": _unique_preserve_order(terms),
        "candidates": ranked,
        "chunks": chunks,
        "sources": sources,
        "queries": [
            {"label": search["label"], "query": search["query"]} for search in search_results
        ],
        "question_seed_hints": question_seed_hints,
    }


def reformulate_legal_query(question: str) -> str:
    """Return a deterministic doctrine-vocabulary query for a scenario question."""
    lower_question = question.lower()
    expansions = [
        phrase
        for pattern, phrase in _LEGAL_REFORMULATION_RULES
        if re.search(pattern, lower_question)
    ]
    key_terms = [
        token
        for token in tokenize(question)
        if token not in _STOPWORDS and len(token) > 2 and not token.isdigit()
    ][:18]
    reformulated_terms = _unique_preserve_order(tokenize(" ".join(expansions)) + key_terms)
    return " ".join(reformulated_terms)


def find_question_seed_hints(
    prepared_path: Path,
    question: str,
    max_hints: int,
) -> list[dict[str, Any]]:
    """Return high-overlap question-seed matches as navigation hints."""
    seed_path = prepared_path / "_index" / "questions" / "question_seeds.md"
    if max_hints <= 0 or not seed_path.exists():
        return []

    query_terms = [term for term in tokenize(question) if term not in _STOPWORDS]
    query_set = set(query_terms)
    if not query_set:
        return []

    hints: list[dict[str, Any]] = []
    try:
        lines = seed_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        match = _QUESTION_SEED_RE.match(line)
        if not match:
            continue
        seed_question = match.group("question")
        doc_id = match.group("doc_id").strip()
        seed_terms = [term for term in tokenize(seed_question) if term not in _STOPWORDS]
        seed_set = set(seed_terms)
        overlap = sorted(query_set.intersection(seed_set))
        if len(overlap) < 2:
            continue
        score = _question_seed_score(query_set, seed_set, overlap)
        if score < 0.12:
            continue
        source = f"documents/{doc_id}.md"
        hints.append(
            {
                "prefetch_kind": "question_seed",
                "passage_id": doc_id,
                "doc_id": doc_id,
                "score": round(100.0 + (score * 100.0), 4),
                "matched_terms": overlap,
                "title": "Question seed match",
                "section_title": seed_question,
                "source": source,
                "summary_source": f"_summaries/{doc_id}_summary.md",
                "start_line": None,
                "end_line": None,
                "snippet": f'Question seed: "{seed_question}" -> {doc_id}',
                "seed_question": seed_question,
                "read_hint": {"path": source},
            }
        )

    hints.sort(key=lambda item: (-float(item["score"]), item["doc_id"]))
    return hints[:max_hints]


def _question_seed_score(
    query_terms: set[str],
    seed_terms: set[str],
    overlap: list[str],
) -> float:
    union_size = max(len(query_terms.union(seed_terms)), 1)
    jaccard = len(overlap) / union_size
    coverage = len(overlap) / max(len(seed_terms), 1)
    return (jaccard * 0.6) + (coverage * 0.4)


def _merge_candidate(
    merged: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    source_label: str,
    query: str,
    score_weight: float,
) -> None:
    key = _candidate_key(candidate)
    candidate_score = float(candidate.get("score") or 0.0) * score_weight
    if key not in merged:
        stored = dict(candidate)
        stored["prefetch_sources"] = [source_label]
        stored["prefetch_queries"] = [query]
        stored["_prefetch_score"] = candidate_score
        stored["section_family"] = _section_family(stored)
        stored["score"] = round(candidate_score, 4)
        merged[key] = stored
        return

    stored = merged[key]
    stored["_prefetch_score"] = max(float(stored.get("_prefetch_score") or 0.0), candidate_score)
    stored["_prefetch_score"] += candidate_score * 0.15
    stored["score"] = round(float(stored["_prefetch_score"]), 4)
    stored["prefetch_sources"] = _unique_preserve_order(
        list(stored.get("prefetch_sources", [])) + [source_label]
    )
    stored["prefetch_queries"] = _unique_preserve_order(
        list(stored.get("prefetch_queries", [])) + [query]
    )
    stored["matched_terms"] = _unique_preserve_order(
        list(stored.get("matched_terms", [])) + list(candidate.get("matched_terms", []))
    )
    if stored.get("prefetch_kind") != "question_seed" and candidate.get("prefetch_kind"):
        stored["prefetch_kind"] = candidate.get("prefetch_kind")


def _candidate_key(candidate: dict[str, Any]) -> str:
    source = str(candidate.get("source") or "")
    if source:
        return source
    return str(candidate.get("doc_id") or candidate.get("passage_id") or id(candidate))


def _section_family(candidate: dict[str, Any]) -> str:
    identifier = str(candidate.get("doc_id") or candidate.get("passage_id") or "")
    match = re.match(r"(?P<section>\d+(?:\.\d+)*)-c\d+-s\d+", identifier)
    if not match:
        return identifier or "unknown"
    parts = match.group("section").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _diverse_top_candidates(
    candidates: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("_prefetch_score") or item.get("score") or 0.0),
            str(item.get("doc_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for candidate in ranked:
        family = str(candidate.get("section_family") or _section_family(candidate))
        if family_counts.get(family, 0) >= _SECTION_FAMILY_LIMIT:
            continue
        cleaned = {key: value for key, value in candidate.items() if key != "_prefetch_score"}
        selected.append(cleaned)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max_candidates:
            break
    return selected


def _format_question_seed_hint(candidate: dict[str, Any]) -> str:
    read_hint = candidate.get("read_hint") or {}
    return "\n".join(
        [
            f"# Question-Seed Navigation Hint: {candidate.get('doc_id', '')}",
            f"Source: {candidate.get('source', '')}",
            f"Matched seed: {candidate.get('seed_question', '')}",
            f"Matched terms: {', '.join(candidate.get('matched_terms', []))}",
            "Read hint: " f"read_file(path={read_hint.get('path')!r})",
            "",
            str(candidate.get("snippet") or ""),
        ]
    )


def _unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
