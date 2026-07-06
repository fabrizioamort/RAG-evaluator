"""BM25 passage indexing for Filesystem RAG.

The prepared filesystem is the durable retrieval artifact for this RAG
implementation, so the BM25 index is stored as JSON under ``_index/passages``.
Keeping the implementation local avoids adding another runtime dependency for
a small, deterministic scorer.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
        DocumentAnalysis,
    )
    from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
        ProcessedDocument,
    )

BM25_INDEX_RELATIVE_PATH = Path("_index") / "passages" / "bm25.json"
BM25_INDEX_VERSION = 1
BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:[-'][a-zA-Z0-9]+)*")
_SNIPPET_MAX_CHARS = 1400


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _normalize_token(token: str) -> str:
    """Normalize lexical tokens with a light suffix stemmer."""
    token = token.lower().strip("'")
    if len(token) <= 3:
        return token

    if token.endswith("'s") and len(token) > 4:
        token = token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        base = token[:-3]
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        return base
    if token.endswith("ed") and len(token) > 5:
        base = token[:-2]
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        return base
    if token.endswith(("ches", "shes", "xes", "zes", "ses")) and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text for BM25 scoring."""
    return [_normalize_token(match.group(0)) for match in _TOKEN_RE.finditer(text)]


def _line_range(lines: list[str], start_line: int, end_line: int) -> str:
    """Return inclusive 1-indexed line range text."""
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def _safe_section_bounds(section: dict[str, Any], total_lines: int) -> tuple[int, int]:
    """Return clamped inclusive section line bounds."""
    start = int(section.get("start_line") or 1)
    end = int(section.get("end_line") or total_lines)
    start = min(max(start, 1), max(total_lines, 1))
    end = min(max(end, start), max(total_lines, 1))
    return start, end


def _analysis_search_text(analysis: DocumentAnalysis) -> str:
    """Return document-level analysis text that improves lexical recall."""
    entity_values: list[str] = []
    for values in analysis.entities.values():
        entity_values.extend(values)

    return "\n".join(
        [
            analysis.summary,
            "\n".join(analysis.question_seeds),
            "\n".join(analysis.topics),
            "\n".join(entity_values),
        ]
    )


def _document_passages(
    doc: ProcessedDocument,
    analysis: DocumentAnalysis,
) -> list[dict[str, Any]]:
    """Create section-level passage records for one processed document."""
    lines = doc.markdown_content.splitlines()
    total_lines = max(len(lines), 1)
    sections = doc.sections or [
        {
            "title": doc.title,
            "start_line": 1,
            "end_line": total_lines,
            "level": 1,
        }
    ]
    analysis_text = _analysis_search_text(analysis)

    passages: list[dict[str, Any]] = []
    for index, section in enumerate(sections, start=1):
        start_line, end_line = _safe_section_bounds(section, total_lines)
        section_title = str(section.get("title") or doc.title)
        passage_text = _line_range(lines, start_line, end_line)
        search_text = "\n".join([doc.title, section_title, passage_text, analysis_text])
        term_counts = Counter(tokenize(search_text))
        if len(term_counts) < 5:
            continue

        passages.append(
            {
                "passage_id": f"{doc.id}#L{start_line}-L{end_line}",
                "doc_id": doc.id,
                "title": doc.title,
                "section_title": section_title,
                "section_index": index,
                "source": f"documents/{doc.id}.md",
                "summary_source": f"_summaries/{doc.id}_summary.md",
                "start_line": start_line,
                "end_line": end_line,
                "line_count": end_line - start_line + 1,
                "length": sum(term_counts.values()),
                "term_counts": dict(sorted(term_counts.items())),
                "preview": passage_text[:_SNIPPET_MAX_CHARS].strip(),
            }
        )

    if passages:
        return passages

    fallback_text = "\n".join([doc.title, doc.markdown_content, analysis_text])
    term_counts = Counter(tokenize(fallback_text))
    return [
        {
            "passage_id": f"{doc.id}#L1-L{total_lines}",
            "doc_id": doc.id,
            "title": doc.title,
            "section_title": doc.title,
            "section_index": 1,
            "source": f"documents/{doc.id}.md",
            "summary_source": f"_summaries/{doc.id}_summary.md",
            "start_line": 1,
            "end_line": total_lines,
            "line_count": total_lines,
            "length": sum(term_counts.values()),
            "term_counts": dict(sorted(term_counts.items())),
            "preview": doc.markdown_content[:_SNIPPET_MAX_CHARS].strip(),
        }
    ]


def build_bm25_payload(documents: list[tuple[ProcessedDocument, DocumentAnalysis]]) -> dict[str, Any]:
    """Build a serializable BM25 index payload."""
    passages: list[dict[str, Any]] = []
    for doc, analysis in documents:
        passages.extend(_document_passages(doc, analysis))

    document_frequencies: Counter[str] = Counter()
    for passage in passages:
        document_frequencies.update(passage["term_counts"].keys())

    total_length = sum(int(passage["length"]) for passage in passages)
    avg_doc_length = total_length / len(passages) if passages else 0.0

    return {
        "version": BM25_INDEX_VERSION,
        "kind": "bm25_passage",
        "parameters": {"k1": BM25_K1, "b": BM25_B},
        "tokenizer": {
            "version": 1,
            "normalization": "lowercase_light_suffix_stemmer",
        },
        "passage_count": len(passages),
        "avg_doc_length": avg_doc_length,
        "document_frequencies": dict(sorted(document_frequencies.items())),
        "passages": passages,
    }


def build_bm25_passage_index(
    documents: list[tuple[ProcessedDocument, DocumentAnalysis]],
    output_path: Path,
) -> dict[str, Any]:
    """Build and persist the BM25 passage index."""
    payload = build_bm25_payload(documents)
    index_path = output_path / BM25_INDEX_RELATIVE_PATH
    _atomic_write_json(index_path, payload)
    print(f"  Created: {BM25_INDEX_RELATIVE_PATH.as_posix()}")
    return payload


class BM25PassageIndex:
    """Runtime searcher for a prepared BM25 passage index."""

    def __init__(self, prepared_path: Path, payload: dict[str, Any]) -> None:
        self.prepared_path = prepared_path
        self.payload = payload
        self.passages: list[dict[str, Any]] = payload.get("passages", [])
        self.document_frequencies: dict[str, int] = payload.get("document_frequencies", {})
        self.avg_doc_length = float(payload.get("avg_doc_length") or 0.0)
        params = payload.get("parameters", {})
        self.k1 = float(params.get("k1", BM25_K1))
        self.b = float(params.get("b", BM25_B))

    @classmethod
    def load(cls, prepared_path: str | Path) -> BM25PassageIndex:
        """Load an existing prepared BM25 passage index."""
        root = Path(prepared_path)
        index_path = root / BM25_INDEX_RELATIVE_PATH
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 passage index not found at {BM25_INDEX_RELATIVE_PATH.as_posix()}"
            )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return cls(root, payload)

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return ranked passages for a query."""
        normalized_to_display: dict[str, str] = {}
        for match in _TOKEN_RE.finditer(query):
            raw_term = match.group(0).lower()
            normalized_to_display.setdefault(_normalize_token(raw_term), raw_term)
        unique_terms = list(normalized_to_display)
        if not unique_terms or not self.passages or self.avg_doc_length <= 0:
            return {
                "query": query,
                "query_terms": list(normalized_to_display.values()),
                "results": [],
            }

        total_passages = len(self.passages)
        scored: list[tuple[float, dict[str, Any], list[str]]] = []

        for passage in self.passages:
            term_counts = passage.get("term_counts", {})
            passage_length = float(passage.get("length") or 0)
            if passage_length <= 0:
                continue

            score = 0.0
            matched_terms: list[str] = []
            for term in unique_terms:
                term_frequency = int(term_counts.get(term, 0))
                if term_frequency <= 0:
                    continue

                doc_frequency = int(self.document_frequencies.get(term, 0))
                idf = math.log(1 + ((total_passages - doc_frequency + 0.5) / (doc_frequency + 0.5)))
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * (passage_length / self.avg_doc_length)
                )
                score += idf * ((term_frequency * (self.k1 + 1)) / denominator)
                matched_terms.append(normalized_to_display[term])

            if score > 0:
                scored.append((score, passage, matched_terms))

        ranked_all = sorted(
            scored,
            key=lambda item: (
                item[0],
                -int(item[1].get("start_line") or 0),
                item[1].get("passage_id", ""),
            ),
            reverse=True,
        )

        # Keep only the best-ranked section per document so top_k covers
        # distinct documents instead of many windows of one strong file.
        best_by_doc: dict[str, tuple[float, dict[str, Any], list[str]]] = {}
        extra_counts: dict[str, int] = {}
        for item in ranked_all:
            doc_id = str(item[1].get("doc_id") or item[1].get("passage_id"))
            if doc_id in best_by_doc:
                extra_counts[doc_id] = extra_counts.get(doc_id, 0) + 1
                continue
            best_by_doc[doc_id] = item
        deduped = list(best_by_doc.values())[: max(top_k, 0)]

        return {
            "query": query,
            "query_terms": list(normalized_to_display.values()),
            "results": [
                self._format_result(
                    score,
                    passage,
                    matched_terms,
                    other_matching_sections=extra_counts.get(
                        str(passage.get("doc_id") or passage.get("passage_id")), 0
                    ),
                )
                for score, passage, matched_terms in deduped
            ],
        }

    def _format_result(
        self,
        score: float,
        passage: dict[str, Any],
        matched_terms: list[str],
        other_matching_sections: int = 0,
    ) -> dict[str, Any]:
        """Return a compact result object with reread hints."""
        source = str(passage.get("source", ""))
        start_line = int(passage.get("start_line") or 1)
        end_line = int(passage.get("end_line") or start_line)
        snippet = self._build_snippet(source, start_line, end_line, matched_terms)
        return {
            "passage_id": passage.get("passage_id"),
            "doc_id": passage.get("doc_id"),
            "score": round(score, 4),
            "matched_terms": matched_terms,
            "other_matching_sections": other_matching_sections,
            "title": passage.get("title"),
            "section_title": passage.get("section_title"),
            "source": source,
            "summary_source": passage.get("summary_source"),
            "start_line": start_line,
            "end_line": end_line,
            "snippet": snippet,
            "read_hint": {
                "path": source,
                "start_line": start_line,
                "end_line": end_line,
            },
        }

    def _build_snippet(
        self,
        source: str,
        start_line: int,
        end_line: int,
        matched_terms: list[str],
    ) -> str:
        """Build a focused snippet for a result."""
        source_path = (self.prepared_path / source).resolve()
        try:
            source_path.relative_to(self.prepared_path.resolve())
        except ValueError:
            return ""
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return ""

        passage_lines = lines[max(0, start_line - 1) : min(len(lines), end_line)]
        if not passage_lines:
            return ""

        if not matched_terms:
            return "\n".join(passage_lines)[:_SNIPPET_MAX_CHARS].strip()

        normalized_terms = set(tokenize(" ".join(matched_terms)))
        hit_indices: list[int] = []
        for index, line in enumerate(passage_lines):
            if normalized_terms.intersection(tokenize(line)):
                hit_indices.append(index)

        if not hit_indices:
            return "\n".join(passage_lines)[:_SNIPPET_MAX_CHARS].strip()

        ranges: list[tuple[int, int]] = []
        for hit_index in hit_indices[:6]:
            start = max(0, hit_index - 3)
            end = min(len(passage_lines), hit_index + 4)
            if ranges and start <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))

        parts = ["\n".join(passage_lines[start:end]) for start, end in ranges]
        snippet = "\n\n...\n\n".join(parts)
        if len(snippet) > _SNIPPET_MAX_CHARS:
            snippet = snippet[:_SNIPPET_MAX_CHARS].rstrip() + "\n... [passage snippet truncated]"
        return snippet.strip()
