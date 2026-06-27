"""Article-ready exports for Legal RAG Bench comparisons.

Pure formatting helpers (no DB access) that turn normalized evaluation members
into CSV / Markdown headline + taxonomy tables and per-question JSONL. The API
layer assembles ``ExportMember`` instances and streams the rendered output.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

RETRIEVAL_MODE_LABELS = {
    "vector_semantic": "dense vector",
    "vector_hybrid": "dense + sparse",
    "graph_rag": "graph traversal",
    "filesystem_rag": "agentic file search",
}

TAXONOMY_ORDER = [
    ("success", "Success"),
    ("hallucination_or_ungrounded", "Hallucination / ungrounded"),
    ("retrieval_error", "Retrieval error"),
    ("reasoning_error", "Reasoning error"),
    ("abstention", "Abstention"),
]

HEADLINE_COLUMNS = [
    "System",
    "Retrieval mode",
    "Retrieval metric",
    "Retrieval score",
    "Correct",
    "Grounded",
    "RAG accuracy",
    "Avg latency",
    "Notes",
]


@dataclass
class ExportMember:
    """One evaluation normalized for export."""

    label: str
    rag_config_name: str | None = None
    rag_type: str | None = None
    pass_rate: float | None = None
    summary_metrics: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None
    legal_rag_bench: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    notes: str = ""


def _retrieval_mode(rag_type: str | None) -> str:
    if not rag_type:
        return "—"
    return RETRIEVAL_MODE_LABELS.get(rag_type, rag_type)


def _retrieval_metric_and_score(legal: dict[str, Any] | None) -> tuple[str, float | None]:
    retrieval = (legal or {}).get("retrieval") or {}
    hit = retrieval.get("hit_at_k_rate")
    gold = retrieval.get("gold_accessed_rate")
    if hit is not None:
        return "hit@5", hit
    if gold is not None:
        return "gold_accessed", gold
    return "—", None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _secs(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def headline_rows(members: list[ExportMember]) -> list[dict[str, str]]:
    """Build the headline metrics table rows (plan section 12)."""
    rows: list[dict[str, str]] = []
    for m in members:
        legal = m.legal_rag_bench or {}
        judge = legal.get("judge") or {}
        metric, score = _retrieval_metric_and_score(legal)
        avg_latency = (m.performance_metrics or {}).get("avg_latency_seconds")
        rows.append(
            {
                "System": m.label,
                "Retrieval mode": _retrieval_mode(m.rag_type),
                "Retrieval metric": metric,
                "Retrieval score": _pct(score),
                "Correct": _pct(judge.get("correct_rate")),
                "Grounded": _pct(judge.get("grounded_rate")),
                "RAG accuracy": _pct(m.pass_rate),
                "Avg latency": _secs(avg_latency),
                "Notes": m.notes or (m.rag_config_name or ""),
            }
        )
    return rows


def taxonomy_rows(members: list[ExportMember]) -> list[dict[str, str]]:
    """Build the taxonomy breakdown table rows (counts per category)."""
    rows: list[dict[str, str]] = []
    for m in members:
        taxonomy = (m.legal_rag_bench or {}).get("taxonomy") or {}
        row: dict[str, str] = {"System": m.label}
        for key, label in TAXONOMY_ORDER:
            row[label] = str(taxonomy.get(key, 0))
        rows.append(row)
    return rows


def taxonomy_columns() -> list[str]:
    return ["System", *(label for _, label in TAXONOMY_ORDER)]


def to_csv(rows: list[dict[str, str]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def to_markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows
    ]
    return "\n".join([header, sep, *body])


def _manifest_section(member: ExportMember) -> list[str]:
    manifest = member.manifest or {}
    lines = [f"### {member.label}", ""]
    lines.append(f"- RAG type: `{member.rag_type or '—'}`")
    if member.rag_config_name:
        lines.append(f"- RAG config: {member.rag_config_name}")
    for label, key in (
        ("Generation model", "generation_model"),
        ("Judge model", "eval_judge_model"),
        ("rag-evaluator version", "rag_evaluator_version"),
        ("Platform version", "platform_version"),
    ):
        if manifest.get(key):
            lines.append(f"- {label}: `{manifest[key]}`")
    for label, key in (
        ("KB version", "kb_version_snapshot"),
        ("Query overrides", "query_overrides"),
        ("Effective config", "effective_config_snapshot"),
        ("Build config", "build_config_snapshot"),
        ("RAG config snapshot", "rag_config_snapshot"),
    ):
        value = manifest.get(key)
        if value:
            lines.append(f"- {label}:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
    lines.append("")
    return lines


def build_markdown_report(
    members: list[ExportMember],
    *,
    title: str = "Legal RAG Bench — Architecture Comparison",
    generated_at: datetime | None = None,
) -> str:
    """Full article-ready Markdown: headline + taxonomy tables + manifests."""
    stamp = (generated_at or datetime.now(timezone.utc)).isoformat()
    parts = [
        f"# {title}",
        "",
        f"_Generated {stamp}_",
        "",
        "## Headline metrics",
        "",
        to_markdown_table(headline_rows(members), HEADLINE_COLUMNS),
        "",
        "## Taxonomy",
        "",
        to_markdown_table(taxonomy_rows(members), taxonomy_columns()),
        "",
        "## Run manifests and config snapshots",
        "",
    ]
    for member in members:
        parts.extend(_manifest_section(member))
    return "\n".join(parts).rstrip() + "\n"


def per_question_jsonl(records: list[dict[str, Any]]) -> str:
    """Serialize per-question records as JSONL (one JSON object per line)."""
    return "".join(json.dumps(record, default=str) + "\n" for record in records)


def build_question_record(
    *,
    member_label: str,
    evaluation_id: str,
    rag_type: str | None,
    rag_config_name: str | None,
    question: str | None,
    expected_answer: str | None,
    generated_answer: str | None,
    scores: dict[str, Any],
    latency_seconds: float | None,
    legal_rag_bench: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one reproducibility record for the per-question JSONL export."""
    legal = legal_rag_bench or {}
    return {
        "evaluation_id": evaluation_id,
        "system": member_label,
        "rag_type": rag_type,
        "rag_config_name": rag_config_name,
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "latency_seconds": latency_seconds,
        "scores": scores,
        "legal_retrieval": legal.get("retrieval"),
        "legal_judge": legal.get("judge"),
        "legal_taxonomy": legal.get("taxonomy"),
    }
