"""Corpus-level synthesis for Filesystem RAG preparation pipeline.

This module generates meta-level files that provide an overview
of the entire corpus and navigation guidance.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag_evaluator.common.llm_utils import get_safe_llm_params
from rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder import (
    DocumentInfo,
)


def generate_corpus_overview(
    documents: list[DocumentInfo],
    output_path: Path,
    use_llm: bool = False,
    client: OpenAI | None = None,
) -> str:
    """Generate the corpus overview file.

    Creates:
    - _meta/corpus_overview.md

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem
        use_llm: Whether to use LLM for synthesis (optional, costs ~$0.10-0.20)
        client: Optional OpenAI client

    Returns:
        Generated corpus overview content
    """
    meta_dir = output_path / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if use_llm and client:
        content = _generate_corpus_overview_llm(documents, client)
    else:
        content = _generate_corpus_overview_heuristic(documents)

    (meta_dir / "corpus_overview.md").write_text(content, encoding="utf-8")
    print("  Created: _meta/corpus_overview.md")

    return content


def _generate_corpus_overview_heuristic(documents: list[DocumentInfo]) -> str:
    """Generate corpus overview using heuristic methods."""
    # Calculate statistics
    total_docs = len(documents)
    total_words = sum(d.doc.word_count for d in documents)

    # Aggregate topics
    topic_counts: Counter[str] = Counter()
    for doc_info in documents:
        if doc_info.analysis.topics:
            topic_counts.update(str(topic).strip().lower() for topic in doc_info.analysis.topics[:5])
        else:
            for topic, score in doc_info.analysis.topic_scores.items():
                if score >= 0.3:
                    topic_counts[str(topic).strip().lower()] += 1

    # Get top topics
    top_topics = [topic for topic, _ in topic_counts.most_common(4)]

    # Aggregate formats
    format_counts: Counter[str] = Counter()
    for doc_info in documents:
        format_counts[doc_info.doc.original_format] += 1

    # Find date range
    dates = [d.doc.modified_date for d in documents if d.doc.modified_date]
    date_range = ""
    if dates:
        dates.sort()
        date_range = f"{dates[0]} - {dates[-1]}"

    # Build content
    lines = [
        "# Corpus Overview",
        "",
        "## Description",
        f"This corpus contains **{total_docs}** documents covering topics related to "
        f"{', '.join(top_topics[:3]) if top_topics else 'various subjects'}.",
        "",
        f"The documents contain approximately **{total_words:,}** words total "
        f"and include content in {', '.join(format_counts.keys())} format(s).",
        "",
        "## Scope",
        f"- **Primary topics:** {', '.join(top_topics) if top_topics else 'Various'}",
    ]

    if date_range:
        lines.append(f"- **Time range:** {date_range}")

    lines.extend(
        [
            f"- **Document types:** {', '.join(format_counts.keys())}",
            "",
            "## Quick Navigation",
            "1. For direct retrieval: Use `_index/passages/bm25.json` through `search_passages`",
            "2. For question matching: See `_index/questions/question_seeds.md`",
            "3. For topic/entity browsing: Use `_index/topics/` and `_index/entities/`",
            "4. For temporal queries: See `_index/temporal/timeline.md`",
            "",
            "## Key Statistics",
            f"- **Total documents:** {total_docs}",
            f"- **Total words:** ~{total_words:,}",
            "- **Primary language:** English",
            f"- **Last updated:** {datetime.now().strftime('%Y-%m-%d')}",
        ]
    )

    return "\n".join(lines)


CORPUS_SYNTHESIS_PROMPT = """Based on the following document summaries and metadata, create a corpus overview.

DOCUMENTS:
{documents_info}

Create a comprehensive overview that includes:

1. A high-level description of what this corpus contains (2-3 paragraphs)
2. The main themes and topics covered
3. The scope and any limitations
4. Recommended navigation strategies for different query types

Format your response as markdown with clear sections.
Start with "# Corpus Overview" as the title.
Include sections: ## Description, ## Scope, ## Quick Navigation, ## Key Statistics"""


def _generate_corpus_overview_llm(
    documents: list[DocumentInfo],
    client: OpenAI,
) -> str:
    """Generate corpus overview using LLM synthesis."""
    # Build document info summary
    doc_summaries: list[str] = []
    for doc_info in documents[:20]:  # Limit to first 20 docs to control cost
        summary = doc_info.analysis.summary[:200]
        topics = ", ".join(doc_info.analysis.topics[:3])
        doc_summaries.append(
            f"- {doc_info.doc.id} ({doc_info.doc.title}): {summary}... Topics: {topics}"
        )

    documents_info = "\n".join(doc_summaries)
    prompt = CORPUS_SYNTHESIS_PROMPT.format(documents_info=documents_info)

    try:
        kwargs: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        kwargs = get_safe_llm_params("gpt-4o-mini", temperature=0.3, **kwargs)

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
    except Exception as e:
        print(f"  Warning: LLM synthesis failed: {e}")

    # Fall back to heuristic
    return _generate_corpus_overview_heuristic(documents)


def generate_navigation_guide(
    output_path: Path,
) -> str:
    """Generate the navigation guide file.

    Creates:
    - _meta/navigation_guide.md

    Args:
        output_path: Base path for prepared filesystem

    Returns:
        Generated navigation guide content
    """
    meta_dir = output_path / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    content = """# Navigation Guide

## Index Structure
- `_index/passages/` - BM25 passage index for ranked retrieval
- `_index/topics/` - Documents organized by extracted corpus-specific topics
- `_index/entities/` - Extracted entities grouped by analyzer-provided type
- `_index/temporal/` - Timeline of events and dates
- `_index/questions/` - Questions each document can answer

## Recommended Navigation Flow
1. Read `_meta/corpus_overview.md` to understand scope
2. Based on query type:
   - **Direct question** -> `search_passages`
   - **Known phrase** -> `grep_search` with `match_all_terms=True`
   - **Topical query** -> `_index/topics/_topic_map.md`
   - **Entity query** -> `_index/entities/_entity_registry.md`
   - **Temporal query** -> `_index/temporal/timeline.md`
3. Drill down to specific topic/entity files
4. Read document summaries before full documents
5. Read specific sections of full documents as needed

## File Naming Convention
- `doc_XXX.md` - Converted document content for generic corpora
- `<passage-id>.md` - Converted document content when a source filename embeds a passage id
- `*.meta.json` - Structured metadata
- `*_summary.md` - Human-readable summary

## Directory Structure
```
_meta/           → This guide and corpus overview
_index/          → Discovery indexes (start here)
  topics/        → Topic-based navigation
  entities/      → Entity-based navigation
  temporal/      → Time-based navigation
  questions/     → Question-to-document mapping
_summaries/      → Concise document summaries
documents/       → Full document content with metadata
_original/       → Original source files (if preserved)
```

## Query Strategies

### For Specific Lookups
1. Use `search_passages` with the direct question
2. Check `_index/questions/question_seeds.md` for direct matches
3. Use `grep_search` on `documents/` for exact multi-term lookups
4. Navigate to specific document and section

### For Exploratory Queries
1. Start with `search_passages` using the issue phrased broadly
2. Use `_index/topics/_topic_map.md` to inspect extracted topic clusters
3. Read summaries of primary documents
4. Dive into full documents as needed

### For Entity-Based Queries
1. Check `_index/entities/_entity_registry.md`
2. Navigate to specific entity type file
3. Find all documents mentioning the entity

## Tips
- Always read summaries before full documents
- Use metadata JSON for precise section locations
- BM25 scores rank lexical query matches; topic labels are browsing aids
- Question seeds show what each document can answer
"""

    (meta_dir / "navigation_guide.md").write_text(content, encoding="utf-8")
    print("  Created: _meta/navigation_guide.md")

    return content


def generate_statistics(
    documents: list[DocumentInfo],
    output_path: Path,
    preparation_time: float = 0.0,
    preparation_cost: float = 0.0,
) -> dict[str, Any]:
    """Generate the statistics JSON file.

    Creates:
    - _meta/statistics.json

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem
        preparation_time: Total preparation time in seconds
        preparation_cost: Estimated LLM cost for preparation

    Returns:
        Statistics dictionary
    """
    meta_dir = output_path / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_words = sum(d.doc.word_count for d in documents)
    total_chars = sum(d.doc.char_count for d in documents)

    # Documents by format
    format_counts: Counter[str] = Counter()
    for doc_info in documents:
        format_counts[doc_info.doc.original_format] += 1

    # Documents by topic
    topic_counts: Counter[str] = Counter()
    for doc_info in documents:
        if doc_info.analysis.topics:
            topic_counts.update(str(topic).strip().lower() for topic in doc_info.analysis.topics[:5])
        else:
            for topic, score in doc_info.analysis.topic_scores.items():
                if score >= 0.3:
                    topic_counts[str(topic).strip().lower()] += 1

    # Entity counts
    entity_counts: dict[str, int] = {}
    for doc_info in documents:
        for entity_type, entities in doc_info.analysis.entities.items():
            key = str(entity_type).strip().lower()
            entity_counts[key] = entity_counts.get(key, 0) + len(entities)

    # Analysis method distribution
    analysis_methods: Counter[str] = Counter()
    for doc_info in documents:
        analysis_methods[doc_info.analysis.analysis_method] += 1

    statistics = {
        "generated_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "total_words": total_words,
        "total_characters": total_chars,
        "documents_by_format": dict(format_counts),
        "documents_by_topic": dict(topic_counts),
        "total_entities": entity_counts,
        "analysis_methods": dict(analysis_methods),
        "preparation_time_seconds": round(preparation_time, 2),
        "preparation_cost_usd": round(preparation_cost, 4),
    }

    stats_path = meta_dir / "statistics.json"
    stats_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print("  Created: _meta/statistics.json")

    return statistics


def copy_original_files(
    documents: list[DocumentInfo],
    output_path: Path,
) -> None:
    """Copy original files to _original directory for reference.

    Creates:
    - _original/{original_filename}

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem
    """
    original_dir = output_path / "_original"
    original_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    for doc_info in documents:
        original_path = Path(doc_info.doc.original_path)
        if original_path.exists():
            dest_path = original_dir / original_path.name
            # Avoid overwriting if same name exists
            if dest_path.exists():
                stem = original_path.stem
                suffix = original_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = original_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            try:
                shutil.copy2(original_path, dest_path)
            except Exception as e:
                print(f"  Warning: Could not copy {original_path.name}: {e}")

    print("  Copied original files to _original/")


def synthesize_all(
    documents: list[DocumentInfo],
    output_path: Path,
    preparation_time: float = 0.0,
    preparation_cost: float = 0.0,
    use_llm_synthesis: bool = False,
    preserve_originals: bool = True,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Generate all synthesis files for the prepared filesystem.

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem
        preparation_time: Total preparation time in seconds
        preparation_cost: Estimated LLM cost
        use_llm_synthesis: Whether to use LLM for corpus overview
        preserve_originals: Whether to copy original files
        client: Optional OpenAI client for LLM synthesis

    Returns:
        Dictionary containing synthesis results
    """
    print("Generating synthesis files...")

    # Generate meta files
    overview = generate_corpus_overview(
        documents, output_path, use_llm=use_llm_synthesis, client=client
    )
    nav_guide = generate_navigation_guide(output_path)
    stats = generate_statistics(documents, output_path, preparation_time, preparation_cost)

    # Copy original files if requested
    if preserve_originals:
        copy_original_files(documents, output_path)

    return {
        "corpus_overview": overview,
        "navigation_guide": nav_guide,
        "statistics": stats,
    }
