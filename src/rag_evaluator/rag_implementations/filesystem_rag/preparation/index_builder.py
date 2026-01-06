"""Index building for Filesystem RAG preparation pipeline.

This module generates the index files that enable efficient navigation
of the prepared filesystem structure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
    DocumentAnalysis,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
)


@dataclass
class DocumentInfo:
    """Combined document and analysis information for index building."""

    doc: ProcessedDocument
    analysis: DocumentAnalysis


def _get_primary_topic(topic_scores: dict[str, float]) -> str:
    """Get the primary topic based on scores.

    Args:
        topic_scores: Dictionary of topic -> score

    Returns:
        Topic name with highest score
    """
    if not topic_scores:
        return "general"
    return max(topic_scores.items(), key=lambda x: x[1])[0]


def _classify_document_topics(
    topic_scores: dict[str, float],
    primary_threshold: float = 0.4,
    secondary_threshold: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Classify document into primary and secondary topics.

    Args:
        topic_scores: Dictionary of topic -> score
        primary_threshold: Minimum score for primary classification
        secondary_threshold: Minimum score for secondary classification

    Returns:
        Tuple of (primary_topics, secondary_topics)
    """
    primary: list[str] = []
    secondary: list[str] = []

    for topic, score in topic_scores.items():
        if score >= primary_threshold:
            primary.append(topic)
        elif score >= secondary_threshold:
            secondary.append(topic)

    # Ensure at least one primary topic
    if not primary and topic_scores:
        best_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        primary.append(best_topic)

    return primary, secondary


def build_topic_map(
    documents: list[DocumentInfo],
    output_path: Path,
) -> dict[str, dict[str, list[str]]]:
    """Build the master topic map and individual topic index files.

    Creates:
    - _index/topics/_topic_map.md
    - _index/topics/{topic}.md for each topic

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem

    Returns:
        Dictionary mapping topics to document IDs
    """
    # Create topics directory
    topics_dir = output_path / "_index" / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    # Classify documents by topic
    topic_docs: dict[str, dict[str, list[str]]] = {
        "technical": {"primary": [], "secondary": []},
        "business": {"primary": [], "secondary": []},
        "science": {"primary": [], "secondary": []},
        "general": {"primary": [], "secondary": []},
    }

    doc_by_id: dict[str, DocumentInfo] = {d.doc.id: d for d in documents}

    for doc_info in documents:
        primary, secondary = _classify_document_topics(doc_info.analysis.topic_scores)
        for topic in primary:
            if topic in topic_docs:
                topic_docs[topic]["primary"].append(doc_info.doc.id)
        for topic in secondary:
            if topic in topic_docs:
                topic_docs[topic]["secondary"].append(doc_info.doc.id)

    # Generate _topic_map.md
    topic_map_content = _generate_topic_map_content(topic_docs)
    (topics_dir / "_topic_map.md").write_text(topic_map_content, encoding="utf-8")
    print("  Created: _index/topics/_topic_map.md")

    # Generate individual topic files
    for topic, doc_ids in topic_docs.items():
        all_ids = doc_ids["primary"] + doc_ids["secondary"]
        if all_ids:
            topic_content = _generate_topic_index_content(topic, doc_ids, doc_by_id, topic_docs)
            (topics_dir / f"{topic}.md").write_text(topic_content, encoding="utf-8")
            print(f"  Created: _index/topics/{topic}.md")

    return topic_docs


def _generate_topic_map_content(
    topic_docs: dict[str, dict[str, list[str]]],
) -> str:
    """Generate content for _topic_map.md."""
    lines = ["# Topic Map", ""]

    for topic, doc_ids in topic_docs.items():
        all_docs = doc_ids["primary"] + doc_ids["secondary"]
        count = len(all_docs)

        lines.append(f"## {topic.title()} ({count} documents)")

        if doc_ids["primary"]:
            lines.append(f"Primary: {', '.join(doc_ids['primary'])}")
        if doc_ids["secondary"]:
            lines.append(f"Secondary: {', '.join(doc_ids['secondary'])}")

        lines.append(f"→ Details: [{topic}.md](topics/{topic}.md)")
        lines.append("")

    return "\n".join(lines)


def _generate_topic_index_content(
    topic: str,
    doc_ids: dict[str, list[str]],
    doc_by_id: dict[str, DocumentInfo],
    all_topic_docs: dict[str, dict[str, list[str]]],
) -> str:
    """Generate content for individual topic index file."""
    lines = [f"# {topic.title()} Documents", ""]

    # Group by subtopics (using document topics)
    subtopics: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_ids["primary"] + doc_ids["secondary"]:
        if doc_id in doc_by_id:
            doc_info = doc_by_id[doc_id]
            # Use first topic from analysis as subtopic
            if doc_info.analysis.topics:
                subtopic = doc_info.analysis.topics[0]
            else:
                subtopic = "General"
            subtopics[subtopic].append(doc_id)

    for subtopic, ids in subtopics.items():
        lines.append(f"## {subtopic.title()}")

        for doc_id in ids:
            if doc_id not in doc_by_id:
                continue
            doc_info = doc_by_id[doc_id]
            is_primary = doc_id in doc_ids["primary"]
            label = "[PRIMARY]" if is_primary else "[SECONDARY]"

            lines.append(f"- **{doc_id}.md** {label}")

            # Summary (truncated)
            summary = doc_info.analysis.summary[:200]
            if len(doc_info.analysis.summary) > 200:
                summary += "..."
            lines.append(f"  - Summary: {summary}")

            # Key sections
            if doc_info.analysis.key_sections:
                section_titles = [s["title"] for s in doc_info.analysis.key_sections[:3]]
                lines.append(f"  - Key sections: {', '.join(section_titles)}")

            # Entities
            all_entities: list[str] = []
            for entity_list in doc_info.analysis.entities.values():
                all_entities.extend(entity_list[:2])
            if all_entities:
                lines.append(f"  - Entities: {', '.join(all_entities[:5])}")

            # Questions
            if doc_info.analysis.question_seeds:
                questions = doc_info.analysis.question_seeds[:2]
                lines.append(f"  - Can answer: {', '.join(f'"{q}"' for q in questions)}")

            lines.append("")

    # See Also section
    lines.append("## See Also")
    related_topics = [t for t in all_topic_docs.keys() if t != topic]
    for related in related_topics:
        if all_topic_docs[related]["primary"] or all_topic_docs[related]["secondary"]:
            lines.append(f"- Related topics: [{related}.md]({related}.md)")

    return "\n".join(lines)


def build_entity_registry(
    documents: list[DocumentInfo],
    output_path: Path,
) -> dict[str, dict[str, list[str]]]:
    """Build the entity registry and individual entity type files.

    Creates:
    - _index/entities/_entity_registry.md
    - _index/entities/{entity_type}.md for each type

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem

    Returns:
        Dictionary mapping entity types to entities and their document occurrences
    """
    entities_dir = output_path / "_index" / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate entities across documents
    entity_docs: dict[str, dict[str, list[str]]] = {
        "people": defaultdict(list),
        "concepts": defaultdict(list),
        "organizations": defaultdict(list),
        "products": defaultdict(list),
    }

    for doc_info in documents:
        for entity_type, entity_list in doc_info.analysis.entities.items():
            if entity_type in entity_docs:
                for entity in entity_list:
                    entity_docs[entity_type][entity].append(doc_info.doc.id)

    # Generate _entity_registry.md
    registry_content = _generate_entity_registry_content(entity_docs)
    (entities_dir / "_entity_registry.md").write_text(registry_content, encoding="utf-8")
    print("  Created: _index/entities/_entity_registry.md")

    # Generate individual entity type files
    for entity_type, entity_map in entity_docs.items():
        if entity_map:
            type_content = _generate_entity_type_content(entity_type, entity_map)
            (entities_dir / f"{entity_type}.md").write_text(type_content, encoding="utf-8")
            print(f"  Created: _index/entities/{entity_type}.md")

    return dict(entity_docs)


def _generate_entity_registry_content(
    entity_docs: dict[str, dict[str, list[str]]],
) -> str:
    """Generate content for _entity_registry.md."""
    lines = ["# Entity Registry", ""]

    for entity_type, entities in entity_docs.items():
        count = len(entities)
        lines.append(f"## {entity_type.title()} ({count} unique)")

        # Show top entities by frequency
        sorted_entities = sorted(entities.items(), key=lambda x: len(x[1]), reverse=True)
        for entity, doc_ids in sorted_entities[:5]:
            lines.append(f"- **{entity}** ({len(doc_ids)} docs): {', '.join(doc_ids[:3])}")

        if count > 5:
            lines.append(f"- ... and {count - 5} more")

        lines.append(f"→ Full list: [{entity_type}.md](entities/{entity_type}.md)")
        lines.append("")

    return "\n".join(lines)


def _generate_entity_type_content(
    entity_type: str,
    entities: dict[str, list[str]],
) -> str:
    """Generate content for individual entity type file."""
    lines = [f"# {entity_type.title()}", ""]

    # Sort by frequency
    sorted_entities = sorted(entities.items(), key=lambda x: len(x[1]), reverse=True)

    for entity, doc_ids in sorted_entities:
        lines.append(f"## {entity}")
        lines.append(f"Mentioned in {len(doc_ids)} document(s):")
        for doc_id in doc_ids:
            lines.append(f"- {doc_id}.md")
        lines.append("")

    return "\n".join(lines)


def build_question_seeds(
    documents: list[DocumentInfo],
    output_path: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Build the question seeds index.

    Creates:
    - _index/questions/question_seeds.md

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem

    Returns:
        Dictionary mapping question categories to (question, doc_id) tuples
    """
    questions_dir = output_path / "_index" / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    # Categorize questions
    categorized: dict[str, list[tuple[str, str]]] = {
        "factual": [],
        "how_to": [],
        "comparison": [],
        "analysis": [],
        "other": [],
    }

    for doc_info in documents:
        for question in doc_info.analysis.question_seeds:
            category = _categorize_question(question)
            categorized[category].append((question, doc_info.doc.id))

    # Generate question_seeds.md
    content = _generate_question_seeds_content(categorized)
    (questions_dir / "question_seeds.md").write_text(content, encoding="utf-8")
    print("  Created: _index/questions/question_seeds.md")

    return categorized


def _categorize_question(question: str) -> str:
    """Categorize a question by its type."""
    q_lower = question.lower()

    if any(word in q_lower for word in ["what is", "who is", "when", "where"]):
        return "factual"
    elif any(word in q_lower for word in ["how to", "how do", "how can", "steps"]):
        return "how_to"
    elif any(word in q_lower for word in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    elif any(word in q_lower for word in ["why", "analyze", "explain", "challenges", "benefits"]):
        return "analysis"
    else:
        return "other"


def _generate_question_seeds_content(
    categorized: dict[str, list[tuple[str, str]]],
) -> str:
    """Generate content for question_seeds.md."""
    category_titles = {
        "factual": "Factual Lookups",
        "how_to": "How-To Questions",
        "comparison": "Comparison Questions",
        "analysis": "Analysis Questions",
        "other": "Other Questions",
    }

    lines = ["# Question Seeds", ""]

    for category, questions in categorized.items():
        if not questions:
            continue

        lines.append(f"## {category_titles.get(category, category.title())}")

        for question, doc_id in questions:
            lines.append(f'- "{question}" → {doc_id}')

        lines.append("")

    return "\n".join(lines)


def build_timeline(
    documents: list[DocumentInfo],
    output_path: Path,
) -> list[dict[str, Any]]:
    """Build the timeline index from temporal markers.

    Creates:
    - _index/temporal/timeline.md

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem

    Returns:
        List of timeline entries sorted by date
    """
    temporal_dir = output_path / "_index" / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)

    # Collect all temporal markers
    all_markers: list[dict[str, Any]] = []

    for doc_info in documents:
        for marker in doc_info.analysis.temporal_markers:
            all_markers.append(
                {
                    "date": marker.get("date", ""),
                    "event": marker.get("event", ""),
                    "doc_id": doc_info.doc.id,
                }
            )

    # Sort by date (simple string sort works for YYYY-MM-DD format)
    all_markers.sort(key=lambda x: x.get("date", ""))

    # Generate timeline.md
    content = _generate_timeline_content(all_markers)
    (temporal_dir / "timeline.md").write_text(content, encoding="utf-8")
    print("  Created: _index/temporal/timeline.md")

    return all_markers


def _generate_timeline_content(markers: list[dict[str, Any]]) -> str:
    """Generate content for timeline.md."""
    lines = ["# Timeline", ""]

    if not markers:
        lines.append("No temporal markers found in documents.")
        return "\n".join(lines)

    # Group by year
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker in markers:
        date = marker.get("date", "")
        # Extract year
        if len(date) >= 4:
            year = date[:4]
        else:
            year = "Unknown"
        by_year[year].append(marker)

    for year in sorted(by_year.keys(), reverse=True):
        lines.append(f"## {year}")

        for marker in by_year[year]:
            date = marker.get("date", "")
            event = marker.get("event", "")[:100]  # Truncate long events
            doc_id = marker.get("doc_id", "")
            lines.append(f"- **{date}**: {event} (→ {doc_id})")

        lines.append("")

    return "\n".join(lines)


def build_all_indexes(
    documents: list[DocumentInfo],
    output_path: Path,
) -> dict[str, Any]:
    """Build all index files for the prepared filesystem.

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem

    Returns:
        Dictionary containing all index data
    """
    print("Building indexes...")

    topic_data = build_topic_map(documents, output_path)
    entity_data = build_entity_registry(documents, output_path)
    question_data = build_question_seeds(documents, output_path)
    timeline_data = build_timeline(documents, output_path)

    return {
        "topics": topic_data,
        "entities": entity_data,
        "questions": question_data,
        "timeline": timeline_data,
    }


def write_document_files(
    documents: list[DocumentInfo],
    output_path: Path,
) -> None:
    """Write document markdown files and metadata JSON files.

    Creates:
    - documents/{doc_id}.md
    - documents/{doc_id}.meta.json
    - _summaries/{doc_id}_summary.md

    Args:
        documents: List of DocumentInfo objects
        output_path: Base path for prepared filesystem
    """
    docs_dir = output_path / "documents"
    summaries_dir = output_path / "_summaries"
    docs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    for doc_info in documents:
        doc = doc_info.doc
        analysis = doc_info.analysis

        # Write markdown content
        doc_path = docs_dir / f"{doc.id}.md"
        doc_path.write_text(doc.markdown_content, encoding="utf-8")

        # Write metadata JSON
        metadata = {
            "id": doc.id,
            "original_file": doc.original_path,
            "original_format": doc.original_format,
            "title": doc.title,
            "word_count": doc.word_count,
            "char_count": doc.char_count,
            "line_count": doc.line_count,
            "language": doc.language,
            "modified_date": doc.modified_date,
            "topics": analysis.topics,
            "topic_scores": analysis.topic_scores,
            "entities": analysis.entities,
            "sections": doc.sections,
            "question_seeds": analysis.question_seeds,
            "summary_path": f"_summaries/{doc.id}_summary.md",
            "related_docs": [],  # Will be populated by synthesizer
            "analysis_method": analysis.analysis_method,
        }
        meta_path = docs_dir / f"{doc.id}.meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Write summary
        summary_content = _generate_summary_file_content(doc, analysis)
        summary_path = summaries_dir / f"{doc.id}_summary.md"
        summary_path.write_text(summary_content, encoding="utf-8")

        print(f"  Written: {doc.id}.md, {doc.id}.meta.json, {doc.id}_summary.md")


def _generate_summary_file_content(
    doc: ProcessedDocument,
    analysis: DocumentAnalysis,
) -> str:
    """Generate content for document summary file."""
    reading_time = max(1, doc.word_count // 200)  # ~200 words per minute

    lines = [
        f"# Summary: {doc.title}",
        "",
        f"**Source:** {doc.id}.md | **Words:** {doc.word_count} | **Reading time:** {reading_time} min",
        "",
        "## Overview",
        analysis.summary,
        "",
        "## Key Points",
    ]

    # Extract key points from summary or sections
    if analysis.key_sections:
        for i, section in enumerate(analysis.key_sections[:5], 1):
            lines.append(f"{i}. {section.get('title', 'Section')}: {section.get('summary', '')}")
    else:
        lines.append("1. See full document for details")

    lines.append("")
    lines.append("## Main Sections")

    for section in doc.sections[:10]:
        title = section.get("title", "")
        start = section.get("start_line", 0)
        end = section.get("end_line", 0)
        lines.append(f"- {title} (lines {start}-{end})")

    lines.append("")
    lines.append("## Key Entities")

    for entity_type, entities in analysis.entities.items():
        if entities:
            lines.append(f"- {entity_type.title()}: {', '.join(entities[:5])}")

    lines.append("")
    lines.append("## Questions This Document Answers")

    for question in analysis.question_seeds[:5]:
        lines.append(f"- {question}")

    return "\n".join(lines)
