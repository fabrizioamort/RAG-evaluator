"""Document analysis for Filesystem RAG preparation pipeline.

This module provides hybrid analysis (heuristic + LLM) to extract
summaries, topics, entities, and question seeds from documents.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from rag_evaluator.common.llm_utils import get_safe_llm_params
from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
)


@dataclass
class DocumentAnalysis:
    """Results from analyzing a document.

    Attributes:
        summary: 2-3 paragraph summary of key points
        topics: List of topic keywords
        topic_scores: Scores for predefined topic categories
        entities: Extracted entities by type
        temporal_markers: Dates/events found in document
        question_seeds: Questions this document can answer
        key_sections: Important sections with summaries
        related_topics: Topics related to this document
        analysis_method: "heuristic" or "llm"
    """

    summary: str
    topics: list[str]
    topic_scores: dict[str, float]
    entities: dict[str, list[str]]
    temporal_markers: list[dict[str, str]]
    question_seeds: list[str]
    key_sections: list[dict[str, str]]
    related_topics: list[str]
    analysis_method: str = "heuristic"


# Predefined topic categories and their keywords
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "technical": [
        "api",
        "code",
        "software",
        "system",
        "database",
        "server",
        "algorithm",
        "function",
        "class",
        "method",
        "implementation",
        "architecture",
        "framework",
        "library",
        "programming",
        "developer",
        "debug",
        "deploy",
        "config",
        "rag",
        "vector",
        "embedding",
        "llm",
        "model",
        "neural",
        "machine learning",
        "ai",
        "retrieval",
        "search",
        "index",
        "query",
        "chunk",
        "tokenize",
    ],
    "business": [
        "company",
        "market",
        "revenue",
        "profit",
        "customer",
        "product",
        "sales",
        "strategy",
        "management",
        "team",
        "growth",
        "investment",
        "budget",
        "roi",
        "stakeholder",
        "client",
        "service",
        "process",
        "workflow",
        "efficiency",
    ],
    "science": [
        "research",
        "study",
        "experiment",
        "data",
        "analysis",
        "hypothesis",
        "theory",
        "method",
        "result",
        "conclusion",
        "evidence",
        "observation",
        "sample",
        "variable",
        "statistical",
        "correlation",
        "significant",
    ],
    "general": [
        "information",
        "document",
        "content",
        "overview",
        "introduction",
        "summary",
        "section",
        "topic",
        "reference",
        "example",
        "note",
    ],
}

# Entity detection patterns
ENTITY_PATTERNS: dict[str, list[str]] = {
    "people": [
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Two capitalized words (names)
        r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.)\s+[A-Z][a-z]+\b",  # Titles + name
    ],
    "organizations": [
        r"\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization|Foundation|Institute)\b",
        r"\b(?:Google|Microsoft|Amazon|OpenAI|Anthropic|Meta|Apple)\b",
    ],
    "products": [
        r"\b(?:ChromaDB|LangChain|OpenAI|GPT-\d|Claude|Pinecone|Qdrant|Neo4j)\b",
        r"\b[A-Z][a-z]+(?:DB|API|SDK|AI|ML)\b",
    ],
    "concepts": [
        r"\b(?:RAG|retrieval.augmented.generation|embedding|vector|semantic.search)\b",
        r"\b(?:machine.learning|deep.learning|neural.network|transformer)\b",
        r"\b(?:natural.language.processing|NLP|LLM|large.language.model)\b",
    ],
}

# Date patterns for temporal markers
DATE_PATTERNS: list[str] = [
    r"\b(\d{4})-(\d{2})-(\d{2})\b",  # YYYY-MM-DD
    r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",  # MM/DD/YYYY
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{4}\b",  # Just year
]


def _calculate_topic_scores(content: str) -> dict[str, float]:
    """Calculate topic scores based on keyword frequency.

    Args:
        content: Document content

    Returns:
        Dictionary of topic -> score (0.0 to 1.0)
    """
    content_lower = content.lower()
    words = re.findall(r"\b\w+\b", content_lower)
    word_count = len(words)

    if word_count == 0:
        return {"technical": 0.0, "business": 0.0, "science": 0.0, "general": 0.0}

    scores: dict[str, float] = {}
    max_score = 0.0

    for topic, keywords in TOPIC_KEYWORDS.items():
        # Count keyword occurrences
        count = sum(1 for word in words if word in keywords)
        # Also check multi-word phrases
        for keyword in keywords:
            if " " in keyword:
                count += content_lower.count(keyword)

        # Normalize score
        score = min(count / (word_count / 100), 1.0)  # Max 1% keyword density = 1.0
        scores[topic] = score
        max_score = max(max_score, score)

    # Normalize so scores sum to ~1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 2) for k, v in scores.items()}
    else:
        # Default distribution if no keywords found
        scores = {"technical": 0.25, "business": 0.25, "science": 0.25, "general": 0.25}

    return scores


def _extract_keywords(content: str, top_n: int = 10) -> list[str]:
    """Extract top keywords using simple TF analysis.

    Args:
        content: Document content
        top_n: Number of keywords to return

    Returns:
        List of top keywords
    """
    # Common stop words to filter out
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "his",
    }

    # Extract words
    words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
    words = [w for w in words if w not in stop_words]

    # Count frequency
    counter = Counter(words)

    # Get top keywords
    return [word for word, _ in counter.most_common(top_n)]


def _extract_entities_heuristic(content: str) -> dict[str, list[str]]:
    """Extract entities using regex patterns.

    Args:
        content: Document content

    Returns:
        Dictionary of entity_type -> list of entities
    """
    entities: dict[str, list[str]] = {
        "people": [],
        "organizations": [],
        "products": [],
        "concepts": [],
    }

    for entity_type, patterns in ENTITY_PATTERNS.items():
        found: set[str] = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = " ".join(match)
                # Clean and normalize
                match = match.strip()
                if len(match) > 2:  # Skip very short matches
                    found.add(match)
        entities[entity_type] = sorted(list(found))[:10]  # Limit to 10 per type

    return entities


def _extract_temporal_markers(content: str) -> list[dict[str, str]]:
    """Extract dates and temporal references.

    Args:
        content: Document content

    Returns:
        List of temporal markers with date and context
    """
    markers: list[dict[str, str]] = []

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, content):
            date_str = match.group(0)
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            context = re.sub(r"\s+", " ", context)

            markers.append({"date": date_str, "event": context})

    # Deduplicate by date
    seen_dates: set[str] = set()
    unique_markers: list[dict[str, str]] = []
    for marker in markers:
        if marker["date"] not in seen_dates:
            seen_dates.add(marker["date"])
            unique_markers.append(marker)

    return unique_markers[:10]  # Limit to 10


def _generate_question_seeds_heuristic(
    title: str, topics: list[str], sections: list[dict[str, Any]]
) -> list[str]:
    """Generate question seeds based on document structure.

    Args:
        title: Document title
        topics: Extracted topics
        sections: Document sections

    Returns:
        List of potential questions
    """
    questions: list[str] = []

    # Questions based on title
    questions.append(f"What is {title}?")
    questions.append(f"How does {title} work?")

    # Questions based on topics
    for topic in topics[:3]:
        questions.append(f"What is {topic}?")
        questions.append(f"How is {topic} used?")

    # Questions based on sections
    for section in sections[:5]:
        section_title = section.get("title", "")
        if section_title:
            questions.append(f"What does the section on {section_title} cover?")

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_questions: list[str] = []
    for q in questions:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_questions.append(q)

    return unique_questions[:10]


def _generate_summary_heuristic(content: str, title: str) -> str:
    """Generate a summary using heuristic extraction.

    Takes the first paragraph and key sentences containing important terms.

    Args:
        content: Document content
        title: Document title

    Returns:
        Generated summary
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return f"Document about {title}."

    # Start with first substantial paragraph (skip headers)
    summary_parts: list[str] = []
    for para in paragraphs[:3]:
        # Skip short lines that are likely headers
        if len(para) > 100:
            summary_parts.append(para)
            break

    # Add sentences containing keywords
    keywords = _extract_keywords(content, 5)
    sentences = re.split(r"[.!?]+", content)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 50 and any(kw in sentence.lower() for kw in keywords):
            if sentence not in summary_parts:
                summary_parts.append(sentence + ".")
                if len(summary_parts) >= 3:
                    break

    if not summary_parts:
        summary_parts = [paragraphs[0][:500] if paragraphs else f"Document about {title}."]

    return " ".join(summary_parts)


def heuristic_analysis(doc: ProcessedDocument) -> DocumentAnalysis:
    """Analyze document using heuristic methods (no LLM calls).

    Best for simple documents under 1000 words.

    Args:
        doc: ProcessedDocument to analyze

    Returns:
        DocumentAnalysis with extracted information
    """
    content = doc.markdown_content
    title = doc.title

    # Calculate topic scores
    topic_scores = _calculate_topic_scores(content)

    # Extract keywords as topics
    topics = _extract_keywords(content, 10)

    # Extract entities
    entities = _extract_entities_heuristic(content)

    # Extract temporal markers
    temporal_markers = _extract_temporal_markers(content)

    # Generate question seeds
    question_seeds = _generate_question_seeds_heuristic(title, topics, doc.sections)

    # Generate summary
    summary = _generate_summary_heuristic(content, title)

    # Key sections from document structure
    key_sections = [
        {"title": s["title"], "summary": f"Section covering {s['title']}", "start_marker": ""}
        for s in doc.sections[:5]
    ]

    # Related topics (topics with lower scores)
    related_topics = [t for t, s in topic_scores.items() if 0.1 < s < 0.5]

    return DocumentAnalysis(
        summary=summary,
        topics=topics,
        topic_scores=topic_scores,
        entities=entities,
        temporal_markers=temporal_markers,
        question_seeds=question_seeds,
        key_sections=key_sections,
        related_topics=related_topics,
        analysis_method="heuristic",
    )


# LLM Analysis Prompt Template
ANALYSIS_PROMPT = """Analyze the following document and extract structured information.

DOCUMENT TITLE: {title}
DOCUMENT CONTENT:
{content}

Provide a JSON response with the following structure:
{{
  "summary": "2-3 paragraph summary of the document's key points and purpose",
  "topics": ["topic1", "topic2", "topic3"],
  "topic_scores": {{
    "technical": 0.0,
    "business": 0.0,
    "science": 0.0,
    "general": 0.0
  }},
  "entities": {{
    "people": ["Name1", "Name2"],
    "concepts": ["concept1", "concept2"],
    "organizations": ["org1", "org2"],
    "products": ["product1", "product2"]
  }},
  "temporal_markers": [
    {{"date": "YYYY-MM", "event": "description"}}
  ],
  "question_seeds": [
    "Question this document can answer 1?",
    "Question this document can answer 2?",
    "Question this document can answer 3?",
    "Question this document can answer 4?",
    "Question this document can answer 5?"
  ],
  "key_sections": [
    {{"title": "Section Name", "summary": "Brief description", "start_marker": "first few words"}}
  ],
  "related_topics": ["related1", "related2"]
}}

Rules:
- topic_scores should sum to approximately 1.0
- Include 5-10 question_seeds covering different aspects
- Be specific in entity extraction
- Only include temporal_markers if dates are mentioned

Respond ONLY with valid JSON, no other text."""


def llm_analysis(
    doc: ProcessedDocument,
    client: OpenAI | None = None,
    model: str | None = None,
) -> DocumentAnalysis:
    """Analyze document using LLM for rich extraction.

    Best for complex documents over 1000 words.

    Args:
        doc: ProcessedDocument to analyze
        client: Optional OpenAI client (created if not provided)
        model: Optional model name (defaults to gpt-4o-mini for cost efficiency)

    Returns:
        DocumentAnalysis with LLM-extracted information
    """
    if client is None:
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)

    if model is None:
        # Use gpt-4o-mini for cost-effective analysis
        model = "gpt-4o-mini"

    content = doc.markdown_content
    title = doc.title

    # Truncate content if too long (keep under ~12k tokens)
    max_chars = 40000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated for analysis...]"

    prompt = ANALYSIS_PROMPT.format(title=title, content=content)

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        kwargs = get_safe_llm_params(model, temperature=0.0, **kwargs)

        response = client.chat.completions.create(**kwargs)

        result_text = response.choices[0].message.content
        if not result_text:
            raise ValueError("Empty response from LLM")

        result = json.loads(result_text)

        return DocumentAnalysis(
            summary=result.get("summary", ""),
            topics=result.get("topics", []),
            topic_scores=result.get(
                "topic_scores",
                {"technical": 0.25, "business": 0.25, "science": 0.25, "general": 0.25},
            ),
            entities=result.get(
                "entities",
                {"people": [], "concepts": [], "organizations": [], "products": []},
            ),
            temporal_markers=result.get("temporal_markers", []),
            question_seeds=result.get("question_seeds", []),
            key_sections=result.get("key_sections", []),
            related_topics=result.get("related_topics", []),
            analysis_method="llm",
        )

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse LLM response as JSON: {e}")
        # Fall back to heuristic analysis
        return heuristic_analysis(doc)
    except Exception as e:
        print(f"  Warning: LLM analysis failed: {e}")
        # Fall back to heuristic analysis
        return heuristic_analysis(doc)


def analyze_document(
    doc: ProcessedDocument,
    force_method: str | None = None,
    word_threshold: int = 1000,
    client: OpenAI | None = None,
) -> DocumentAnalysis:
    """Analyze a document using hybrid approach.

    Decision logic:
    - Documents under word_threshold: Use heuristic analysis (free)
    - Documents over word_threshold: Use LLM analysis (paid)

    Args:
        doc: ProcessedDocument to analyze
        force_method: Force "heuristic" or "llm" (bypasses threshold)
        word_threshold: Word count threshold for LLM usage (default: 1000)
        client: Optional OpenAI client for LLM analysis

    Returns:
        DocumentAnalysis with extracted information
    """
    method = force_method

    if method is None:
        # Decide based on document complexity
        if doc.word_count < word_threshold:
            method = "heuristic"
            print(f"  Using heuristic analysis for {doc.id} ({doc.word_count} words)")
        else:
            method = "llm"
            print(f"  Using LLM analysis for {doc.id} ({doc.word_count} words)")

    if method == "heuristic":
        return heuristic_analysis(doc)
    else:
        return llm_analysis(doc, client=client)
