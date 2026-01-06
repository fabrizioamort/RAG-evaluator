"""Document processing for Filesystem RAG preparation pipeline.

This module handles loading raw documents and converting them to markdown format
with structure detection for the prepared filesystem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_evaluator.common.document_loaders import Document, create_loader


@dataclass
class RawDocument:
    """Represents a raw document before processing.

    Attributes:
        id: Unique identifier (e.g., "doc_001")
        original_path: Path to the original file
        original_format: File format (e.g., "pdf", "txt", "docx")
        raw_content: Extracted text content
        file_size: Size in bytes
        modified_date: Last modification date (ISO format) or None
    """

    id: str
    original_path: str
    original_format: str
    raw_content: str
    file_size: int
    modified_date: str | None = None


@dataclass
class ProcessedDocument:
    """Represents a document after markdown conversion.

    Attributes:
        id: Unique identifier matching RawDocument
        original_path: Path to the original file
        original_format: File format
        markdown_content: Content converted to markdown
        title: Extracted or inferred document title
        word_count: Number of words in document
        char_count: Number of characters
        line_count: Number of lines
        language: Detected language (default: "en")
        modified_date: Last modification date or None
        sections: List of detected section info
        metadata: Additional format-specific metadata
    """

    id: str
    original_path: str
    original_format: str
    markdown_content: str
    title: str
    word_count: int
    char_count: int
    line_count: int
    language: str = "en"
    modified_date: str | None = None
    sections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_documents(documents_path: str) -> list[RawDocument]:
    """Load all documents from a directory and convert to RawDocument objects.

    Args:
        documents_path: Path to directory containing raw documents

    Returns:
        List of RawDocument objects, sorted by filename

    Raises:
        ValueError: If documents_path does not exist
    """
    path = Path(documents_path)
    if not path.exists():
        raise ValueError(f"Documents path does not exist: {documents_path}")

    if not path.is_dir():
        raise ValueError(f"Documents path is not a directory: {documents_path}")

    raw_documents: list[RawDocument] = []
    supported_extensions = {".txt", ".pdf", ".docx"}

    # Get all files recursively and sort by name
    files = sorted(
        [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
    )

    for idx, file_path in enumerate(files, start=1):
        doc_id = f"doc_{idx:03d}"
        try:
            loader = create_loader(str(file_path))
            doc: Document = loader.load(str(file_path))

            # Get file modification time
            modified_date: str | None = None
            try:
                mtime = file_path.stat().st_mtime
                modified_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                pass

            raw_doc = RawDocument(
                id=doc_id,
                original_path=str(file_path),
                original_format=file_path.suffix.lower().lstrip("."),
                raw_content=doc.content,
                file_size=file_path.stat().st_size,
                modified_date=modified_date,
            )
            raw_documents.append(raw_doc)
            print(f"  Loaded: {file_path.name} -> {doc_id}")

        except Exception as e:
            print(f"  Warning: Failed to load {file_path.name}: {e}")
            continue

    print(f"Loaded {len(raw_documents)} documents from {documents_path}")
    return raw_documents


def detect_txt_structure(content: str) -> list[dict[str, Any]]:
    """Detect structure in plain text files using heuristics.

    Detects:
    - ALL CAPS lines as potential headers
    - Lines followed by === or --- as headers
    - Numbered sections (1., 1.1., etc.)
    - Blank line separated paragraphs

    Args:
        content: Raw text content

    Returns:
        List of detected sections with title, start_line, level
    """
    sections: list[dict[str, Any]] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        line_num = i + 1  # 1-indexed
        level = 0
        title = ""

        # Check for ALL CAPS headers (at least 3 chars, max 100)
        if (
            stripped.isupper()
            and len(stripped) >= 3
            and len(stripped) <= 100
            and not stripped.startswith("#")
        ):
            level = 1
            title = stripped.title()  # Convert to title case

        # Check for underlined headers (next line is === or ---)
        elif i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and len(next_line) >= 3:
                if all(c == "=" for c in next_line):
                    level = 1
                    title = stripped
                elif all(c == "-" for c in next_line):
                    level = 2
                    title = stripped

        # Check for numbered sections like "1.", "1.1", "1.1.1"
        numbered_match = re.match(r"^(\d+(?:\.\d+)*)\s*[.:\-)\]]\s*(.+)", stripped)
        if numbered_match:
            section_num = numbered_match.group(1)
            section_title = numbered_match.group(2).strip()
            # Determine level by dot count
            level = section_num.count(".") + 1
            title = f"{section_num} {section_title}"

        if level > 0 and title:
            sections.append({"title": title, "start_line": line_num, "level": level})

    return sections


def _convert_txt_to_markdown(content: str, title: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert plain text to markdown with structure detection.

    Args:
        content: Raw text content
        title: Document title

    Returns:
        Tuple of (markdown_content, sections_list)
    """
    sections = detect_txt_structure(content)
    lines = content.split("\n")
    output_lines: list[str] = []

    # Add title as H1 if not already present
    if not content.strip().startswith("#"):
        output_lines.append(f"# {title}")
        output_lines.append("")

    # Track which lines are headers to convert
    header_lines: dict[int, int] = {}  # line_num -> level
    for section in sections:
        header_lines[section["start_line"]] = section["level"]

    # Process lines
    i = 0
    while i < len(lines):
        line_num = i + 1
        line = lines[i]

        if line_num in header_lines:
            level = header_lines[line_num]
            # Convert to markdown header
            header_prefix = "#" * (level + 1)  # +1 because title is H1
            # Clean the line - remove ALL CAPS formatting, underlines
            clean_title = line.strip()
            if clean_title.isupper():
                clean_title = clean_title.title()
            output_lines.append(f"{header_prefix} {clean_title}")

            # Skip underline if present
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (
                    all(c == "=" for c in next_line) or all(c == "-" for c in next_line)
                ):
                    i += 1
        else:
            output_lines.append(line)

        i += 1

    # Clean up excessive blank lines
    markdown = "\n".join(output_lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown, sections


def _convert_pdf_to_markdown(content: str, title: str, metadata: dict[str, Any]) -> str:
    """Convert PDF extracted text to markdown.

    Args:
        content: Raw text extracted from PDF
        title: Document title
        metadata: PDF metadata

    Returns:
        Markdown formatted content
    """
    lines: list[str] = []

    # Add title
    lines.append(f"# {title}")
    lines.append("")

    # Add metadata if available
    if metadata.get("author"):
        lines.append(f"**Author:** {metadata['author']}")
    if metadata.get("pages"):
        lines.append(f"**Pages:** {metadata['pages']}")
    if metadata.get("author") or metadata.get("pages"):
        lines.append("")

    # Process content - try to detect headers and structure
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect potential headers (short lines, possibly capitalized)
        if len(para) < 100 and "\n" not in para:
            # Check if it looks like a header
            if para.isupper() or (para[0].isupper() and not para.endswith(".")):
                # Likely a header
                lines.append(f"## {para.title() if para.isupper() else para}")
                lines.append("")
                continue

        # Regular paragraph
        lines.append(para)
        lines.append("")

    return "\n".join(lines)


def _convert_docx_to_markdown(content: str, title: str, metadata: dict[str, Any]) -> str:
    """Convert DOCX extracted text to markdown.

    Args:
        content: Raw text extracted from DOCX
        title: Document title
        metadata: DOCX metadata

    Returns:
        Markdown formatted content
    """
    lines: list[str] = []

    # Add title
    lines.append(f"# {title}")
    lines.append("")

    # Add metadata if available
    if metadata.get("author"):
        lines.append(f"**Author:** {metadata['author']}")
        lines.append("")

    # DOCX content is usually already well-structured from paragraph extraction
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check for table rows (contain |)
        if " | " in para:
            # Format as markdown table row
            lines.append(f"| {para} |")
            continue

        # Check if it looks like a header (short, no period at end)
        if len(para) < 80 and "\n" not in para and not para.endswith("."):
            if para.isupper():
                lines.append(f"## {para.title()}")
            elif para[0].isupper():
                lines.append(f"## {para}")
            else:
                lines.append(para)
            lines.append("")
            continue

        # Regular paragraph
        lines.append(para)
        lines.append("")

    return "\n".join(lines)


def _extract_title(content: str, original_path: str, metadata: dict[str, Any]) -> str:
    """Extract or infer document title.

    Priority:
    1. Title from metadata
    2. First markdown header
    3. First line if short
    4. Filename without extension

    Args:
        content: Document content
        original_path: Original file path
        metadata: Document metadata

    Returns:
        Extracted or inferred title
    """
    # Check metadata
    if metadata.get("title"):
        return str(metadata["title"])

    # Check for markdown header
    lines = content.strip().split("\n")
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()

    # Check if first line is short (likely a title)
    if lines and len(lines[0].strip()) < 100 and lines[0].strip():
        first_line = lines[0].strip()
        if not first_line.endswith("."):
            return first_line

    # Fall back to filename
    path = Path(original_path)
    return path.stem.replace("_", " ").replace("-", " ").title()


def _detect_sections_from_markdown(content: str) -> list[dict[str, Any]]:
    """Detect sections from markdown headers.

    Args:
        content: Markdown content

    Returns:
        List of section dicts with title, start_line, end_line, level
    """
    sections: list[dict[str, Any]] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Match markdown headers
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({"title": title, "start_line": i + 1, "level": level})

    # Calculate end lines
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section["end_line"] = sections[i + 1]["start_line"] - 1
        else:
            section["end_line"] = len(lines)

    return sections


def convert_to_markdown(raw_doc: RawDocument, doc_metadata: dict[str, Any]) -> ProcessedDocument:
    """Convert a RawDocument to markdown format with structure detection.

    Args:
        raw_doc: The raw document to convert
        doc_metadata: Additional metadata from the document loader

    Returns:
        ProcessedDocument with markdown content and extracted metadata
    """
    content = raw_doc.raw_content
    fmt = raw_doc.original_format

    # Extract title first
    title = _extract_title(content, raw_doc.original_path, doc_metadata)

    # Convert based on format
    sections: list[dict[str, Any]] = []
    if fmt == "txt":
        markdown_content, sections = _convert_txt_to_markdown(content, title)
    elif fmt == "pdf":
        markdown_content = _convert_pdf_to_markdown(content, title, doc_metadata)
    elif fmt == "docx":
        markdown_content = _convert_docx_to_markdown(content, title, doc_metadata)
    else:
        # Default: wrap in markdown with title
        markdown_content = f"# {title}\n\n{content}"

    # Detect sections from final markdown if not already detected
    if not sections:
        sections = _detect_sections_from_markdown(markdown_content)

    # Calculate statistics
    word_count = len(markdown_content.split())
    char_count = len(markdown_content)
    line_count = len(markdown_content.split("\n"))

    # Detect language (simple heuristic - check for common words)
    language = "en"  # Default to English

    return ProcessedDocument(
        id=raw_doc.id,
        original_path=raw_doc.original_path,
        original_format=raw_doc.original_format,
        markdown_content=markdown_content,
        title=title,
        word_count=word_count,
        char_count=char_count,
        line_count=line_count,
        language=language,
        modified_date=raw_doc.modified_date,
        sections=sections,
        metadata=doc_metadata,
    )
