"""Filesystem navigation tools for the RAG agent.

This module provides the tools that the agent uses to navigate
and read from the prepared filesystem structure.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_evaluator.rag_implementations.filesystem_rag.passage_index import BM25PassageIndex

_MAX_FULL_READ_BYTES = 100_000
_MAX_SECTION_SIBLINGS = 40
_REGEX_METACHARACTERS = set("\\.^$*+?{}[]|()")
_PASSAGE_STEM_RE = re.compile(r"^(?P<section>\d+(?:\.\d+)*)-c\d+-s\d+$")
_NOISE_HEADER_RE = re.compile(r"^#\s+[0-9A-Fa-f]{8}\s+Passage\s+\d+")
_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)")


class FilesystemRAGTools:
    """Tools for navigating the prepared filesystem.

    Provides navigation and search operations:
    - list_directory: List files and folders
    - read_file: Read file contents with progressive disclosure
    - grep_search: Search for patterns in files
    - search_passages: Rank indexed passages with BM25
    - find_files: Find files by glob pattern
    - get_file_info: Get file metadata without reading content
    """

    def __init__(self, prepared_path: str) -> None:
        """Initialize tools with the prepared filesystem root.

        Args:
            prepared_path: Absolute path to the prepared filesystem root
        """
        self.prepared_path = Path(prepared_path)
        if not self.prepared_path.exists():
            raise ValueError(f"Prepared path does not exist: {prepared_path}")
        self._passage_index: BM25PassageIndex | None = None
        self._sibling_cache: dict[str, list[dict[str, Any]]] = {}

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path to absolute, ensuring it's within prepared_path.

        Args:
            relative_path: Path relative to prepared root

        Returns:
            Absolute Path object

        Raises:
            ValueError: If path attempts to escape prepared root
        """
        # Handle empty or root path
        if not relative_path or relative_path == ".":
            return self.prepared_path

        # Normalize and resolve
        full_path = (self.prepared_path / relative_path).resolve()

        # Security check: ensure path is within prepared_path
        try:
            full_path.relative_to(self.prepared_path.resolve())
        except ValueError:
            raise ValueError(f"Path escapes prepared root: {relative_path}")

        return full_path

    def list_directory(self, path: str = ".") -> list[dict[str, Any]]:
        """List contents of a directory with metadata.

        Args:
            path: Relative path from prepared root (e.g., "_index/topics")

        Returns:
            List of entries, each with:
            - name: str (file or directory name)
            - type: str ("file" or "directory")
            - size: int (bytes, for files only)

        Raises:
            ValueError: If path doesn't exist or isn't a directory
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        if not full_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        entries: list[dict[str, Any]] = []

        for item in sorted(full_path.iterdir()):
            entry: dict[str, Any] = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
            }

            if item.is_file():
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    entry["size"] = 0

            entries.append(entry)

        return entries

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        headers_only: bool = False,
    ) -> dict[str, Any]:
        """Read file contents with progressive disclosure support.

        Args:
            path: Relative path to file
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (inclusive)
            headers_only: If True and file > 500 lines, return only headers

        Returns:
            Dictionary with:
            - content: str (file content or headers)
            - total_lines: int
            - is_partial: bool (True if start_line/end_line used or headers_only)
            - headers: list[dict] (if headers_only, list of {line, level, text})

        Raises:
            ValueError: If path doesn't exist or isn't a file
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"File does not exist: {path}")

        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Security check: Ignore binary files
        binary_extensions = {".bin", ".sqlite3", ".db", ".pyc", ".exe", ".dll", ".so", ".pkl"}
        if full_path.suffix.lower() in binary_extensions:
            return {
                "content": f"Error: Cannot read binary file '{path}'. Access denied.",
                "total_lines": 0,
                "is_partial": False,
                "headers": [],
            }

        full_read_requested = start_line is None and end_line is None and not headers_only
        if full_read_requested:
            try:
                stat = full_path.stat()
            except OSError:
                stat = None
            if stat is not None and stat.st_size > _MAX_FULL_READ_BYTES:
                total_lines = self._count_lines(full_path)
                return {
                    "content": (
                        f"File '{path}' is too large for a full read "
                        f"({stat.st_size} bytes, {total_lines} lines). Use grep_search "
                        "with targeted terms, headers_only=True, or start_line/end_line."
                    ),
                    "total_lines": total_lines,
                    "is_partial": True,
                    "headers": [],
                    "truncated": True,
                    "size_bytes": stat.st_size,
                }

        # Read file content
        try:
            # First check if it looks like binary by reading first block
            with open(full_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return {
                        "content": f"Error: File '{path}' appears to be binary. Access denied.",
                        "total_lines": 0,
                        "is_partial": False,
                        "headers": [],
                    }

            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            content = full_path.read_text(encoding="latin-1")

        lines = content.split("\n")
        total_lines = len(lines)
        sibling_extra = self._section_sibling_info(full_path)

        # Headers only mode for large files
        if headers_only and total_lines > 500:
            headers = self._extract_headers(lines)
            header_content = "\n".join(f"{'#' * h['level']} {h['text']}" for h in headers)
            return {
                "content": header_content,
                "total_lines": total_lines,
                "is_partial": True,
                "headers": headers,
                **sibling_extra,
            }

        # Line range mode
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total_lines

            # Clamp to valid range
            start_idx = max(0, start_idx)
            end_idx = min(total_lines, end_idx)

            selected_lines = lines[start_idx:end_idx]
            return {
                "content": "\n".join(selected_lines),
                "total_lines": total_lines,
                "is_partial": True,
                "headers": [],
                **sibling_extra,
            }

        # Full file mode
        return {
            "content": content,
            "total_lines": total_lines,
            "is_partial": False,
            "headers": [],
            **sibling_extra,
        }

    def _count_lines(self, path: Path) -> int:
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for _ in f)
        except UnicodeDecodeError:
            with open(path, encoding="latin-1") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def _section_sibling_info(self, full_path: Path) -> dict[str, Any]:
        """Return sibling-chunk map keys for a documents/ passage file.

        Passage files are named ``<section>-c<chunk>-s<sentence>.md``. The map
        lists the other chunks of the same section with their first informative
        header so the agent can sweep siblings instead of satisficing on one.
        """
        try:
            rel_path = full_path.relative_to(self.prepared_path.resolve())
        except ValueError:
            return {}
        if rel_path.parts[0] != "documents" or full_path.suffix != ".md":
            return {}
        match = _PASSAGE_STEM_RE.match(full_path.stem)
        if not match:
            return {}

        section = match.group("section")
        entries = self._sibling_cache.get(section)
        if entries is None:
            entries = self._build_section_sibling_entries(section, full_path.parent)
            self._sibling_cache[section] = entries

        rel_posix = rel_path.as_posix()
        siblings = [entry for entry in entries if entry["file"] != rel_posix]
        if not siblings:
            return {}

        info: dict[str, Any] = {
            "section_id": section,
            "section_siblings": siblings[:_MAX_SECTION_SIBLINGS],
        }
        omitted = len(siblings) - _MAX_SECTION_SIBLINGS
        if omitted > 0:
            info["section_siblings_omitted"] = omitted
        return info

    def _build_section_sibling_entries(
        self, section: str, directory: Path
    ) -> list[dict[str, Any]]:
        """List all passage files of a section with their informative titles."""
        section_stem_re = re.compile(rf"^{re.escape(section)}-c\d+-s\d+$")
        entries: list[dict[str, Any]] = []
        for path in sorted(directory.glob(f"{section}-c*.md")):
            if not section_stem_re.match(path.stem):
                continue
            try:
                rel = path.relative_to(self.prepared_path.resolve()).as_posix()
            except ValueError:
                continue
            entries.append({"file": rel, "title": self._sibling_title(path)})
        return entries

    def _sibling_title(self, path: Path) -> str:
        """First informative markdown header, skipping one noise header."""
        header_lines: list[str] = []
        try:
            with open(path, encoding="utf-8") as f:
                for index, line in enumerate(f):
                    if index >= 15 or len(header_lines) >= 2:
                        break
                    if _HEADER_RE.match(line):
                        header_lines.append(line.rstrip())
        except (OSError, UnicodeDecodeError):
            return path.stem

        if not header_lines:
            return path.stem
        chosen = header_lines[0]
        if _NOISE_HEADER_RE.match(chosen) and len(header_lines) > 1:
            chosen = header_lines[1]
        header_match = _HEADER_RE.match(chosen)
        return header_match.group(1).strip() if header_match else path.stem

    def _extract_headers(self, lines: list[str]) -> list[dict[str, Any]]:
        """Extract markdown headers from lines.

        Args:
            lines: List of file lines

        Returns:
            List of header dicts with line, level, text
        """
        headers: list[dict[str, Any]] = []

        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+)", line)
            if match:
                headers.append(
                    {
                        "line": i + 1,  # 1-indexed
                        "level": len(match.group(1)),
                        "text": match.group(2).strip(),
                    }
                )

        return headers

    def grep_search(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*.md",
        max_results: int = 20,
        context_lines: int = 3,
        match_all_terms: bool = False,
    ) -> dict[str, Any]:
        """Search for a pattern in files with ranking and truncation metadata.

        A plain multi-word pattern with zero hits is transparently re-run in
        AND-mode (match_all_terms=True), since the words are usually not
        adjacent on one line; the result is marked with "fallback".

        Args:
            pattern: Regex pattern, or whitespace/comma-separated terms when
                match_all_terms is True
            path: Directory to search in
            file_pattern: Glob pattern for files to search
            max_results: Maximum number of results to return
            context_lines: Number of lines before/after each match
            match_all_terms: If True, only return files containing every term

        Returns:
            Dictionary with search metadata and ranked matches grouped by file.

        Raises:
            ValueError: If path doesn't exist
        """
        result = self._grep_once(
            pattern, path, file_pattern, max_results, context_lines, match_all_terms
        )
        if (
            not match_all_terms
            and result["total_matches"] == 0
            and not _REGEX_METACHARACTERS.intersection(pattern)
            and len(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'_-]*", pattern)) >= 2
        ):
            result = self._grep_once(
                pattern, path, file_pattern, max_results, context_lines, True
            )
            result["fallback"] = "match_all_terms"
        return result

    def _grep_once(
        self,
        pattern: str,
        path: str,
        file_pattern: str,
        max_results: int,
        context_lines: int,
        match_all_terms: bool,
    ) -> dict[str, Any]:
        """Run one grep pass and return ranked, grouped matches."""
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        context_lines = max(0, min(context_lines, 10))
        max_results = max(1, min(max_results, 100))

        term_patterns: list[re.Pattern[str]] = []
        regex: re.Pattern[str] | None = None
        terms: list[str] = []
        if match_all_terms:
            terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'_-]*", pattern)
            if not terms:
                raise ValueError("match_all_terms=True requires at least one search term")
            term_patterns = [
                re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms
            ]
        else:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        # Find matching files
        if full_path.is_file():
            files = [full_path]
        else:
            files = sorted(full_path.rglob(file_pattern))

        files_searched = 0
        flat_matches: list[dict[str, Any]] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            files_searched += 1

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            lines = content.split("\n")
            line_matches: list[dict[str, Any]] = []
            if match_all_terms:
                term_hit_counts = {
                    term: len(term_patterns[index].findall(content))
                    for index, term in enumerate(terms)
                }
                if any(count == 0 for count in term_hit_counts.values()):
                    continue

                for i, line in enumerate(lines):
                    line_terms = [
                        term
                        for index, term in enumerate(terms)
                        if term_patterns[index].search(line)
                    ]
                    if not line_terms:
                        continue
                    line_matches.append({"line_index": i, "matched_terms": line_terms})
            else:
                if regex is None:
                    continue
                for i, line in enumerate(lines):
                    if regex.search(line):
                        line_matches.append({"line_index": i, "matched_terms": [pattern]})

            if not line_matches:
                continue

            try:
                rel_path = file_path.relative_to(self.prepared_path)
            except ValueError:
                rel_path = file_path
            rel_path_str = str(rel_path)
            file_match_count = len(line_matches)

            for match in line_matches:
                line_index = int(match["line_index"])
                context_start = max(0, line_index - context_lines)
                context_end = min(len(lines), line_index + context_lines + 1)
                context = "\n".join(lines[context_start:context_end])
                matched_terms = list(match["matched_terms"])
                distinct_context_terms = self._count_distinct_terms(context, term_patterns, terms)
                exact_line_terms = len(set(matched_terms))
                score = file_match_count + (distinct_context_terms * 20) + (exact_line_terms * 10)
                if match_all_terms and distinct_context_terms == len(terms):
                    score += 30

                flat_matches.append(
                    {
                        "file": rel_path_str,
                        "line_number": line_index + 1,
                        "content": lines[line_index],
                        "context": context,
                        "context_start_line": context_start + 1,
                        "context_end_line": context_end,
                        "matched_terms": matched_terms,
                        "file_match_count": file_match_count,
                        "score": score,
                    }
                )

        ranked_matches = sorted(
            flat_matches,
            key=lambda item: (-item["file_match_count"], -item["score"], item["file"], item["line_number"]),
        )
        returned_matches = ranked_matches[:max_results]
        grouped_files = self._group_grep_matches(returned_matches)
        total_matches = len(flat_matches)

        return {
            "pattern": pattern,
            "path": path,
            "file_pattern": file_pattern,
            "match_all_terms": match_all_terms,
            "terms": terms,
            "context_lines": context_lines,
            "files_searched": files_searched,
            "files_with_matches": len({match["file"] for match in flat_matches}),
            "total_matches": total_matches,
            "returned_matches": len(returned_matches),
            "truncated": len(returned_matches) < total_matches,
            "results": returned_matches,
            "files": grouped_files,
        }

    def _count_distinct_terms(
        self,
        text: str,
        term_patterns: list[re.Pattern[str]],
        terms: list[str],
    ) -> int:
        """Count query terms present in a context window."""
        if not term_patterns:
            return 1
        return sum(
            1 for index, _term in enumerate(terms) if term_patterns[index].search(text)
        )

    def _group_grep_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group ranked grep matches by file while preserving rank order."""
        grouped: list[dict[str, Any]] = []
        by_file: dict[str, dict[str, Any]] = {}
        for match in matches:
            file_path = str(match["file"])
            if file_path not in by_file:
                group = {
                    "file": file_path,
                    "file_match_count": match["file_match_count"],
                    "returned_matches": 0,
                    "matches": [],
                }
                by_file[file_path] = group
                grouped.append(group)
            compact_match = {
                key: value
                for key, value in match.items()
                if key not in {"file", "file_match_count"}
            }
            by_file[file_path]["matches"].append(compact_match)
            by_file[file_path]["returned_matches"] += 1

        return grouped

    def search_passages(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the prepared BM25 passage index.

        Args:
            query: Natural-language or keyword search query
            top_k: Maximum number of ranked passages to return

        Returns:
            Dictionary containing query_terms and ranked passage results. Each
            result includes a snippet and a read_hint for read_file.
        """
        try:
            if self._passage_index is None:
                self._passage_index = BM25PassageIndex.load(self.prepared_path)
            return self._passage_index.search(query, top_k=top_k)
        except FileNotFoundError as exc:
            return {
                "query": query,
                "query_terms": [],
                "results": [],
                "error": f"{exc}. Re-run filesystem preparation to build it.",
            }

    def find_files(
        self,
        pattern: str,
        path: str = ".",
    ) -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "doc_*.md", "**/summary*.md")
            path: Directory to search in

        Returns:
            List of matching file paths (relative to prepared root)

        Raises:
            ValueError: If path doesn't exist
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        results: list[str] = []

        if full_path.is_file():
            # Check if single file matches pattern
            if fnmatch.fnmatch(full_path.name, pattern):
                try:
                    rel_path = full_path.relative_to(self.prepared_path)
                    results.append(str(rel_path))
                except ValueError:
                    results.append(str(full_path))
        else:
            # Search directory
            for file_path in full_path.rglob("*"):
                if file_path.is_file() and fnmatch.fnmatch(file_path.name, pattern):
                    try:
                        rel_path = file_path.relative_to(self.prepared_path)
                        results.append(str(rel_path))
                    except ValueError:
                        results.append(str(file_path))

        return sorted(results)

    def get_file_info(self, path: str) -> dict[str, Any]:
        """Get metadata about a file without reading content.

        Args:
            path: Path to file

        Returns:
            Dictionary with:
            - size: int (bytes)
            - lines: int (line count)
            - modified: str (ISO date)
            - type: str (extension)

        Raises:
            ValueError: If path doesn't exist or isn't a file
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"File does not exist: {path}")

        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Get file stats
        stat = full_path.stat()

        # Count lines without loading entire file into memory
        line_count = 0
        try:
            with open(full_path, encoding="utf-8") as f:
                for _ in f:
                    line_count += 1
        except (UnicodeDecodeError, OSError):
            line_count = 0

        # Get modification time
        try:
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        except (OSError, ValueError):
            modified = "unknown"

        return {
            "size": stat.st_size,
            "lines": line_count,
            "modified": modified,
            "type": full_path.suffix.lstrip(".") or "unknown",
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for the agent.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and folders in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Relative path from prepared root (e.g., '_index/topics')"
                                ),
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read file contents. Use headers_only=True for large files "
                        "to get structure first."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to file",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional start line (1-indexed)",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional end line (inclusive)",
                            },
                            "headers_only": {
                                "type": "boolean",
                                "description": (
                                    "If true and file >500 lines, "
                                    "return only headers with line numbers"
                                ),
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep_search",
                    "description": (
                        "Search files with ranked results, context, total match counts, "
                        "and truncation metadata. Use match_all_terms=True when several "
                        "terms must appear in the same file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": (
                                    "Regex pattern, or terms separated by spaces/commas "
                                    "when match_all_terms is true"
                                ),
                            },
                            "path": {
                                "type": "string",
                                "description": (
                                    "Directory to search in (default: current directory)"
                                ),
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": ("Glob pattern for files to search (default: *.md)"),
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum ranked matches to return",
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Lines of context before/after each match",
                            },
                            "match_all_terms": {
                                "type": "boolean",
                                "description": (
                                    "If true, return only files that contain every term "
                                    "from pattern"
                                ),
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_passages",
                    "description": (
                        "Rank prepared document passages with BM25. Use this for "
                        "natural-language searches, reformulated legal issues, or "
                        "multi-term lookups before reading the best source lines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or reformulated legal issue",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of passages to return",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_files",
                    "description": "Find files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": (
                                    "Glob pattern (e.g., 'doc_*.md', '**/summary*.md')"
                                ),
                            },
                            "path": {
                                "type": "string",
                                "description": (
                                    "Directory to search in (default: current directory)"
                                ),
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_info",
                    "description": "Get metadata about a file without reading its content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to file",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
        ]
