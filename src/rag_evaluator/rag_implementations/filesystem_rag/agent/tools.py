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


class FilesystemRAGTools:
    """Tools for navigating the prepared filesystem.

    Provides five core navigation operations:
    - list_directory: List files and folders
    - read_file: Read file contents with progressive disclosure
    - grep_search: Search for patterns in files
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

        # Read file content
        try:
            # First check if it looks like binary by reading first block
            with open(full_path, "rb") as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
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

        # Headers only mode for large files
        if headers_only and total_lines > 500:
            headers = self._extract_headers(lines)
            header_content = "\n".join(f"{'#' * h['level']} {h['text']}" for h in headers)
            return {
                "content": header_content,
                "total_lines": total_lines,
                "is_partial": True,
                "headers": headers,
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
            }

        # Full file mode
        return {
            "content": content,
            "total_lines": total_lines,
            "is_partial": False,
            "headers": [],
        }

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
    ) -> list[dict[str, Any]]:
        """Search for pattern in files.

        Args:
            pattern: Regex pattern to search (case-insensitive)
            path: Directory to search in
            file_pattern: Glob pattern for files to search
            max_results: Maximum number of results to return

        Returns:
            List of matches, each with:
            - file: str (relative path)
            - line_number: int
            - content: str (matching line)
            - context: str (surrounding lines)

        Raises:
            ValueError: If path doesn't exist
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        results: list[dict[str, Any]] = []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        # Find matching files
        if full_path.is_file():
            files = [full_path]
        else:
            files = list(full_path.rglob(file_pattern))

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            lines = content.split("\n")

            for i, line in enumerate(lines):
                if regex.search(line):
                    # Get context (1 line before and after)
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 2)
                    context_lines = lines[context_start:context_end]

                    # Get relative path
                    try:
                        rel_path = file_path.relative_to(self.prepared_path)
                    except ValueError:
                        rel_path = file_path

                    results.append(
                        {
                            "file": str(rel_path),
                            "line_number": i + 1,
                            "content": line,
                            "context": "\n".join(context_lines),
                        }
                    )

                    if len(results) >= max_results:
                        return results

        return results

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
                        "Search for a pattern in files. Returns matching lines with context."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern to search (case-insensitive)",
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
                        },
                        "required": ["pattern"],
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
