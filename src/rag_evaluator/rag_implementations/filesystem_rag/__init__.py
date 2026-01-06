"""Filesystem-based RAG implementation.

This module provides a RAG implementation that uses LLM-guided agent
navigation of a prepared filesystem structure instead of vector
similarity search.
"""

from rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag import FilesystemRAG

__all__ = ["FilesystemRAG"]
