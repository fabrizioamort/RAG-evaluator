"""API routes for the RAG Evaluation Platform."""

from app.api import (
    evaluations,
    health,
    knowledge_bases,
    projects,
    rag_configs,
    test_sets,
    test_templates,
)

__all__ = [
    "health",
    "projects",
    "knowledge_bases",
    "test_sets",
    "test_templates",
    "rag_configs",
    "evaluations",
]
