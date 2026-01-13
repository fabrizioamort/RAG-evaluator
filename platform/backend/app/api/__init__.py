"""API routes for the RAG Evaluation Platform."""

from app.api import health, knowledge_bases, projects, test_sets, test_templates

__all__ = [
    "health",
    "projects",
    "knowledge_bases",
    "test_sets",
    "test_templates",
]
