"""Evaluation module for RAG implementations."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_evaluator.evaluation.evaluator import RAGEvaluator
    from rag_evaluator.evaluation.report_generator import ReportGenerator

__all__ = ["RAGEvaluator", "ReportGenerator"]


def __getattr__(name: str) -> Any:
    """Lazy-load public evaluation symbols to avoid import-time side effects."""
    if name == "RAGEvaluator":
        from rag_evaluator.evaluation.evaluator import RAGEvaluator

        return RAGEvaluator
    if name == "ReportGenerator":
        from rag_evaluator.evaluation.report_generator import ReportGenerator

        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
