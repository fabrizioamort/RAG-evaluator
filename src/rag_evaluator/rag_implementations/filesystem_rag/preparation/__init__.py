"""Filesystem RAG preparation pipeline components."""

from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
    DocumentAnalysis,
    analyze_document,
    heuristic_analysis,
    llm_analysis,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
    RawDocument,
    convert_to_markdown,
    detect_txt_structure,
    load_documents,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder import (
    DocumentInfo,
    build_all_indexes,
    build_entity_registry,
    build_question_seeds,
    build_timeline,
    build_topic_map,
    write_document_files,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.pipeline import (
    PreparationMetrics,
    PreparationPipeline,
    run_preparation,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.synthesizer import (
    copy_original_files,
    generate_corpus_overview,
    generate_navigation_guide,
    generate_statistics,
    synthesize_all,
)

__all__ = [
    # document_processor
    "RawDocument",
    "ProcessedDocument",
    "load_documents",
    "convert_to_markdown",
    "detect_txt_structure",
    # analyzer
    "DocumentAnalysis",
    "analyze_document",
    "heuristic_analysis",
    "llm_analysis",
    # index_builder
    "DocumentInfo",
    "build_topic_map",
    "build_entity_registry",
    "build_question_seeds",
    "build_timeline",
    "build_all_indexes",
    "write_document_files",
    # synthesizer
    "generate_corpus_overview",
    "generate_navigation_guide",
    "generate_statistics",
    "copy_original_files",
    "synthesize_all",
    # pipeline
    "PreparationMetrics",
    "PreparationPipeline",
    "run_preparation",
]
