"""Preparation pipeline for Filesystem RAG.

This module orchestrates the complete preparation workflow:
1. Load raw documents
2. Convert to markdown
3. Analyze documents (hybrid: heuristic + LLM)
4. Build indexes
5. Generate synthesis files
6. Validate output
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
    DocumentAnalysis,
    analyze_document,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
    RawDocument,
    convert_to_markdown,
    load_documents,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder import (
    DocumentInfo,
    build_all_indexes,
    write_document_files,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.synthesizer import (
    synthesize_all,
)


@dataclass
class PreparationMetrics:
    """Metrics collected during preparation.

    Attributes:
        total_documents: Number of documents processed
        documents_by_format: Count by format (pdf, txt, docx)
        total_words: Total word count across all documents
        documents_analyzed_heuristic: Count analyzed with heuristics
        documents_analyzed_llm: Count analyzed with LLM
        preparation_time_seconds: Total preparation time
        estimated_cost_usd: Estimated LLM API cost
        validation_passed: Whether validation succeeded
        validation_errors: List of validation errors if any
    """

    total_documents: int = 0
    documents_by_format: dict[str, int] = field(default_factory=dict)
    total_words: int = 0
    documents_analyzed_heuristic: int = 0
    documents_analyzed_llm: int = 0
    preparation_time_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)


# Estimated costs per document for LLM analysis (gpt-4o-mini)
LLM_COST_PER_DOC_USD = 0.015


class PreparationPipeline:
    """Orchestrates the complete filesystem preparation workflow.

    Usage:
        pipeline = PreparationPipeline(
            input_path="data/raw",
            output_path="data/prepared/filesystem_rag"
        )
        metrics = pipeline.run()
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        word_threshold: int = 1000,
        use_llm_synthesis: bool = False,
        preserve_originals: bool = True,
        force_analysis_method: str | None = None,
    ) -> None:
        """Initialize the preparation pipeline.

        Args:
            input_path: Path to directory containing raw documents
            output_path: Path for prepared filesystem output
            word_threshold: Word count threshold for LLM vs heuristic analysis
            use_llm_synthesis: Whether to use LLM for corpus overview synthesis
            preserve_originals: Whether to copy original files to _original/
            force_analysis_method: Force "heuristic" or "llm" for all documents
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.word_threshold = word_threshold
        self.use_llm_synthesis = use_llm_synthesis
        self.preserve_originals = preserve_originals
        self.force_analysis_method = force_analysis_method

        self._client: OpenAI | None = None
        self._metrics = PreparationMetrics()
        self._raw_documents: list[RawDocument] = []
        self._processed_documents: list[ProcessedDocument] = []
        self._analyses: list[DocumentAnalysis] = []
        self._document_infos: list[DocumentInfo] = []

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout,
            )
        return self._client

    def run(self) -> dict[str, Any]:
        """Execute the complete preparation pipeline.

        Returns:
            Dictionary containing preparation metrics and results
        """
        start_time = time.time()
        print(f"\n{'=' * 60}")
        print("FILESYSTEM RAG PREPARATION PIPELINE")
        print(f"{'=' * 60}")
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_path}")
        print(f"{'=' * 60}\n")

        try:
            # Step 0: Ensure output directory exists. Forced rebuilds are
            # handled by callers that own the storage path.
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Load documents
            self._step_load_documents()

            # Step 2: Convert to markdown
            self._step_convert_to_markdown()

            # Step 3: Analyze documents
            self._step_analyze_documents()

            # Step 4: Build document infos
            self._step_build_document_infos()

            # Step 5: Write document files
            self._step_write_document_files()

            # Step 6: Build indexes
            self._step_build_indexes()

            # Step 7: Generate synthesis files
            self._step_synthesize()

            # Step 8: Validate output
            self._step_validate()

        except Exception as e:
            print(f"\nERROR: Pipeline failed: {e}")
            self._metrics.validation_passed = False
            self._metrics.validation_errors.append(f"Pipeline error: {str(e)}")

        # Calculate final metrics
        self._metrics.preparation_time_seconds = time.time() - start_time

        self._print_summary()

        return {
            "metrics": self._metrics,
            "output_path": str(self.output_path),
            "documents_processed": len(self._processed_documents),
        }

    def _step_load_documents(self) -> None:
        """Step 1: Load raw documents from input directory."""
        print("Step 1/8: Loading documents...")
        self._raw_documents = load_documents(str(self.input_path))
        self._metrics.total_documents = len(self._raw_documents)

        # Count by format
        for doc in self._raw_documents:
            fmt = doc.original_format
            self._metrics.documents_by_format[fmt] = (
                self._metrics.documents_by_format.get(fmt, 0) + 1
            )

        print(f"  Loaded {len(self._raw_documents)} documents\n")

    def _step_convert_to_markdown(self) -> None:
        """Step 2: Convert raw documents to markdown."""
        print("Step 2/8: Converting to markdown...")

        for raw_doc in self._raw_documents:
            # Get metadata from raw document
            metadata = {
                "format": raw_doc.original_format,
                "file_size": raw_doc.file_size,
            }

            processed = convert_to_markdown(raw_doc, metadata)
            self._processed_documents.append(processed)
            self._metrics.total_words += processed.word_count

        print(f"  Converted {len(self._processed_documents)} documents")
        print(f"  Total words: {self._metrics.total_words:,}\n")

    def _step_analyze_documents(self) -> None:
        """Step 3: Analyze documents using hybrid approach."""
        print("Step 3/8: Analyzing documents...")

        client = self._get_client() if self.force_analysis_method != "heuristic" else None

        for processed in self._processed_documents:
            analysis = analyze_document(
                processed,
                force_method=self.force_analysis_method,
                word_threshold=self.word_threshold,
                client=client,
            )
            self._analyses.append(analysis)

            # Track analysis method
            if analysis.analysis_method == "heuristic":
                self._metrics.documents_analyzed_heuristic += 1
            else:
                self._metrics.documents_analyzed_llm += 1
                self._metrics.estimated_cost_usd += LLM_COST_PER_DOC_USD

        print(f"  Analyzed {len(self._analyses)} documents")
        print(f"  Heuristic: {self._metrics.documents_analyzed_heuristic}")
        print(f"  LLM: {self._metrics.documents_analyzed_llm}")
        print(f"  Estimated cost: ${self._metrics.estimated_cost_usd:.2f}\n")

    def _step_build_document_infos(self) -> None:
        """Step 4: Combine processed documents with analyses."""
        print("Step 4/8: Building document info objects...")

        for processed, analysis in zip(self._processed_documents, self._analyses):
            doc_info = DocumentInfo(doc=processed, analysis=analysis)
            self._document_infos.append(doc_info)

        print(f"  Created {len(self._document_infos)} document infos\n")

    def _step_write_document_files(self) -> None:
        """Step 5: Write document markdown and metadata files."""
        print("Step 5/8: Writing document files...")

        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)

        write_document_files(self._document_infos, self.output_path)
        print()

    def _step_build_indexes(self) -> None:
        """Step 6: Build all index files."""
        print("Step 6/8: Building indexes...")

        build_all_indexes(self._document_infos, self.output_path)
        print()

    def _step_synthesize(self) -> None:
        """Step 7: Generate synthesis files."""
        print("Step 7/8: Generating synthesis files...")

        client = self._get_client() if self.use_llm_synthesis else None

        # Add cost for LLM synthesis if used
        if self.use_llm_synthesis:
            self._metrics.estimated_cost_usd += 0.15  # Approximate cost

        synthesize_all(
            self._document_infos,
            self.output_path,
            preparation_time=0.0,  # Will be updated at end
            preparation_cost=self._metrics.estimated_cost_usd,
            use_llm_synthesis=self.use_llm_synthesis,
            preserve_originals=self.preserve_originals,
            client=client,
        )
        print()

    def _step_validate(self) -> None:
        """Step 8: Validate the prepared filesystem."""
        print("Step 8/8: Validating output...")

        errors: list[str] = []

        # Check required directories exist
        required_dirs = [
            "_meta",
            "_index/topics",
            "_index/entities",
            "_index/temporal",
            "_index/questions",
            "_index/passages",
            "_summaries",
            "documents",
        ]

        for dir_name in required_dirs:
            dir_path = self.output_path / dir_name
            if not dir_path.exists():
                errors.append(f"Missing directory: {dir_name}")

        # Check required meta files
        required_files = [
            "_meta/corpus_overview.md",
            "_meta/navigation_guide.md",
            "_meta/statistics.json",
            "_index/topics/_topic_map.md",
            "_index/entities/_entity_registry.md",
            "_index/questions/question_seeds.md",
            "_index/passages/bm25.json",
            "_index/temporal/timeline.md",
        ]

        for file_name in required_files:
            file_path = self.output_path / file_name
            if not file_path.exists():
                errors.append(f"Missing file: {file_name}")

        # Check each document has required files
        docs_dir = self.output_path / "documents"
        summaries_dir = self.output_path / "_summaries"

        for doc_info in self._document_infos:
            doc_id = doc_info.doc.id

            # Check markdown file
            md_path = docs_dir / f"{doc_id}.md"
            if not md_path.exists():
                errors.append(f"Missing document file: {doc_id}.md")

            # Check metadata file
            meta_path = docs_dir / f"{doc_id}.meta.json"
            if not meta_path.exists():
                errors.append(f"Missing metadata file: {doc_id}.meta.json")

            # Check summary file
            summary_path = summaries_dir / f"{doc_id}_summary.md"
            if not summary_path.exists():
                errors.append(f"Missing summary file: {doc_id}_summary.md")

        if errors:
            self._metrics.validation_passed = False
            self._metrics.validation_errors = errors
            print(f"  VALIDATION FAILED: {len(errors)} errors found")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more errors")
        else:
            self._metrics.validation_passed = True
            print("  Validation PASSED: All files present")

        print()

    def _print_summary(self) -> None:
        """Print preparation summary."""
        print(f"{'=' * 60}")
        print("PREPARATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Documents processed: {self._metrics.total_documents}")
        print(f"Total words: {self._metrics.total_words:,}")
        print(f"Time elapsed: {self._metrics.preparation_time_seconds:.1f}s")
        print(f"Estimated cost: ${self._metrics.estimated_cost_usd:.2f}")
        print(f"Validation: {'PASSED' if self._metrics.validation_passed else 'FAILED'}")
        print(f"Output: {self.output_path}")
        print(f"{'=' * 60}\n")


def run_preparation(
    input_path: str,
    output_path: str = "data/prepared/filesystem_rag",
    word_threshold: int = 1000,
    use_llm_synthesis: bool = False,
    preserve_originals: bool = True,
    force_analysis_method: str | None = None,
) -> dict[str, Any]:
    """Convenience function to run the preparation pipeline.

    Args:
        input_path: Path to directory containing raw documents
        output_path: Path for prepared filesystem output
        word_threshold: Word count threshold for LLM vs heuristic analysis
        use_llm_synthesis: Whether to use LLM for corpus overview synthesis
        preserve_originals: Whether to copy original files to _original/
        force_analysis_method: Force "heuristic" or "llm" for all documents

    Returns:
        Dictionary containing preparation metrics and results
    """
    pipeline = PreparationPipeline(
        input_path=input_path,
        output_path=output_path,
        word_threshold=word_threshold,
        use_llm_synthesis=use_llm_synthesis,
        preserve_originals=preserve_originals,
        force_analysis_method=force_analysis_method,
    )
    return pipeline.run()
