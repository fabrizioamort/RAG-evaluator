"""Integration tests for Filesystem RAG preparation pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag_evaluator.rag_implementations.filesystem_rag.preparation.pipeline import (
    PreparationPipeline,
)


@pytest.mark.integration
def test_full_pipeline_with_real_data() -> None:
    """Test full preparation pipeline using real data from data/raw.

    This test confirms that:
    1. Documents are loaded from data/raw
    2. Setup runs without error
    3. Output files are generated in the correct structure
    """
    input_path = Path("data/raw")

    # Ensure input data exists
    # Ensure input data exists (ignoring .gitkeep and hidden files)
    if not input_path.exists() or not any(
        f.name != ".gitkeep" and not f.name.startswith(".") for f in input_path.iterdir()
    ):
        pytest.skip("No actual data found in data/raw for integration test")

    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "prepared_output"

        # Run pipeline with heuristic analysis to avoid API costs during test
        pipeline = PreparationPipeline(
            input_path=str(input_path),
            output_path=str(output_path),
            force_analysis_method="heuristic",
            word_threshold=1000,
        )

        result = pipeline.run()

        # Verify execution success
        metrics = result["metrics"]
        assert metrics.validation_passed, f"Validation failed: {metrics.validation_errors}"
        assert metrics.total_documents > 0

        # Verify directory structure
        assert (output_path / "documents").exists()
        assert (output_path / "_index" / "topics").exists()
        assert (output_path / "_index" / "entities").exists()
        assert (output_path / "_index" / "passages").exists()
        assert (output_path / "_meta").exists()

        # Verify key files
        assert (output_path / "_meta" / "corpus_overview.md").exists()
        assert (output_path / "_index" / "topics" / "_topic_map.md").exists()
        assert (output_path / "_index" / "passages" / "bm25.json").exists()

        # Verify document generation (at least one)
        docs_dir = output_path / "documents"
        md_files = list(docs_dir.glob("*.md"))
        json_files = list(docs_dir.glob("*.meta.json"))
        assert len(md_files) == len(json_files)
        assert len(md_files) > 0
