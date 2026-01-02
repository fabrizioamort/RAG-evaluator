"""Integration tests for evaluation framework."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from rag_evaluator.evaluation.evaluator import RAGEvaluator
from rag_evaluator.evaluation.report_generator import ReportGenerator
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG

# Skip all tests if OpenAI API key is not set
pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")


@pytest.fixture
def temp_data_dirs():
    """Create temporary directories for testing.

    Yields:
        Tuple of (raw_dir, chroma_dir, reports_dir)
    """
    temp_dir = tempfile.mkdtemp()
    raw_dir = Path(temp_dir) / "raw"
    chroma_dir = Path(temp_dir) / "chroma_db"
    reports_dir = Path(temp_dir) / "reports"

    raw_dir.mkdir()
    chroma_dir.mkdir()
    reports_dir.mkdir()

    yield raw_dir, chroma_dir, reports_dir

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except (PermissionError, OSError):
        # On Windows, ChromaDB might hold file locks
        pass


@pytest.fixture
def sample_documents(temp_data_dirs):
    """Create sample documents for testing.

    Args:
        temp_data_dirs: Temporary directories fixture

    Returns:
        Path to raw documents directory
    """
    raw_dir, _, _ = temp_data_dirs

    # Create sample document 1
    doc1 = raw_dir / "sample1.txt"
    doc1.write_text(
        """Vector databases are specialized systems for storing and searching embeddings.
        They use techniques like approximate nearest neighbor (ANN) for efficient similarity search.
        Popular vector databases include ChromaDB, Pinecone, and Weaviate.""",
        encoding="utf-8",
    )

    # Create sample document 2
    doc2 = raw_dir / "sample2.txt"
    doc2.write_text(
        """RAG (Retrieval Augmented Generation) combines language models with external knowledge.
        It retrieves relevant context from a knowledge base before generating answers.
        This helps reduce hallucinations and provides more accurate responses.""",
        encoding="utf-8",
    )

    return raw_dir


@pytest.fixture
def mini_test_set(temp_data_dirs):
    """Create a minimal test set for integration testing.

    Args:
        temp_data_dirs: Temporary directories fixture

    Returns:
        Path to test set JSON file
    """
    raw_dir, _, _ = temp_data_dirs

    test_set = {
        "metadata": {"version": "1.0", "description": "Mini test set"},
        "test_cases": [
            {
                "id": "tc_int_001",
                "question": "What is RAG?",
                "expected_answer": "RAG (Retrieval Augmented Generation) combines language models with external knowledge retrieval.",
                "ground_truth_context": ["RAG combines language models with external knowledge"],
                "difficulty": "easy",
                "category": "definition",
            },
            {
                "id": "tc_int_002",
                "question": "What are some popular vector databases?",
                "expected_answer": "Popular vector databases include ChromaDB, Pinecone, and Weaviate.",
                "ground_truth_context": ["ChromaDB, Pinecone, and Weaviate"],
                "difficulty": "easy",
                "category": "factual_recall",
            },
        ],
    }

    test_set_path = raw_dir.parent / "mini_test_set.json"
    with open(test_set_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f)

    return test_set_path


@pytest.mark.integration
def test_end_to_end_evaluation(sample_documents, mini_test_set, temp_data_dirs):
    """Test complete evaluation workflow end-to-end.

    Args:
        sample_documents: Sample documents fixture
        mini_test_set: Mini test set fixture
        temp_data_dirs: Temporary directories fixture
    """
    raw_dir, chroma_dir, reports_dir = temp_data_dirs

    # Initialize ChromaDB RAG
    rag = ChromaSemanticRAG(
        collection_name="test_integration",
        persist_directory=str(chroma_dir),
    )

    # Prepare documents
    rag.prepare_documents(str(raw_dir))

    # Initialize evaluator
    evaluator = RAGEvaluator(test_set_path=str(mini_test_set))

    # Run evaluation (non-verbose to avoid cluttering test output)
    results = evaluator.evaluate(rag, verbose=False)

    # Verify results structure
    assert results["rag_implementation"] == "ChromaDB Semantic Search"
    assert results["test_cases_count"] == 2
    assert "timestamp" in results
    assert "metrics_summary" in results
    assert "detailed_results" in results
    assert "performance_metrics" in results
    assert "pass_rate" in results

    # Verify metrics are calculated
    metrics_summary = results["metrics_summary"]
    assert "faithfulness_avg" in metrics_summary
    assert "answer_relevancy_avg" in metrics_summary
    assert "contextual_precision_avg" in metrics_summary
    assert "contextual_recall_avg" in metrics_summary
    assert "hallucination_avg" in metrics_summary

    # Verify detailed results
    assert len(results["detailed_results"]) == 2
    for detail in results["detailed_results"]:
        assert "test_case_id" in detail
        assert "question" in detail
        assert "answer" in detail
        assert "metrics" in detail


@pytest.mark.integration
def test_report_generation_integration(sample_documents, mini_test_set, temp_data_dirs):
    """Test report generation with real evaluation results.

    Args:
        sample_documents: Sample documents fixture
        mini_test_set: Mini test set fixture
        temp_data_dirs: Temporary directories fixture
    """
    raw_dir, chroma_dir, reports_dir = temp_data_dirs

    # Initialize and prepare RAG
    rag = ChromaSemanticRAG(
        collection_name="test_report_gen",
        persist_directory=str(chroma_dir),
    )
    rag.prepare_documents(str(raw_dir))

    # Run evaluation
    evaluator = RAGEvaluator(test_set_path=str(mini_test_set))
    results = evaluator.evaluate(rag, verbose=False)

    # Generate reports
    report_gen = ReportGenerator(output_dir=str(reports_dir))
    files = report_gen.generate_report(results, output_format="both")

    # Verify files were created
    assert "json" in files
    assert "markdown" in files

    json_path = Path(files["json"])
    md_path = Path(files["markdown"])

    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON content
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)
        assert json_data["rag_implementation"] == "ChromaDB Semantic Search"
        assert json_data["test_cases_count"] == 2

    # Verify Markdown content
    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()
        assert "# RAG Evaluation Report" in md_content
        assert "ChromaDB Semantic Search" in md_content
        assert "tc_int_001" in md_content
        assert "tc_int_002" in md_content


@pytest.mark.integration
@pytest.mark.slow
def test_comparison_integration(sample_documents, mini_test_set, temp_data_dirs):
    """Test comparing multiple RAG implementations.

    Args:
        sample_documents: Sample documents fixture
        mini_test_set: Mini test set fixture
        temp_data_dirs: Temporary directories fixture

    Note:
        This test is marked as slow because it evaluates multiple implementations
    """
    raw_dir, chroma_dir, reports_dir = temp_data_dirs

    # Initialize two RAG instances with different collection names
    rag1 = ChromaSemanticRAG(
        collection_name="test_compare_1",
        persist_directory=str(chroma_dir),
    )
    rag1.prepare_documents(str(raw_dir))

    rag2 = ChromaSemanticRAG(
        collection_name="test_compare_2",
        persist_directory=str(chroma_dir),
    )
    rag2.prepare_documents(str(raw_dir))

    # Modify second RAG's name for comparison
    rag2.name = "ChromaDB Variant"

    # Run comparison
    evaluator = RAGEvaluator(test_set_path=str(mini_test_set))
    comparison = evaluator.compare_implementations([rag1, rag2], verbose=False)

    # Verify comparison results
    assert len(comparison) == 2
    assert "ChromaDB Semantic Search" in comparison
    assert "ChromaDB Variant" in comparison

    # Generate comparison report
    report_gen = ReportGenerator(output_dir=str(reports_dir))
    files = report_gen.generate_comparison_report(comparison, output_format="both")

    assert "json" in files
    assert "markdown" in files
    assert Path(files["json"]).exists()
    assert Path(files["markdown"]).exists()

    # Verify comparison markdown content
    with open(files["markdown"], encoding="utf-8") as f:
        content = f.read()
        assert "Comparison Report" in content
        assert "ChromaDB Semantic Search" in content
        assert "ChromaDB Variant" in content
