from pathlib import Path

import pytest

from rag_evaluator.config import settings
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


@pytest.mark.skipif(not settings.openai_api_key, reason="OpenAI API key not set")
def test_chroma_rag_with_mixed_formats(tmp_path: Path) -> None:
    """Test ChromaRAG with TXT, PDF, and DOCX documents"""

    # Create dummy documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # TXT
    (docs_dir / "test.txt").write_text("This is a text file content.", encoding="utf-8")

    # Note: We can't easily create valid PDF/DOCX files from scratch without libraries
    # that might not be available or are complex to use just for creating test data.
    # So we will rely on mocking the loaders' return values during this integration test
    # OR we skip the actual file creation and just use TXT if we want to be safe,
    # BUT the point is to test loading.
    # A better approach for integration test without committing binary files is to use
    # the fact that we tested loaders in unit tests, and here we just verify that ChromaRAG
    # calls them.
    # However, to be a true integration test, we should ideally have real files.
    # Since I cannot assume I can validly generate PDFs/DOCX here easily, I will
    # rely on the fact that I verified loaders in unit tests with mocks.
    # I will just test that the integration logic works with TXT files for now,
    # and maybe trust the unit tests for the actual parsing.
    #
    # Wait, the prompt implies I effectively "implement ONLY Task 1".
    # I will add a simple test that uses the TXT file and verifies it loads.
    # If I had sample PDF/DOCX I would use them.
    # I will create a test that mocks `create_loader` to return a dummy loader for
    # .pdf and .docx extensions so we don't need real binary files, but we verify
    # the RAG class iterates and loads them.

    (docs_dir / "test.pdf").touch()
    (docs_dir / "test.docx").touch()

    # We patch create_loader to return a mock loader that returns a valid Document
    from unittest.mock import MagicMock, patch

    from rag_evaluator.common.document_loaders import Document

    with patch("rag_evaluator.rag_implementations.vector_semantic.chroma_rag.create_loader") as mock_create:
        mock_loader = MagicMock()
        mock_loader.load.return_value = Document(
            content="Mocked content",
            metadata={"format": "mock"},
            source="mock_source"
        )
        mock_create.return_value = mock_loader

        # Initialize RAG
        # Use a temporary persistence directory
        rag = ChromaSemanticRAG(persist_directory=str(tmp_path / "chroma"))

        # Prepare documents
        rag.prepare_documents(str(docs_dir))

        # Verify it tried to load all 3 files
        assert mock_create.call_count == 3

        # Query
        result = rag.query("content")
        assert result["answer"]
        assert len(result["context"]) > 0
