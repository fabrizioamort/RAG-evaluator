import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from rag_evaluator.common.document_loaders import (
    DocumentLoader, TXTLoader, PDFLoader, DOCXLoader,
    create_loader, Document
)
import pypdf

def test_txt_loader(tmp_path: Path) -> None:
    """Test TXT loader"""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Test content", encoding="utf-8")

    loader = TXTLoader()
    doc = loader.load(str(txt_file))

    assert doc.content == "Test content"
    assert doc.metadata["format"] == "txt"
    assert doc.source == str(txt_file)
    assert doc.metadata["size"] == 12

def test_loader_factory(tmp_path: Path) -> None:
    """Test loader factory"""
    assert isinstance(create_loader("test.txt"), TXTLoader)
    assert isinstance(create_loader("test.pdf"), PDFLoader)
    assert isinstance(create_loader("test.docx"), DOCXLoader)

    with pytest.raises(ValueError):
        create_loader("test.xyz")

def test_pdf_loader_mock() -> None:
    """Test PDF loader with mock pypdf"""
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        
        # Mock pages
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 Content"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 Content"
        
        mock_reader.pages = [page1, page2]
        
        # Mock metadata
        mock_reader.metadata.title = "Test PDF"
        mock_reader.metadata.author = "Test Author"
        
        mock_reader_cls.return_value = mock_reader
        
        loader = PDFLoader()
        # We need a real file path even for mock, or mock open
        with patch("builtins.open", MagicMock()):
            doc = loader.load("dummy.pdf")
            
        assert "Page 1 Content" in doc.content
        assert "Page 2 Content" in doc.content
        assert doc.metadata["format"] == "pdf"
        assert doc.metadata["title"] == "Test PDF"
        assert doc.metadata["author"] == "Test Author"
        assert doc.metadata["pages"] == 2

def test_docx_loader_mock() -> None:
    """Test DOCX loader with mock python-docx"""
    with patch("rag_evaluator.common.document_loaders.DocxDocument") as mock_docx_cls:
        mock_doc = MagicMock()
        
        # Mock paragraphs
        p1 = MagicMock()
        p1.text = "Paragraph 1"
        p2 = MagicMock()
        p2.text = "Paragraph 2"
        mock_doc.paragraphs = [p1, p2]
        
        # Mock tables
        mock_doc.tables = []
        
        # Mock core properties
        mock_doc.core_properties.title = "Test DOCX"
        mock_doc.core_properties.author = "Test Author"
        
        mock_docx_cls.return_value = mock_doc
        
        loader = DOCXLoader()
        doc = loader.load("dummy.docx")
        
        assert "Paragraph 1" in doc.content
        assert "Paragraph 2" in doc.content
        assert doc.metadata["format"] == "docx"
        assert doc.metadata["title"] == "Test DOCX"
        assert doc.metadata["author"] == "Test Author"

def test_pdf_encrypted_mock() -> None:
    """Test encrypted PDF handling"""
    with patch("pypdf.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        
        # Mock decrypt failing
        mock_reader.decrypt.return_value = 0 # 0 means fail
        
        mock_reader_cls.return_value = mock_reader
        
        loader = PDFLoader()
        with patch("builtins.open", MagicMock()):
            with pytest.raises(ValueError, match="encrypted"):
                loader.load("encrypted.pdf")
