"""Unit tests for Filesystem RAG index builder."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_evaluator.rag_implementations.filesystem_rag.preparation.analyzer import (
    DocumentAnalysis,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.document_processor import (
    ProcessedDocument,
)
from rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder import (
    DocumentInfo,
    _classify_document_topics,  # Accessing private function for direct testing
    build_all_indexes,
    build_entity_registry,
    build_question_seeds,
    build_timeline,
    build_topic_map,
    write_document_files,
)


class TestIndexBuilder(unittest.TestCase):
    def setUp(self) -> None:
        # Create mock ProcessedDocument
        self.mock_doc_1 = ProcessedDocument(
            id="doc_001",
            original_path="path/to/doc1.txt",
            original_format="txt",
            markdown_content="# Doc 1",
            title="Document 1",
            word_count=100,
            char_count=500,
            line_count=10,
            sections=[{"title": "Section 1", "start_line": 1, "end_line": 10}],
        )
        self.mock_doc_2 = ProcessedDocument(
            id="doc_002",
            original_path="path/to/doc2.txt",
            original_format="txt",
            markdown_content="# Doc 2",
            title="Document 2",
            word_count=200,
            char_count=1000,
            line_count=20,
            sections=[{"title": "Section 1", "start_line": 1, "end_line": 20}],
        )

        # Create mock DocumentAnalysis
        self.mock_analysis_1 = DocumentAnalysis(
            summary="Summary 1",
            topics=["technical", "ai"],
            topic_scores={"technical": 0.8, "business": 0.1, "science": 0.1, "general": 0.0},
            entities={
                "people": ["Alice"],
                "concepts": ["RAG"],
                "organizations": [],
                "products": [],
            },
            temporal_markers=[{"date": "2023-01-01", "event": "Start"}],
            question_seeds=["What is RAG?"],
            key_sections=[{"title": "Section 1", "summary": "Key section"}],
            related_topics=[],
        )
        self.mock_analysis_2 = DocumentAnalysis(
            summary="Summary 2",
            topics=["business"],
            topic_scores={"technical": 0.2, "business": 0.6, "science": 0.1, "general": 0.1},
            entities={
                "people": ["Bob"],
                "concepts": ["ROI"],
                "organizations": ["Corp"],
                "products": [],
            },
            temporal_markers=[{"date": "2023-12-31", "event": "End"}],
            question_seeds=["How to calculate ROI?"],
            key_sections=[],
            related_topics=[],
        )

        self.doc_infos = [
            DocumentInfo(doc=self.mock_doc_1, analysis=self.mock_analysis_1),
            DocumentInfo(doc=self.mock_doc_2, analysis=self.mock_analysis_2),
        ]

    def test_classify_document_topics(self) -> None:
        """Test classification of documents into primary and secondary topics."""
        # Case 1: clear winner
        scores = {"technical": 0.8, "business": 0.1}
        primary, secondary = _classify_document_topics(scores)
        self.assertEqual(primary, ["technical"])
        self.assertEqual(secondary, [])

        # Case 2: split
        scores = {"technical": 0.5, "business": 0.45}
        primary, secondary = _classify_document_topics(scores, primary_threshold=0.4)
        self.assertIn("technical", primary)
        self.assertIn("business", primary)
        self.assertEqual(secondary, [])

        # Case 3: secondary
        # Assuming original logic: if none >= primary, take max as primary.
        # Max is technical (0.3).
        # Secondary condition: score >= 0.2. Technical(0.3) is >= 0.2, so it would also be in secondary
        # UNLESS the fix we implemented (remove from secondary if forced primary) works.
        scores = {"technical": 0.3, "business": 0.2}
        primary, secondary = _classify_document_topics(
            scores, primary_threshold=0.4, secondary_threshold=0.2
        )
        # Should force at least one primary if none meet threshold
        self.assertEqual(primary, ["technical"])
        # Technical should NOT be in secondary due to the fix
        self.assertNotIn("technical", secondary)
        self.assertEqual(secondary, ["business"])

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_build_topic_map(self, mock_write_text: MagicMock, mock_mkdir: MagicMock) -> None:
        """Test generation of topic map and topic files."""
        output_path = Path("/mock/output")

        topic_docs = build_topic_map(self.doc_infos, output_path)

        # Check if topic docs were correctly categorized
        self.assertIn("doc_001", topic_docs["technical"]["primary"])
        self.assertIn("doc_002", topic_docs["business"]["primary"])

        # Check if directories were created
        mock_mkdir.assert_called()

        # Check if write_text was called
        # We assume calls were made, exact number depends on active topics
        self.assertTrue(mock_write_text.called)

        # Since logic is correct we essentially want to ensure no exceptions and logical grouping returns are correct.
        self.assertEqual(len(topic_docs["technical"]["primary"]), 1)
        self.assertEqual(len(topic_docs["business"]["primary"]), 1)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_build_entity_registry(self, mock_write_text: MagicMock, mock_mkdir: MagicMock) -> None:
        """Test generation of entity registry."""
        output_path = Path("/mock/output")

        entity_docs = build_entity_registry(self.doc_infos, output_path)

        # Start verification
        self.assertIn("people", entity_docs)
        self.assertIn("Alice", entity_docs["people"])
        self.assertIn("doc_001", entity_docs["people"]["Alice"])

        self.assertIn("Bob", entity_docs["people"])
        self.assertIn("doc_002", entity_docs["people"]["Bob"])

        # Verify calls
        mock_mkdir.assert_called()
        self.assertTrue(mock_write_text.called)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_build_question_seeds(self, mock_write_text: MagicMock, mock_mkdir: MagicMock) -> None:
        """Test generation of question seeds."""
        output_path = Path("/mock/output")

        categorized = build_question_seeds(self.doc_infos, output_path)

        # "What is RAG?" -> Factual
        self.assertIn("factual", categorized)
        # Check if tuple ('What is RAG?', 'doc_001') is in list
        factual_questions = [q[0] for q in categorized["factual"]]
        self.assertIn("What is RAG?", factual_questions)

        # "How to calculate ROI?" -> How To
        how_to_questions = [q[0] for q in categorized["how_to"]]
        self.assertIn("How to calculate ROI?", how_to_questions)

        mock_mkdir.assert_called()
        self.assertTrue(mock_write_text.called)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_build_timeline(self, mock_write_text: MagicMock, mock_mkdir: MagicMock) -> None:
        """Test generation of timeline."""
        output_path = Path("/mock/output")

        timeline = build_timeline(self.doc_infos, output_path)

        # Should be sorted by date
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0]["date"], "2023-01-01")
        self.assertEqual(timeline[1]["date"], "2023-12-31")

        mock_mkdir.assert_called()
        self.assertTrue(mock_write_text.called)

    @patch(
        "rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder.build_topic_map"
    )
    @patch(
        "rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder.build_entity_registry"
    )
    @patch(
        "rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder.build_question_seeds"
    )
    @patch(
        "rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder.build_timeline"
    )
    def test_build_all_indexes(
        self,
        mock_timeline: MagicMock,
        mock_questions: MagicMock,
        mock_entities: MagicMock,
        mock_topics: MagicMock,
    ) -> None:
        """Test build_all_indexes calls all sub-builders."""
        output_path = Path("/mock/output")

        # Setup expected return values
        mock_timeline.return_value = []
        mock_questions.return_value = {}
        mock_entities.return_value = {}
        mock_topics.return_value = {}

        result = build_all_indexes(self.doc_infos, output_path)

        mock_topics.assert_called_once()
        mock_entities.assert_called_once()
        mock_questions.assert_called_once()
        mock_timeline.assert_called_once()

        self.assertIn("topics", result)
        self.assertIn("entities", result)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("rag_evaluator.rag_implementations.filesystem_rag.preparation.index_builder.os.replace")
    def test_write_document_files(
        self,
        mock_replace: MagicMock,
        mock_write_text: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        """Test writing of document, metadata, and summary files."""
        output_path = Path("/mock/output")

        write_document_files(self.doc_infos, output_path)

        mock_mkdir.assert_called()
        # Should write 3 files per doc: .md, .meta.json, _summary.md
        # 2 docs * 3 = 6 calls
        self.assertEqual(mock_write_text.call_count, 6)
        self.assertEqual(mock_replace.call_count, 6)
