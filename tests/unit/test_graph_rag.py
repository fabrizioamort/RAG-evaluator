"""Unit tests for Graph RAG implementation."""

from unittest.mock import MagicMock, Mock, patch

import pytest

import rag_evaluator.rag_implementations.graph_rag.neo4j_rag as neo4j_rag_module
from rag_evaluator.rag_implementations.graph_rag import Neo4jGraphRAG


class TestNeo4jGraphRAG:
    """Test cases for Neo4jGraphRAG class."""

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_initialization(
        self,
        mock_graph_rag: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Test Neo4jGraphRAG initialization."""
        # Setup mocks
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        # Initialize
        rag = Neo4jGraphRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
        )

        # Verify
        assert rag.name == "Neo4j Graph RAG"
        assert rag.neo4j_uri == "bolt://localhost:7687"
        assert rag.neo4j_username == "neo4j"
        assert rag.neo4j_password == "password"

        # Verify driver was created
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "password")
        )
        mock_driver.verify_connectivity.assert_called_once()

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_initialization_blank_connection_values_fallback_to_settings(
        self,
        mock_graph_rag: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Blank connection params should fallback to settings values."""
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        with (
            patch.object(neo4j_rag_module.settings, "neo4j_uri", "bolt://env-host:7687"),
            patch.object(neo4j_rag_module.settings, "neo4j_username", "env-user"),
            patch.object(neo4j_rag_module.settings, "neo4j_password", "env-pass"),
        ):
            rag = Neo4jGraphRAG(
                neo4j_uri="   ",
                neo4j_username="",
                neo4j_password="   ",
            )

        assert rag.neo4j_uri == "bolt://env-host:7687"
        assert rag.neo4j_username == "env-user"
        assert rag.neo4j_password == "env-pass"
        mock_graph_db.driver.assert_called_once_with(
            "bolt://env-host:7687", auth=("env-user", "env-pass")
        )

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_initialization_raises_clear_error_when_neo4j_unreachable(
        self,
        mock_graph_rag: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Connection failures should raise a clear Neo4j-specific error."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.side_effect = RuntimeError("connection refused")
        mock_graph_db.driver.return_value = mock_driver

        with pytest.raises(RuntimeError, match="Cannot connect to Neo4j"):
            Neo4jGraphRAG(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphIndexer")
    def test_prepare_documents(
        self,
        mock_indexer_class: Mock,
        mock_graph_rag: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Test document preparation."""
        # Setup mocks
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        mock_indexer = MagicMock()
        mock_indexer.index_documents.return_value = {
            "documents_processed": 5,
            "sources": ["doc1.txt", "doc2.txt"],
            "total_nodes": 100,
            "total_relationships": 50,
            "node_labels": {"Entity": 50, "Concept": 30, "Person": 20},
        }
        mock_indexer_class.return_value = mock_indexer

        # Initialize RAG
        rag = Neo4jGraphRAG()

        # Prepare documents
        rag.prepare_documents("data/test")

        # Verify indexer was called
        mock_indexer.index_documents.assert_called_once_with("data/test")

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_query_success(
        self,
        mock_graph_rag_class: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Test successful query execution."""
        # Setup mocks
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock GraphRAG response
        mock_rag_pipeline = MagicMock()
        mock_graph_rag_class.return_value = mock_rag_pipeline

        # Mock retriever result
        mock_item1 = MagicMock()
        mock_item1.content = "Test content 1"
        mock_item1.metadata = {"entities": ["Entity1", "Entity2"], "related_entities": ["Related1"]}

        mock_item2 = MagicMock()
        mock_item2.content = "Test content 2"
        mock_item2.metadata = {"entities": [], "related_entities": []}

        mock_retriever_result = MagicMock()
        mock_retriever_result.items = [mock_item1, mock_item2]

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.retriever_result = mock_retriever_result

        mock_rag_pipeline.search.return_value = mock_response

        # Initialize RAG
        rag = Neo4jGraphRAG()

        # Query
        result = rag.query("What is this about?", top_k=5)

        # Verify
        assert result["answer"] == "Test answer"
        assert len(result["context"]) == 2
        assert "Entity1, Entity2" in result["context"][0]
        assert "Related1" in result["context"][0]
        assert result["metadata"]["chunks_retrieved"] == 2
        assert result["metadata"]["graph_enhanced"] is True
        assert "retrieval_time" in result["metadata"]

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_query_error_handling(
        self,
        mock_graph_rag_class: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Test query error handling."""
        # Setup mocks
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock GraphRAG to raise exception
        mock_rag_pipeline = MagicMock()
        mock_graph_rag_class.return_value = mock_rag_pipeline
        mock_rag_pipeline.search.side_effect = Exception("Test error")

        # Initialize RAG
        rag = Neo4jGraphRAG()

        # Query
        result = rag.query("What is this about?", top_k=5)

        # Verify error handling
        assert "Error querying graph RAG" in result["answer"]
        assert result["context"] == []
        assert result["metadata"]["chunks_retrieved"] == 0
        assert "error" in result["metadata"]

    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphDatabase")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAIEmbeddings")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.OpenAILLM")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.VectorCypherRetriever")
    @patch("rag_evaluator.rag_implementations.graph_rag.neo4j_rag.GraphRAG")
    def test_get_metrics(
        self,
        mock_graph_rag: Mock,
        mock_retriever: Mock,
        mock_llm: Mock,
        mock_embeddings: Mock,
        mock_graph_db: Mock,
    ) -> None:
        """Test metrics retrieval."""
        # Setup mocks
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock node count query
        mock_node_result = MagicMock()
        mock_node_result.single.return_value = {"count": 100}

        # Mock relationship count query
        mock_rel_result = MagicMock()
        mock_rel_result.single.return_value = {"count": 50}

        mock_session.run.side_effect = [mock_node_result, mock_rel_result]

        mock_graph_db.driver.return_value = mock_driver

        # Initialize RAG
        rag = Neo4jGraphRAG()

        # Get metrics
        metrics = rag.get_metrics()

        # Verify
        assert metrics["avg_retrieval_time"] == 0.0
        assert metrics["total_queries"] == 0
        assert metrics["total_nodes"] == 100
        assert metrics["total_relationships"] == 50
