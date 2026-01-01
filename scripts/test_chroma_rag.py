"""Quick test script for ChromaDB RAG implementation."""

from pathlib import Path

from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG


def main() -> None:
    """Test ChromaDB RAG with sample documents."""
    print("=" * 60)
    print("Testing ChromaDB Semantic Search RAG")
    print("=" * 60)

    # Initialize RAG
    print("\n1. Initializing ChromaDB RAG...")
    rag = ChromaSemanticRAG(collection_name="test_demo")
    print(f"   [OK] Initialized: {rag.name}")

    # Check if sample documents exist
    sample_dir = Path("data/raw")
    sample_files = list(sample_dir.glob("*.txt"))

    if not sample_files:
        print(f"\n   [ERROR] No sample documents found in {sample_dir}")
        print("   Please ensure sample documents exist in data/raw/")
        return

    print(f"   [OK] Found {len(sample_files)} sample documents")

    # Prepare documents
    print(f"\n2. Preparing documents from {sample_dir}...")
    try:
        rag.prepare_documents(str(sample_dir))
        print("   [OK] Documents indexed successfully")
    except Exception as e:
        print(f"   [ERROR] Error preparing documents: {e}")
        return

    # Get initial metrics
    metrics = rag.get_metrics()
    print("\n3. Initial Metrics:")
    print(f"   - Total chunks: {metrics['total_chunks']}")
    print(f"   - Collection: {metrics['collection_name']}")

    # Test queries
    test_questions = [
        "What is RAG?",
        "What are the main steps in RAG?",
        "What vector databases are mentioned?",
    ]

    print(f"\n4. Testing {len(test_questions)} queries...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Query {i}: {question}")
        try:
            result = rag.query(question, top_k=3)
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Chunks retrieved: {result['metadata']['chunks_retrieved']}")
            print(f"   Retrieval time: {result['metadata']['retrieval_time']:.3f}s")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")

    # Final metrics
    final_metrics = rag.get_metrics()
    print("\n5. Final Metrics:")
    print(f"   - Total queries: {final_metrics['total_queries']}")
    print(f"   - Avg retrieval time: {final_metrics['avg_retrieval_time']:.3f}s")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
