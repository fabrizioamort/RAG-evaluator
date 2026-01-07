"""Clean up data and databases for a fresh start."""

import shutil
from pathlib import Path

from neo4j import GraphDatabase

from rag_evaluator.config import settings


def clean_filesystem():
    """Remove local data directories."""
    paths_to_remove = [
        Path(settings.chroma_persist_directory),
        Path("data/prepared/filesystem_rag"),
    ]

    for path in paths_to_remove:
        if path.exists():
            print(f"Removing {path}...")
            shutil.rmtree(path)
            print(f"  Deleted {path}")
        else:
            print(f"  {path} does not exist, skipping.")


def clean_neo4j():
    """Clear Neo4j database."""
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
        )
        with driver.session() as session:
            print("  Running: MATCH (n) DETACH DELETE n")
            session.run("MATCH (n) DETACH DELETE n")
            print("  Neo4j database cleared.")
        driver.close()
    except Exception as e:
        print(f"  Error clearing Neo4j: {e}")
        print("  Please ensure Neo4j is running.")


def main():
    print("=== RAG Evaluator Clean-up Tool ===\n")
    
    print("1. Cleaning filesystem artifacts (ChromaDB, Filesystem RAG)...")
    clean_filesystem()
    print("\n2. Cleaning Neo4j database...")
    clean_neo4j()
    
    print("\nNote: Qdrant collection is automatically cleared when running 'prepare'.")
    print("\n=== Clean-up Complete ===")


if __name__ == "__main__":
    main()
