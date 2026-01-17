"""Migrate existing evaluations to use indexes."""

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Add platform/backend to sys.path
sys.path.append(str(Path(__file__).parent.parent / "platform" / "backend"))

from app.database import get_db_context
from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.rag_config import RAGConfig


def generate_physical_id() -> str:
    """Generate a unique physical ID for an index."""
    return f"idx_{uuid.uuid4().hex[:24]}"


def get_storage_type(rag_type: str) -> str:
    """Map RAG type to storage type."""
    mapping = {
        "vector_semantic": "chroma",
        "vector_hybrid": "qdrant",
        "graph_rag": "neo4j",
        "filesystem_rag": "filesystem",
    }
    return mapping.get(rag_type, "unknown")


async def create_legacy_index(
    session, eval: Evaluation, kb: KnowledgeBase, rag_config: RAGConfig
) -> KnowledgeBaseIndex:
    """Create a legacy index for an evaluation."""

    physical_id = generate_physical_id()
    storage_type = get_storage_type(rag_config.rag_type)

    config_snapshot = {
        "rag_type": rag_config.rag_type,
        "parameters": rag_config.parameters,
        "llm_provider": rag_config.llm_provider,
        "llm_model": rag_config.llm_model,
        "llm_base_url": rag_config.llm_base_url,
    }

    # Check if there are documents in the KB to set document_count
    # We can't easily know the document count at the time of evaluation if it changed,
    # but we can use the current count or 0.
    # Since we don't have the docs loaded here, we'll skip or do a separate query if needed.
    # For now, default to 0 or try to use current KB doc count if available.

    index = KnowledgeBaseIndex(
        knowledge_base_id=kb.id,
        kb_version_id=eval.kb_version_id,  # Use version from eval if available
        rag_config_id=rag_config.id,
        name=f"Legacy Index - {rag_config.name} (Migrated)",
        description=f"Auto-generated for Evaluation {eval.id}. Physical storage may not exist.",
        status="ready",  # Mark as ready so UI displays it, even if storage is missing
        physical_id=physical_id,
        storage_type=storage_type,
        config_snapshot=config_snapshot,
        document_count=0,  # Unknown
        created_at=eval.started_at or datetime.now(UTC),
        build_completed_at=eval.started_at or datetime.now(UTC),
    )

    session.add(index)
    await session.flush()  # Get ID
    await session.refresh(index)

    return index


async def migrate():
    print("Starting migration of existing evaluations...")

    async with get_db_context() as session:
        # Find evaluations without index
        query = (
            select(Evaluation)
            .where(Evaluation.knowledge_base_index_id.is_(None))
            .where(Evaluation.knowledge_base_id.isnot(None))
            .where(Evaluation.rag_config_id.isnot(None))
            .options(selectinload(Evaluation.knowledge_base), selectinload(Evaluation.rag_config))
        )

        result = await session.execute(query)
        evaluations = result.scalars().all()

        print(f"Found {len(evaluations)} evaluations to migrate.")

        migrated_count = 0
        created_indices = 0

        for eval in evaluations:
            try:
                # Check if compatible index exists
                # We look for an index with same KB, RAG config
                idx_query = (
                    select(KnowledgeBaseIndex)
                    .where(KnowledgeBaseIndex.knowledge_base_id == eval.knowledge_base_id)
                    .where(KnowledgeBaseIndex.rag_config_id == eval.rag_config_id)
                    .where(KnowledgeBaseIndex.status == "ready")
                    .order_by(KnowledgeBaseIndex.created_at.desc())
                )

                idx_result = await session.execute(idx_query)
                existing_index = idx_result.scalars().first()

                if existing_index:
                    print(f"Linking Eval {eval.id} to existing Index {existing_index.id}")
                    eval.knowledge_base_index_id = existing_index.id
                else:
                    print(f"Creating legacy index for Eval {eval.id}")
                    if not eval.knowledge_base or not eval.rag_config:
                        print(f"Skipping Eval {eval.id}: Missing KB or Config relation")
                        continue

                    new_index = await create_legacy_index(
                        session, eval, eval.knowledge_base, eval.rag_config
                    )
                    eval.knowledge_base_index_id = new_index.id
                    created_indices += 1

                migrated_count += 1

            except Exception as e:
                print(f"Error migrating Eval {eval.id}: {e}")

        await session.commit()
        print(
            f"Migration complete. Migrated {migrated_count} evaluations. Created {created_indices} new legacy indexes."
        )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(migrate())
