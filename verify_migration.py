import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import func, select

# Ensure we are in the project root
root_path = Path(__file__).parent
sys.path.append(str(root_path / "platform" / "backend"))

os.chdir(str(root_path))
from dotenv import load_dotenv

load_dotenv(root_path / "platform" / "backend" / ".env")

# Fix DB URL if needed
db_url = os.getenv("DATABASE_URL", "")
if "sqlite" in db_url and "///./" in db_url:
    abs_db_path = root_path / "platform" / "backend" / "storage" / "dev.db"
    fixed_db_url = f"sqlite+aiosqlite:///{abs_db_path}"
    os.environ["DATABASE_URL"] = fixed_db_url

from app.database import async_session_maker
from app.models import Document, KnowledgeBase, RAGConfig


async def main():
    async with async_session_maker() as session:
        # Check KBs
        kb_query = select(KnowledgeBase)
        kb_res = await session.execute(kb_query)
        kbs = kb_res.scalars().all()
        print(f"Knowledge Bases: {len(kbs)}")
        for kb in kbs:
            print(f" - {kb.name} (ID: {kb.id})")

            # Check Docs
            doc_query = select(func.count(Document.id)).where(Document.knowledge_base_id == kb.id)
            doc_count = (await session.execute(doc_query)).scalar()
            print(f"   Documents: {doc_count}")

        # Check RAG Configs
        rc_query = select(RAGConfig)
        rc_res = await session.execute(rc_query)
        configs = rc_res.scalars().all()
        print(f"RAG Configs: {len(configs)}")
        for rc in configs:
            print(f" - {rc.name} ({rc.rag_type})")


if __name__ == "__main__":
    asyncio.run(main())
