import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import select

# Ensure we are in the project root
root_path = Path(__file__).parent
sys.path.append(str(root_path / "platform" / "backend"))

os.chdir(str(root_path))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(root_path / "platform" / "backend" / ".env")

# Fix DB URL if needed (same as other scripts)
db_url = os.getenv("DATABASE_URL", "")
if "sqlite" in db_url and "///./" in db_url:
    abs_db_path = root_path / "platform" / "backend" / "storage" / "dev.db"
    fixed_db_url = f"sqlite+aiosqlite:///{abs_db_path}"
    os.environ["DATABASE_URL"] = fixed_db_url

from app.database import async_session_maker  # noqa: E402
from app.models import Evaluation  # noqa: E402


async def main():
    async with async_session_maker() as session:
        result = await session.execute(select(Evaluation.id).limit(1))
        eval_id = result.scalar()
        print(eval_id)


if __name__ == "__main__":
    asyncio.run(main())
