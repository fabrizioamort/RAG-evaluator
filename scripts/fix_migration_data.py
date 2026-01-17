import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import select

# Ensure we are in the project root
root_path = Path(__file__).parent.parent
if not (root_path / "platform" / "backend").exists():
    print("Please run this script from the project root.")
    sys.exit(1)

# Set working directory to root for data path resolution
os.chdir(str(root_path))

# Load environment variables for the backend BEFORE importing app modules
from dotenv import load_dotenv  # noqa: E402

env_path = root_path / "platform" / "backend" / ".env"
load_dotenv(env_path)

# If DATABASE_URL is relative to platform/backend/storage, fix it
db_url = os.getenv("DATABASE_URL", "")
if "sqlite" in db_url and "///./" in db_url:
    abs_db_path = root_path / "platform" / "backend" / "storage" / "dev.db"
    fixed_db_url = f"sqlite+aiosqlite:///{abs_db_path}"
    os.environ["DATABASE_URL"] = fixed_db_url
    print(f"Fixed DB URL to absolute path: {fixed_db_url}")

# NOW add platform/backend to sys.path and import app modules
backend_path = root_path / "platform" / "backend"
sys.path.append(str(backend_path))

from app.database import async_session_maker  # noqa: E402
from app.models import Evaluation  # noqa: E402


async def main():
    async with async_session_maker() as session:
        # Find all evaluations with pass_rate > 1.0
        query = select(Evaluation).where(Evaluation.pass_rate > 1.0)
        result = await session.execute(query)
        evaluations = result.scalars().all()

        if not evaluations:
            print("No evaluations found with pass_rate > 1.0. Database is clean.")
            return

        print(f"Found {len(evaluations)} evaluations with invalid pass_rate.")

        count = 0
        for evaluation in evaluations:
            old_rate = evaluation.pass_rate
            new_rate = old_rate / 100.0
            evaluation.pass_rate = new_rate
            print(f"Fixing Evaluation {evaluation.id}: {old_rate} -> {new_rate}")
            count += 1

        await session.commit()
        print(f"Successfully fixed {count} evaluations.")


if __name__ == "__main__":
    asyncio.run(main())
