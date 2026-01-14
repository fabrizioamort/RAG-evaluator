import asyncio
import json
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
from dotenv import load_dotenv

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

from app.database import async_session_maker
from app.models import Evaluation, EvaluationResult, TestCase


async def main():
    async with async_session_maker() as session:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            print("Reports directory not found.")
            return

        json_files = list(reports_dir.glob("*.json"))
        # Filter out comparison files if they exist
        report_files = [
            f for f in json_files if f.name.startswith("eval_") and "comparison" not in f.name
        ]

        total_updates = 0
        for report_file in report_files:
            print(f"Processing {report_file.name}...")
            with open(report_file, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading {report_file}: {e}")
                    continue

            # Find matching evaluation
            query = select(Evaluation).where(
                Evaluation.notes == f"Imported from {report_file.name}"
            )
            result = await session.execute(query)
            evaluation = result.scalar_one_or_none()

            if not evaluation:
                print(f"Evaluation for {report_file.name} not found in DB.")
                continue

            detailed_results = data.get("detailed_results", [])
            updates_count = 0

            for res_data in detailed_results:
                question = res_data.get("question")
                answer = res_data.get("answer", "")

                # Find matching test case
                tc_query = select(TestCase).where(
                    TestCase.test_set_id == evaluation.test_set_id, TestCase.question == question
                )
                tc_result = await session.execute(tc_query)
                test_case = tc_result.scalar_one_or_none()

                if not test_case:
                    print(f"Warning: Test case not found for question: {question[:30]}...")
                    continue

                # Find the EvaluationResult
                er_query = select(EvaluationResult).where(
                    EvaluationResult.evaluation_id == evaluation.id,
                    EvaluationResult.test_case_id == test_case.id,
                )
                er_result = await session.execute(er_query)
                eval_result = er_result.scalar_one_or_none()

                if eval_result:
                    # Update if empty or different
                    # (We care mostly about empty, but might as well correct if different)
                    if not eval_result.generated_answer or eval_result.generated_answer == "":
                        eval_result.generated_answer = answer
                        session.add(eval_result)
                        updates_count += 1
                else:
                    print(f"Warning: EvaluationResult not found for TC {test_case.id}")

            print(f"Updated {updates_count} answers for {report_file.name}")
            total_updates += updates_count

        await session.commit()
        print(f"Done fixing answers. Total updates: {total_updates}")


if __name__ == "__main__":
    asyncio.run(main())
