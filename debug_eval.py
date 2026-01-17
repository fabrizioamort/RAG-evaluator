import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import select

root_path = Path("platform/backend").resolve()
sys.path.append(str(root_path))
os.chdir(str(Path(".").resolve()))

from dotenv import load_dotenv  # noqa: E402

env_path = root_path / ".env"
load_dotenv(env_path)

db_url = os.getenv("DATABASE_URL", "")
if "sqlite" in db_url and "///./" in db_url:
    abs_db_path = root_path / "storage" / "dev.db"
    fixed_db_url = f"sqlite+aiosqlite:///{abs_db_path}"
    os.environ["DATABASE_URL"] = fixed_db_url
    print(f"Fixed DB URL: {fixed_db_url}")

from app.database import async_session_maker  # noqa: E402
from app.models import Evaluation, EvaluationResult  # noqa: E402


async def main():
    async with async_session_maker() as session:
        # Check specific ID
        import uuid

        eval_id = uuid.UUID("0a4daad2-4663-447b-a5a6-c0b5576a3930")
        result = await session.execute(select(Evaluation).where(Evaluation.id == eval_id))
        eval_obj = result.scalar_one_or_none()

        if eval_obj:
            print(f"Status: '{eval_obj.status}'")

            # Check results
            query = (
                select(EvaluationResult).where(EvaluationResult.evaluation_id == eval_id).limit(1)
            )
            r_result = await session.execute(query)
            res = r_result.scalar_one_or_none()
            if res:
                print(f"Sample Result ID: {res.id}")
                print(f"Test Case ID: {res.test_case_id}")
                print(f"Generated Answer: '{res.generated_answer}'")

                if res.test_case_id:
                    from app.models import TestCase

                    tc_res = await session.execute(
                        select(TestCase).where(TestCase.id == res.test_case_id)
                    )
                    tc = tc_res.scalar_one_or_none()
                    if tc:
                        print(f"Linked Question: '{tc.question}'")
                    else:
                        print("Linked Test Case NOT FOUND in DB")
            else:
                print("No results found")
        else:
            print("Evaluation not found")


if __name__ == "__main__":
    asyncio.run(main())
