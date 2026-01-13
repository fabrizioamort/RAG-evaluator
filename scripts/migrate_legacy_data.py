import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from app.models import Project, TestSet, TestCase, Evaluation, EvaluationResult
from sqlalchemy import select

async def migrate_test_sets(session, project_id: uuid.UUID):
    test_set_path = Path("data/test_set.json")
    if not test_set_path.exists():
        print(f"Test set file {test_set_path} not found.")
        return None

    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_set_name = "Legacy CLI Test Set"
    # Check if already exists
    query = select(TestSet).where(TestSet.project_id == project_id, TestSet.name == test_set_name)
    result = await session.execute(query)
    test_set = result.scalar_one_or_none()

    if not test_set:
        test_case_count = len(data.get("test_cases", []))
        test_set = TestSet(
            id=uuid.uuid4(),
            project_id=project_id,
            name=test_set_name,
            description="Imported from data/test_set.json",
            tags=["legacy", "cli"]
        )
        session.add(test_set)
        await session.flush()
        print(f"Created TestSet: {test_set_name}")

        test_cases = data.get("test_cases", [])
        for tc_data in test_cases:
            test_case = TestCase(
                id=uuid.uuid4(),
                test_set_id=test_set.id,
                question=tc_data.get("question", ""),
                expected_answer=tc_data.get("expected_answer", ""),
                ground_truth_context=tc_data.get("ground_truth_context", []),
                difficulty=tc_data.get("difficulty", "medium"),
                category=tc_data.get("category", "general"),
                is_generated=False,
                is_reviewed=True
            )
            session.add(test_case)
        print(f"Imported {len(test_cases)} test cases.")
    else:
        print(f"TestSet {test_set_name} already exists, skipping.")

    return test_set.id

async def migrate_reports(session, project_id: uuid.UUID, test_set_id: uuid.UUID):
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("Reports directory not found.")
        return

    json_files = list(reports_dir.glob("*.json"))
    # Filter out comparison files if they exist
    report_files = [f for f in json_files if f.name.startswith("eval_") and not "comparison" in f.name]

    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {report_file}: {e}")
                continue

        impl_name = data.get("rag_implementation", "Unknown Implementation")
        
        # Check if evaluation already exists
        query = select(Evaluation).where(
            Evaluation.project_id == project_id,
            Evaluation.notes == f"Imported from {report_file.name}"
        )
        result = await session.execute(query)
        if result.scalar_one_or_none():
            print(f"Evaluation for {report_file.name} already exists, skipping.")
            continue

        # Extract timestamp from filename or data
        # Filename: eval_impl_YYYY-MM-DD_HHMMSS.json
        try:
            parts = report_file.stem.split("_")
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            completed_at = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
        except:
            completed_at = datetime.now()

        eval_run = Evaluation(
            id=uuid.uuid4(),
            project_id=project_id,
            test_set_id=test_set_id,
            status="completed",
            started_at=completed_at,
            completed_at=completed_at,
            summary_metrics=data.get("metrics", {}),
            performance_metrics=data.get("performance_metrics", {}),
            pass_rate=data.get("pass_rate", 0.0),
            notes=f"Imported from {report_file.name}",
            tags=["legacy", "cli", impl_name.lower().replace(" ", "-")]
        )
        session.add(eval_run)
        await session.flush()

        detailed_results = data.get("detailed_results", [])
        for res_data in detailed_results:
            # Try to find matching test case by question
            tc_query = select(TestCase).where(
                TestCase.test_set_id == test_set_id,
                TestCase.question == res_data.get("question")
            )
            tc_result = await session.execute(tc_query)
            test_case = tc_result.scalar_one_or_none()

            res_metrics = res_data.get("metrics", {})
            result_entry = EvaluationResult(
                id=uuid.uuid4(),
                evaluation_id=eval_run.id,
                test_case_id=test_case.id if test_case else None,
                generated_answer=res_data.get("actual_answer", ""),
                faithfulness_score=res_metrics.get("faithfulness"),
                relevancy_score=res_metrics.get("answer_relevancy"),
                precision_score=res_metrics.get("contextual_precision"),
                recall_score=res_metrics.get("contextual_recall"),
                latency_seconds=res_data.get("retrieval_time", 0.0)
            )
            session.add(result_entry)

        print(f"Imported report: {report_file.name}")

async def main():
    async with async_session_maker() as session:
        # 1. Ensure Legacy Project exists
        project_name = "Legacy CLI Migration"
        query = select(Project).where(Project.name == project_name)
        result = await session.execute(query)
        project = result.scalar_one_or_none()

        if not project:
            project = Project(
                id=uuid.uuid4(),
                name=project_name,
                description="Project containing data imported from the CLI version.",
                status="active",
                tags=["migration", "legacy"]
            )
            session.add(project)
            await session.commit()
            print(f"Created Project: {project_name}")
        else:
            print(f"Project {project_name} already exists.")

        # 2. Migrate Test Sets
        test_set_id = await migrate_test_sets(session, project.id)
        if test_set_id:
            await session.commit()

            # 3. Migrate Reports
            await migrate_reports(session, project.id, test_set_id)
            await session.commit()

    print("\nMigration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
