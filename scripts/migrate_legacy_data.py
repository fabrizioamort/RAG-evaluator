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
from app.models import Project, TestSet, TestCase, Evaluation, EvaluationResult, KnowledgeBase, Document, KnowledgeBaseVersion, RAGConfig
from sqlalchemy import select
from app.api.deps import get_db

async def migrate_knowledge_base(session, project_id: uuid.UUID):
    kb_name = "Legacy Knowledge Base"
    
    # Check if exists
    query = select(KnowledgeBase).where(KnowledgeBase.project_id == project_id, KnowledgeBase.name == kb_name)
    result = await session.execute(query)
    kb = result.scalar_one_or_none()

    raw_data_path = Path("data/raw")
    storage_path = Path("platform/backend/storage")
    storage_path.mkdir(parents=True, exist_ok=True)

    if not kb:
        kb = KnowledgeBase(
            id=uuid.uuid4(),
            project_id=project_id,
            name=kb_name,
            description="Knowledge base imported from data/raw PDFs.",
            status="ready",
            current_version=1,
            storage_path=str(storage_path)
        )
        session.add(kb)
        await session.flush()
        print(f"Created KnowledgeBase: {kb_name}")
    else:
        print(f"KnowledgeBase {kb_name} already exists.")

    # Process files
    pdf_files = list(raw_data_path.glob("*.pdf"))
    docs_to_add = []
    
    for pdf_file in pdf_files:
        # Check if doc exists
        # Simplified check: just by filename in this KB
        doc_query = select(Document).where(Document.knowledge_base_id == kb.id, Document.filename == pdf_file.name)
        doc_res = await session.execute(doc_query)
        if doc_res.scalar_one_or_none():
            continue

        # Copy file
        dest_path = storage_path / pdf_file.name
        if not dest_path.exists():
            import shutil
            shutil.copy2(pdf_file, dest_path)
        
        file_stats = dest_path.stat()
        
        doc = Document(
            id=uuid.uuid4(),
            knowledge_base_id=kb.id,
            filename=pdf_file.name,
            file_path=str(dest_path),
            content_type="application/pdf",
            size_bytes=file_stats.st_size,
            status="indexed"
        )
        session.add(doc)
        docs_to_add.append(doc)
    
    await session.flush()
    if docs_to_add:
        print(f"Imported {len(docs_to_add)} documents.")
        
        # Create Version 1 if it doesn't exist
        v_query = select(KnowledgeBaseVersion).where(KnowledgeBaseVersion.knowledge_base_id == kb.id, KnowledgeBaseVersion.version_number == 1)
        v_res = await session.execute(v_query)
        if not v_res.scalar_one_or_none():
             # Create snapshot
             snapshot = [
                 {"id": str(d.id), "filename": d.filename, "size": d.size_bytes} 
                 for d in docs_to_add
             ]
             kb_version = KnowledgeBaseVersion(
                 id=uuid.uuid4(),
                 knowledge_base_id=kb.id,
                 version_number=1,
                 change_type="initial_import",
                 change_description="Imported from legacy data directory",
                 document_snapshot=snapshot
             )
             session.add(kb_version)
             print("Created KnowledgeBaseVersion 1")

    return kb.id

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
        try:
            parts = report_file.stem.split("_")
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            completed_at = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
        except:
            completed_at = datetime.now()

        pass_rate = data.get("pass_rate", 0.0)
        if pass_rate > 1.0:
            pass_rate = pass_rate / 100.0

        eval_run = Evaluation(
            id=uuid.uuid4(),
            project_id=project_id,
            test_set_id=test_set_id,
            status="completed",
            started_at=completed_at,
            completed_at=completed_at,
            summary_metrics=data.get("metrics", {}),
            performance_metrics=data.get("performance_metrics", {}),
            pass_rate=pass_rate,
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
                generated_answer=res_data.get("answer", ""),
                faithfulness_score=res_metrics.get("faithfulness"),
                relevancy_score=res_metrics.get("answer_relevancy"),
                precision_score=res_metrics.get("contextual_precision"),
                recall_score=res_metrics.get("contextual_recall"),
                latency_seconds=res_data.get("retrieval_time", 0.0)
            )
            session.add(result_entry)

        print(f"Imported report: {report_file.name}")

async def migrate_rag_configs(session, project_id: uuid.UUID):
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return

    json_files = list(reports_dir.glob("*.json"))
    report_files = [f for f in json_files if f.name.startswith("eval_") and not "comparison" in f.name]
    
    created_configs = 0
    
    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue
        
        impl_name = data.get("rag_implementation", "Unknown Implementation")
        
        # Check if config exists
        query = select(RAGConfig).where(RAGConfig.project_id == project_id, RAGConfig.name == impl_name)
        result = await session.execute(query)
        if result.scalar_one_or_none():
            continue
            
        # Infer type
        rag_type = "custom"
        if "chroma" in impl_name.lower():
            rag_type = "vector_semantic"
        elif "hybrid" in impl_name.lower():
            rag_type = "vector_hybrid"
        elif "graph" in impl_name.lower() or "neo4j" in impl_name.lower():
            rag_type = "graph_rag"
        elif "files" in impl_name.lower():
             rag_type = "filesystem_agent"

        # Create Config
        config = RAGConfig(
            id=uuid.uuid4(),
            project_id=project_id,
            name=impl_name,
            rag_type=rag_type,
            parameters={"imported": True, "source": report_file.name},
            llm_provider="openai",
            llm_model="gpt-4o"
        )
        session.add(config)
        created_configs += 1
    
    if created_configs > 0:
        print(f"Created {created_configs} RAG Configs based on reports.")

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

        # 2. Migrate KBs
        kb_id = await migrate_knowledge_base(session, project.id)
        await session.commit()

        # 3. Migrate Test Sets
        test_set_id = await migrate_test_sets(session, project.id)
        if test_set_id:
            await session.commit()

            # 4. Migrate RAG Configs
            await migrate_rag_configs(session, project.id)
            await session.commit()

            # 5. Migrate Reports
            await migrate_reports(session, project.id, test_set_id)
            await session.commit()

    print("\nMigration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
