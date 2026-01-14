"""Test generation service for creating test cases from knowledge base content.

This service generates test cases by:
1. Loading document content from a knowledge base
2. Chunking documents into processable segments
3. Using LLM to generate question/answer pairs from chunks
4. Validating generated test cases through quality gates
5. Storing provenance information for traceability
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document
from app.models.test_case import TestCase
from app.models.test_generation_job import TestGenerationJob
from app.models.test_template import TestTemplate
from app.services.artifact_store import ArtifactStore, get_artifact_store
from app.services.llm_provider import LLMProviderService, TokenUsage
from app.services.test_quality_gate import (
    TestQualityGateService,
    get_quality_gate_service,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# Generation prompts
SYSTEM_PROMPT_GENERATE = """You are an expert at creating high-quality test questions for evaluating RAG (Retrieval Augmented Generation) systems.

Your task is to generate question-answer pairs from the provided text context. The questions should:
1. Be answerable ONLY from the provided context
2. Test understanding, not just keyword matching
3. Have clear, unambiguous answers
4. Vary in complexity and question type

For each question, provide:
- A clear, well-formed question
- The expected answer (derived from the context)
- The difficulty level (easy, medium, hard)
- The question type (factual, inferential, comparative, multi_hop)

Respond in valid JSON format only."""

USER_PROMPT_TEMPLATE = """Generate {count} high-quality test question(s) from the following text context.

CONTEXT:
{context}

{template_instruction}

Respond with a JSON array of objects, each containing:
{{
  "question": "The question text",
  "expected_answer": "The expected answer",
  "difficulty": "easy|medium|hard",
  "question_type": "factual|inferential|comparative|multi_hop"
}}

Generate exactly {count} question(s). Each question must be answerable from the context above."""

TEMPLATE_INSTRUCTION_MAP = {
    "factual": "Focus on factual questions that test direct recall of information.",
    "inferential": "Focus on questions that require drawing conclusions or making inferences.",
    "comparative": "Focus on questions that compare or contrast different concepts.",
    "multi_hop": "Focus on questions that require connecting multiple pieces of information.",
}


@dataclass
class GenerationConfig:
    """Configuration for test case generation."""

    # Number of test cases to generate
    target_count: int = 20

    # Questions per chunk
    questions_per_chunk: int = 2

    # Chunk size for document splitting (characters)
    chunk_size: int = 2000

    # Chunk overlap (characters)
    chunk_overlap: int = 200

    # Difficulty distribution (should sum to 1.0)
    difficulty_distribution: dict[str, float] | None = None

    # Template IDs to use (empty for all)
    template_ids: list[uuid.UUID] | None = None

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Quality gate settings
    skip_semantic_check: bool = False

    def __post_init__(self) -> None:
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}


@dataclass
class GenerationProgress:
    """Progress update for test generation."""

    status: str
    current: int
    total: int
    generated: int
    rejected: int
    message: str


@dataclass
class GeneratedTestCase:
    """A generated test case before saving."""

    question: str
    expected_answer: str
    ground_truth_context: list[str]
    difficulty: str
    question_type: str
    template_id: uuid.UUID | None
    quality_score: float
    provenance: dict[str, Any]


class TestGeneratorService:
    """Service for generating test cases from knowledge base content."""

    def __init__(
        self,
        db: AsyncSession,
        llm_service: LLMProviderService | None = None,
        artifact_store: ArtifactStore | None = None,
        quality_gate: TestQualityGateService | None = None,
    ) -> None:
        """Initialize test generator service.

        Args:
            db: Database session.
            llm_service: LLM service for generation.
            artifact_store: Artifact store for provenance.
            quality_gate: Quality gate service.
        """
        self.db = db
        self._llm_service = llm_service
        self._artifact_store = artifact_store
        self._quality_gate = quality_gate
        self._cancelled = False
        self._total_usage = TokenUsage()

    @property
    def llm_service(self) -> LLMProviderService:
        """Lazy-load LLM service."""
        if self._llm_service is None:
            self._llm_service = LLMProviderService()
        return self._llm_service

    @property
    def artifact_store(self) -> ArtifactStore:
        """Lazy-load artifact store."""
        if self._artifact_store is None:
            self._artifact_store = get_artifact_store()
        return self._artifact_store

    @property
    def quality_gate(self) -> TestQualityGateService:
        """Lazy-load quality gate service."""
        if self._quality_gate is None:
            self._quality_gate = get_quality_gate_service(llm_service=self.llm_service)
        return self._quality_gate

    def cancel(self) -> None:
        """Cancel the current generation."""
        self._cancelled = True

    async def generate(
        self,
        job_id: uuid.UUID,
        config: GenerationConfig,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
    ) -> list[GeneratedTestCase]:
        """Generate test cases for a test generation job.

        Args:
            job_id: Test generation job ID.
            config: Generation configuration.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of generated test cases.
        """
        self._cancelled = False
        self._total_usage = TokenUsage()

        # Load job with relationships
        job = await self._load_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        await self.db.commit()

        try:
            # Load existing questions from test set to avoid duplicates
            existing_questions = await self._get_existing_questions(job.test_set_id)
            await self.quality_gate.add_existing_questions(existing_questions)

            # Load document content
            documents = await self._load_documents(job.knowledge_base_id)
            if not documents:
                raise ValueError("No documents found in knowledge base")

            # Chunk documents
            chunks = self._chunk_documents(documents, config)
            job.questions_total = min(config.target_count, len(chunks) * config.questions_per_chunk)
            await self.db.commit()

            logger.info(
                "Starting test generation",
                job_id=str(job_id),
                document_count=len(documents),
                chunk_count=len(chunks),
                target_count=config.target_count,
            )

            # Generate test cases from chunks
            generated_cases: list[GeneratedTestCase] = []
            rejected_count = 0
            chunk_index = 0

            while len(generated_cases) < config.target_count and chunk_index < len(chunks):
                if self._cancelled:
                    logger.info("Generation cancelled", job_id=str(job_id))
                    break

                chunk = chunks[chunk_index]
                chunk_index += 1

                # Determine how many questions to generate from this chunk
                remaining = config.target_count - len(generated_cases)
                questions_to_generate = min(config.questions_per_chunk, remaining)

                # Report progress
                if progress_callback:
                    progress_callback(
                        GenerationProgress(
                            status="running",
                            current=chunk_index,
                            total=len(chunks),
                            generated=len(generated_cases),
                            rejected=rejected_count,
                            message=f"Processing chunk {chunk_index}/{len(chunks)}",
                        )
                    )

                # Generate questions from this chunk
                try:
                    new_cases, new_rejected = await self._generate_from_chunk(
                        chunk=chunk,
                        count=questions_to_generate,
                        config=config,
                        kb_id=job.knowledge_base_id,
                    )
                    generated_cases.extend(new_cases)
                    rejected_count += new_rejected

                    # Update job progress
                    job.questions_generated = len(generated_cases)
                    job.questions_rejected = rejected_count
                    await self.db.commit()

                except Exception as e:
                    logger.warning(
                        "Failed to generate from chunk",
                        chunk_index=chunk_index,
                        error=str(e),
                    )
                    continue

            # Update job status
            if self._cancelled:
                job.status = "cancelled"
            else:
                job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.questions_generated = len(generated_cases)
            job.questions_rejected = rejected_count
            await self.db.commit()

            # Final progress report
            if progress_callback:
                progress_callback(
                    GenerationProgress(
                        status=job.status,
                        current=len(chunks),
                        total=len(chunks),
                        generated=len(generated_cases),
                        rejected=rejected_count,
                        message=f"Generation complete: {len(generated_cases)} test cases",
                    )
                )

            logger.info(
                "Test generation completed",
                job_id=str(job_id),
                generated=len(generated_cases),
                rejected=rejected_count,
            )

            return generated_cases

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await self.db.commit()
            logger.error("Test generation failed", job_id=str(job_id), error=str(e))
            raise

    async def generate_and_save(
        self,
        job_id: uuid.UUID,
        config: GenerationConfig,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
    ) -> list[TestCase]:
        """Generate and save test cases.

        Args:
            job_id: Test generation job ID.
            config: Generation configuration.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of saved TestCase model instances.
        """
        # Generate test cases
        generated = await self.generate(job_id, config, progress_callback)

        # Load job to get test_set_id
        job = await self._load_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Save test cases
        saved_cases: list[TestCase] = []
        for gen_case in generated:
            test_case = await self._save_test_case(job.test_set_id, gen_case)
            saved_cases.append(test_case)

        await self.db.commit()

        logger.info(
            "Saved generated test cases",
            job_id=str(job_id),
            count=len(saved_cases),
        )

        return saved_cases

    async def _load_job(self, job_id: uuid.UUID) -> TestGenerationJob | None:
        """Load a test generation job with relationships."""
        query = (
            select(TestGenerationJob)
            .options(
                selectinload(TestGenerationJob.test_set),
                selectinload(TestGenerationJob.knowledge_base),
            )
            .where(TestGenerationJob.id == job_id)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_existing_questions(self, test_set_id: uuid.UUID) -> list[str]:
        """Get existing questions in a test set."""
        query = select(TestCase.question).where(TestCase.test_set_id == test_set_id)
        result = await self.db.execute(query)
        return [row[0] for row in result.fetchall()]

    async def _load_documents(self, kb_id: uuid.UUID | None) -> list[tuple[str, str]]:
        """Load document content from a knowledge base.

        Returns:
            List of (filename, content) tuples.
        """
        if kb_id is None:
            return []

        query = (
            select(Document)
            .where(Document.knowledge_base_id == kb_id)
            .where(Document.status == "processed")
        )
        result = await self.db.execute(query)
        documents = result.scalars().all()

        loaded: list[tuple[str, str]] = []
        for doc in documents:
            try:
                file_path = Path(doc.file_path)
                if file_path.exists():
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                    loaded.append((doc.filename, content))
                else:
                    logger.warning(
                        "Document file not found",
                        doc_id=str(doc.id),
                        file_path=str(file_path),
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load document",
                    doc_id=str(doc.id),
                    error=str(e),
                )
                continue

        return loaded

    def _chunk_documents(
        self,
        documents: list[tuple[str, str]],
        config: GenerationConfig,
    ) -> list[dict[str, Any]]:
        """Chunk documents into processable segments.

        Args:
            documents: List of (filename, content) tuples.
            config: Generation configuration.

        Returns:
            List of chunks with metadata.
        """
        chunks = []

        for filename, content in documents:
            # Simple character-based chunking with overlap
            start = 0
            chunk_index = 0

            while start < len(content):
                end = start + config.chunk_size
                chunk_text = content[start:end]

                # Try to end at a sentence boundary
                if end < len(content):
                    # Look for sentence endings
                    for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                        last_sep = chunk_text.rfind(sep)
                        if last_sep > config.chunk_size // 2:
                            chunk_text = chunk_text[: last_sep + len(sep)]
                            end = start + len(chunk_text)
                            break

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "source_file": filename,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    }
                )

                chunk_index += 1
                start = end - config.chunk_overlap

        return chunks

    async def _generate_from_chunk(
        self,
        chunk: dict[str, Any],
        count: int,
        config: GenerationConfig,
        kb_id: uuid.UUID | None,
    ) -> tuple[list[GeneratedTestCase], int]:
        """Generate test cases from a single chunk.

        Args:
            chunk: Chunk with text and metadata.
            count: Number of questions to generate.
            config: Generation configuration.
            kb_id: Knowledge base ID.

        Returns:
            Tuple of (generated cases, rejected count).
        """
        # Build template instruction
        template_instruction = ""
        if config.template_ids:
            # Load templates
            query = select(TestTemplate).where(TestTemplate.id.in_(config.template_ids))
            result = await self.db.execute(query)
            templates = result.scalars().all()
            if templates:
                template_hints = []
                for t in templates:
                    template_hints.append(f"- {t.name}: {t.question_template}")
                template_instruction = "Use these question templates as inspiration:\n" + "\n".join(
                    template_hints
                )

        # Build the prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            count=count,
            context=chunk["text"],
            template_instruction=template_instruction,
        )

        # Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.llm_service.completion(
            model=config.llm_model,
            messages=messages,
            provider=config.llm_provider,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Track token usage
        self._total_usage.prompt_tokens += response.usage.prompt_tokens
        self._total_usage.completion_tokens += response.usage.completion_tokens
        self._total_usage.total_tokens += response.usage.total_tokens

        # Parse response
        try:
            raw_cases = self._parse_llm_response(response.content)
        except Exception as e:
            logger.warning("Failed to parse LLM response", error=str(e))
            return [], 0

        # Validate each generated case through quality gates
        generated: list[GeneratedTestCase] = []
        rejected = 0

        for raw_case in raw_cases:
            question = raw_case.get("question", "").strip()
            expected_answer = raw_case.get("expected_answer", "").strip()
            difficulty = raw_case.get("difficulty", "medium").lower()
            question_type = raw_case.get("question_type", "factual").lower()

            if not question or not expected_answer:
                rejected += 1
                continue

            # Validate through quality gate
            context = [chunk["text"]]
            validation = await self.quality_gate.validate(
                question=question,
                expected_answer=expected_answer,
                context=context,
                skip_semantic_check=config.skip_semantic_check,
            )

            if not validation.passed:
                logger.debug(
                    "Test case rejected by quality gate",
                    reason=validation.rejection_reason,
                    question=question[:50] + "...",
                )
                rejected += 1
                continue

            # Build provenance information
            provenance = {
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "generation_model": config.llm_model,
                "generation_provider": config.llm_provider,
                "generated_at": datetime.utcnow().isoformat(),
                "kb_id": str(kb_id) if kb_id else None,
                "validation_details": validation.details,
            }

            generated.append(
                GeneratedTestCase(
                    question=question,
                    expected_answer=expected_answer,
                    ground_truth_context=context,
                    difficulty=difficulty,
                    question_type=question_type,
                    template_id=None,
                    quality_score=validation.score,
                    provenance=provenance,
                )
            )

        return generated, rejected

    def _parse_llm_response(self, content: str) -> list[dict[str, Any]]:
        """Parse the LLM response into a list of test case dictionaries.

        Args:
            content: Raw LLM response content.

        Returns:
            List of parsed test case dictionaries.
        """
        # Try to find JSON array in the response
        content = content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        # Try to find array brackets
        if not content.startswith("["):
            start = content.find("[")
            if start != -1:
                end = content.rfind("]") + 1
                content = content[start:end]

        # Parse JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            return []
        except json.JSONDecodeError:
            # Try to extract individual objects
            logger.warning("Failed to parse as JSON, attempting fallback extraction")
            return []

    async def _save_test_case(
        self, test_set_id: uuid.UUID, gen_case: GeneratedTestCase
    ) -> TestCase:
        """Save a generated test case to the database.

        Args:
            test_set_id: Test set ID.
            gen_case: Generated test case data.

        Returns:
            Saved TestCase model instance.
        """
        # Store provenance as artifact
        provenance_artifact = await self.artifact_store.store_json(
            db=self.db,
            data=gen_case.provenance,
            kind=ArtifactStore.KIND_PROVENANCE,
        )

        # Create test case
        test_case = TestCase(
            test_set_id=test_set_id,
            template_id=gen_case.template_id,
            question=gen_case.question,
            expected_answer=gen_case.expected_answer,
            ground_truth_context=gen_case.ground_truth_context,
            difficulty=gen_case.difficulty,
            category=gen_case.question_type,
            question_type=gen_case.question_type,
            is_generated=True,
            is_reviewed=False,
            quality_score=gen_case.quality_score,
            provenance_artifact_id=provenance_artifact.id,
        )

        self.db.add(test_case)
        await self.db.flush()
        await self.db.refresh(test_case)

        return test_case

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage for the generation session."""
        return self._total_usage


async def generate_stream(
    db: AsyncSession,
    job_id: uuid.UUID,
    config: GenerationConfig,
) -> AsyncGenerator[GenerationProgress, None]:
    """Stream test case generation progress.

    Args:
        db: Database session.
        job_id: Test generation job ID.
        config: Generation configuration.

    Yields:
        GenerationProgress updates.
    """
    progress_queue: asyncio.Queue[GenerationProgress] = asyncio.Queue()

    def progress_callback(progress: GenerationProgress) -> None:
        progress_queue.put_nowait(progress)

    # Start generation in a task
    service = TestGeneratorService(db)
    gen_task = asyncio.create_task(service.generate_and_save(job_id, config, progress_callback))

    # Yield progress updates
    while not gen_task.done():
        try:
            progress = await asyncio.wait_for(
                progress_queue.get(),
                timeout=1.0,
            )
            yield progress
        except asyncio.TimeoutError:
            continue

    # Wait for task to complete and handle any exceptions
    try:
        await gen_task
    except Exception as e:
        yield GenerationProgress(
            status="failed",
            current=0,
            total=0,
            generated=0,
            rejected=0,
            message=str(e),
        )


# Factory function
def get_test_generator_service(
    db: AsyncSession,
    llm_service: LLMProviderService | None = None,
    artifact_store: ArtifactStore | None = None,
    quality_gate: TestQualityGateService | None = None,
) -> TestGeneratorService:
    """Create a test generator service instance.

    Args:
        db: Database session.
        llm_service: Optional LLM service.
        artifact_store: Optional artifact store.
        quality_gate: Optional quality gate service.

    Returns:
        TestGeneratorService instance.
    """
    return TestGeneratorService(
        db=db,
        llm_service=llm_service,
        artifact_store=artifact_store,
        quality_gate=quality_gate,
    )
