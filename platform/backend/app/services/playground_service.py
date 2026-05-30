"""Playground service for executing RAG queries and managing query history."""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_base_index import KnowledgeBaseIndex
from app.models.playground_query import PlaygroundQuery
from app.schemas.playground import (
    PlaygroundIndexInfo,
    PlaygroundQueryDetail,
    PlaygroundQueryHistoryItem,
    PlaygroundQueryResponse,
    PlaygroundQueryResult,
    QueryMetrics,
    RetrievalTraceResponse,
    RetrievalTraceStepResponse,
    RetrievedChunkResponse,
    RetrievedContextResponse,
)
from app.services.cost_tracker import get_cost_tracker
from app.services.rag_adapter import get_rag_adapter_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PlaygroundService:
    """Service for executing playground queries and managing history."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the playground service.

        Args:
            db: Database session for persistence.
        """
        self.db = db
        self.rag_adapter = get_rag_adapter_service()
        self.cost_tracker = get_cost_tracker()

    async def get_available_indexes(
        self,
        project_id: uuid.UUID | None = None,
        kb_id: uuid.UUID | None = None,
    ) -> list[PlaygroundIndexInfo]:
        """Get list of indexes available for playground queries.

        Only returns indexes with status='ready'.

        Args:
            project_id: Optional filter by project.
            kb_id: Optional filter by knowledge base.

        Returns:
            List of available indexes with their metadata.
        """
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.status == "ready")
            .options(
                selectinload(KnowledgeBaseIndex.knowledge_base).selectinload(
                    KnowledgeBase.project
                ),
                selectinload(KnowledgeBaseIndex.rag_config),
            )
        )

        if kb_id:
            query = query.where(KnowledgeBaseIndex.knowledge_base_id == kb_id)

        result = await self.db.execute(query)
        indexes = result.scalars().all()

        # Filter by project if specified (need to check via KB relationship)
        if project_id:
            indexes = [idx for idx in indexes if idx.knowledge_base.project_id == project_id]

        return [
            PlaygroundIndexInfo(
                id=idx.id,
                name=idx.name,
                rag_type=idx.config_snapshot.get("rag_type", "unknown"),
                knowledge_base_id=idx.knowledge_base_id,
                knowledge_base_name=idx.knowledge_base.name,
                project_id=idx.knowledge_base.project_id,
                project_name=idx.knowledge_base.project.name
                if idx.knowledge_base.project
                else "Unknown",
                document_count=idx.document_count,
                chunk_count=idx.chunk_count,
                status=idx.status,
            )
            for idx in indexes
        ]

    async def execute_query(
        self,
        question: str,
        index_ids: list[uuid.UUID],
        top_k: int = 5,
        query_overrides: dict[str, Any] | None = None,
    ) -> PlaygroundQueryResponse:
        """Execute a query against multiple indexes in parallel.

        Args:
            question: The question to ask.
            index_ids: List of index IDs to query.
            top_k: Number of chunks to retrieve.

        Returns:
            PlaygroundQueryResponse with results from all indexes.
        """
        query_id = uuid.uuid4()
        start_time = time.time()

        # Load all indexes
        query = (
            select(KnowledgeBaseIndex)
            .where(KnowledgeBaseIndex.id.in_(index_ids))
            .options(
                selectinload(KnowledgeBaseIndex.knowledge_base).selectinload(
                    KnowledgeBase.project
                ),
                selectinload(KnowledgeBaseIndex.rag_config),
            )
        )
        result = await self.db.execute(query)
        indexes = {idx.id: idx for idx in result.scalars().all()}

        # Validate all indexes exist and are ready
        for idx_id in index_ids:
            if idx_id not in indexes:
                raise ValueError(f"Index {idx_id} not found")
            if indexes[idx_id].status != "ready":
                raise ValueError(f"Index {indexes[idx_id].name} is not ready (status: {indexes[idx_id].status})")

        effective_overrides = dict(query_overrides or {})
        effective_overrides.setdefault("top_k", top_k)

        # Validate overrides before launching parallel queries.
        for idx_id in index_ids:
            self.rag_adapter.build_effective_config(indexes[idx_id], effective_overrides)

        # Execute queries in parallel
        tasks = [
            self._execute_single_query(indexes[idx_id], question, effective_overrides)
            for idx_id in index_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        query_results: list[PlaygroundQueryResult] = []
        for idx_id, result in zip(index_ids, results):
            index = indexes[idx_id]
            if isinstance(result, Exception):
                logger.error(
                    "Query failed for index",
                    index_id=str(idx_id),
                    error=str(result),
                )
                query_results.append(
                    PlaygroundQueryResult(
                        index_id=idx_id,
                        index_name=index.name,
                        rag_type=index.config_snapshot.get("rag_type", "unknown"),
                        knowledge_base_name=index.knowledge_base.name,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                query_results.append(result)

        total_time = (time.time() - start_time) * 1000  # ms

        # Save to history
        await self._save_query_history(
            query_id=query_id,
            question=question,
            index_ids=index_ids,
            top_k=top_k,
            query_overrides=effective_overrides,
            results=query_results,
            total_time_ms=total_time,
        )

        return PlaygroundQueryResponse(
            query_id=query_id,
            question=question,
            results=query_results,
            created_at=datetime.now(timezone.utc),
        )

    async def _execute_single_query(
        self,
        index: KnowledgeBaseIndex,
        question: str,
        query_overrides: dict[str, Any],
    ) -> PlaygroundQueryResult:
        """Execute a query against a single index.

        Args:
            index: The index to query.
            question: The question to ask.
            top_k: Number of chunks to retrieve.

        Returns:
            PlaygroundQueryResult with answer and trace.
        """
        # Load RAG instance without rebuilding the ready index.
        rag, effective = self.rag_adapter.load_rag_for_index_query(index, query_overrides)

        try:
            # Execute query with trace
            result = await self.rag_adapter.query_with_trace(rag, question, effective.top_k)

            # Extract components from result
            # query_with_trace returns: answer, context, metadata, retrieval_trace
            answer = result.get("answer", "")
            metadata = result.get("metadata", {})
            retrieval_trace = result.get("retrieval_trace", {})  # This is a dict
            context_chunks = result.get("context", [])

            # Extract timing from metadata
            retrieval_time_sec = metadata.get("retrieval_time", 0)
            generation_time_sec = metadata.get("generation_time", 0)

            # Extract token usage from metadata
            token_usage = metadata.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            # Build context response from retrieval_trace (chunks are there)
            context_response = None
            retrieved_chunks = retrieval_trace.get("retrieved_chunks", [])
            if retrieved_chunks or context_chunks:
                chunk_details = []
                for chunk in retrieved_chunks:
                    chunk_details.append(
                        RetrievedChunkResponse(
                            content=chunk.get("content", ""),
                            document_id=chunk.get("document_id", ""),
                            chunk_id=chunk.get("chunk_id", ""),
                            score=chunk.get("score", 0.0),
                            rank=chunk.get("rank", 0),
                            source=chunk.get("source", ""),
                            metadata=chunk.get("metadata", {}),
                        )
                    )
                context_response = RetrievedContextResponse(
                    chunks=context_chunks,
                    chunk_details=chunk_details,
                )

            # Build trace response (retrieval_trace is a dict)
            trace_response = None
            if retrieval_trace:
                steps = []
                for step in retrieval_trace.get("steps", []):
                    steps.append(
                        RetrievalTraceStepResponse(
                            step_type=step.get("type", "unknown"),
                            duration_ms=step.get("duration_ms", 0),
                            input_data=step.get("input"),
                            output_summary=step.get("output_summary"),
                            metadata=step.get("metadata", {}),
                        )
                    )
                trace_response = RetrievalTraceResponse(
                    strategy=retrieval_trace.get("strategy", "unknown"),
                    steps=steps,
                    total_duration_ms=retrieval_trace.get("total_duration_ms", 0),
                    fusion_details=retrieval_trace.get("fusion_details"),
                )

            # Calculate metrics (convert seconds to ms)
            retrieval_time = retrieval_time_sec * 1000
            generation_time = generation_time_sec * 1000

            # Calculate cost
            cost_usd = None
            if prompt_tokens or completion_tokens:
                cost_usd = self.cost_tracker.calculate_cost(
                    effective.generation_model, prompt_tokens, completion_tokens
                )

            metrics = QueryMetrics(
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost_usd,
            )

            return PlaygroundQueryResult(
                index_id=index.id,
                index_name=index.name,
                rag_type=index.config_snapshot.get("rag_type", "unknown"),
                knowledge_base_name=index.knowledge_base.name,
                answer=answer,
                retrieved_context=context_response,
                trace=trace_response,
                metrics=metrics,
                effective_config_snapshot=effective.effective_config_snapshot,
                success=True,
            )

        except Exception as e:
            logger.exception(
                "Error executing query",
                index_id=str(index.id),
                error=str(e),
            )
            return PlaygroundQueryResult(
                index_id=index.id,
                index_name=index.name,
                rag_type=index.config_snapshot.get("rag_type", "unknown"),
                knowledge_base_name=index.knowledge_base.name,
                success=False,
                error=str(e),
            )
        finally:
            # Clean up RAG instance
            try:
                rag.close()
            except Exception:
                pass

    async def _save_query_history(
        self,
        query_id: uuid.UUID,
        question: str,
        index_ids: list[uuid.UUID],
        top_k: int,
        query_overrides: dict[str, Any],
        results: list[PlaygroundQueryResult],
        total_time_ms: float,
    ) -> None:
        """Save a query to the history.

        Args:
            query_id: Unique ID for this query.
            question: The question asked.
            index_ids: List of index IDs queried.
            top_k: Top K parameter used.
            query_overrides: Query-time overrides used.
            results: Results from each index.
            total_time_ms: Total execution time.
        """
        success_count = sum(1 for r in results if r.success)

        # Convert results to dict for JSON storage
        results_dict = [r.model_dump(mode="json") for r in results]

        query = PlaygroundQuery(
            id=query_id,
            question=question,
            top_k=top_k,
            index_ids=[str(idx) for idx in index_ids],  # Store as strings for JSON
            results=results_dict,
            index_count=len(index_ids),
            success_count=success_count,
            total_time_ms=total_time_ms,
            extra_data={"query_overrides": query_overrides},
        )

        self.db.add(query)
        await self.db.commit()

    async def get_query_history(
        self,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[PlaygroundQueryHistoryItem], int]:
        """Get paginated query history.

        Args:
            offset: Number of items to skip.
            limit: Maximum items to return.

        Returns:
            Tuple of (history items, total count).
        """
        # Get total count
        count_query = select(func.count(PlaygroundQuery.id))
        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        # Get items
        query = (
            select(PlaygroundQuery)
            .order_by(PlaygroundQuery.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(query)
        queries = result.scalars().all()

        items = []
        for q in queries:
            # Extract index names from results
            index_names = [r.get("index_name", "Unknown") for r in q.results]

            items.append(
                PlaygroundQueryHistoryItem(
                    id=q.id,
                    created_at=q.created_at,
                    question=q.question,
                    index_count=q.index_count,
                    index_names=index_names,
                    success_count=q.success_count,
                    total_time_ms=q.total_time_ms,
                )
            )

        return items, total

    async def get_query_detail(self, query_id: uuid.UUID) -> PlaygroundQueryDetail | None:
        """Get full details of a saved query.

        Args:
            query_id: The query ID to retrieve.

        Returns:
            Query detail or None if not found.
        """
        query = select(PlaygroundQuery).where(PlaygroundQuery.id == query_id)
        result = await self.db.execute(query)
        q = result.scalar_one_or_none()

        if not q:
            return None

        # Convert stored results back to schema objects
        results = [PlaygroundQueryResult.model_validate(r) for r in q.results]

        return PlaygroundQueryDetail(
            id=q.id,
            created_at=q.created_at,
            question=q.question,
            top_k=q.top_k,
            query_overrides=(q.extra_data or {}).get("query_overrides", {}),
            results=results,
        )

    async def delete_query(self, query_id: uuid.UUID) -> bool:
        """Delete a query from history.

        Args:
            query_id: The query ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        query = select(PlaygroundQuery).where(PlaygroundQuery.id == query_id)
        result = await self.db.execute(query)
        q = result.scalar_one_or_none()

        if not q:
            return False

        await self.db.delete(q)
        await self.db.commit()
        return True


def get_playground_service(db: AsyncSession) -> PlaygroundService:
    """Factory function to create a PlaygroundService.

    Args:
        db: Database session.

    Returns:
        PlaygroundService instance.
    """
    return PlaygroundService(db)
