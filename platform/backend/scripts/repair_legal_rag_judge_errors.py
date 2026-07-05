"""Repair or rejudge Legal RAG Bench judge results for an existing evaluation.

This script does not rerun RAG generation. It reuses stored questions, expected
answers, generated answers, and retrieved-context artifacts to rerun the Legal
RAG binary judge.

Usage:
    uv run python scripts/repair_legal_rag_judge_errors.py EVALUATION_ID --dry-run
    uv run python scripts/repair_legal_rag_judge_errors.py EVALUATION_ID
    uv run python scripts/repair_legal_rag_judge_errors.py EVALUATION_ID --all
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db_context
from app.models.evaluation import Evaluation
from app.models.evaluation_result import EvaluationResult
from app.models.test_case import TestCase
from app.services.artifact_store import ArtifactStore, get_artifact_store
from app.services.legal_rag_bench_judge import LegalRAGBenchJudge
from app.services.legal_rag_bench_metrics import (
    derive_success_signals,
    derive_taxonomy,
    summarize_legal_rag_metrics,
)
from app.services.provider_resolver import resolve_provider_endpoint


def _needs_repair(raw_metrics: dict[str, Any] | None) -> bool:
    legal = (raw_metrics or {}).get("legal_rag_bench")
    if not isinstance(legal, dict):
        return False

    judge = legal.get("judge")
    if isinstance(judge, dict) and judge.get("parse_error"):
        return True
    return legal.get("taxonomy") is None and isinstance(judge, dict)


def _needs_rejudge(raw_metrics: dict[str, Any] | None, *, rejudge_all: bool) -> bool:
    legal = (raw_metrics or {}).get("legal_rag_bench")
    if not isinstance(legal, dict):
        return False
    if rejudge_all:
        return isinstance(legal.get("judge"), dict)
    return _needs_repair(raw_metrics)


async def _load_raw_metrics(
    store: ArtifactStore,
    db: Any,
    result: EvaluationResult,
) -> dict[str, Any] | None:
    if not result.raw_metrics_artifact_id:
        return None
    raw_metrics = await store.retrieve_json_by_id(db, result.raw_metrics_artifact_id)
    return raw_metrics if isinstance(raw_metrics, dict) else None


async def _load_retrieved_context(
    store: ArtifactStore,
    db: Any,
    result: EvaluationResult,
) -> list[str]:
    if not result.retrieved_context_artifact_id:
        return []
    context = await store.retrieve_json_by_id(db, result.retrieved_context_artifact_id)
    if isinstance(context, list):
        return [str(item) for item in context]
    return []


async def _collect_legal_summary(
    store: ArtifactStore,
    db: Any,
    evaluation_id: uuid.UUID,
) -> dict[str, Any] | None:
    result = await db.execute(
        select(EvaluationResult).where(EvaluationResult.evaluation_id == evaluation_id)
    )
    legal_results: list[dict[str, Any]] = []
    for row in result.scalars().all():
        raw_metrics = await _load_raw_metrics(store, db, row)
        legal = (raw_metrics or {}).get("legal_rag_bench")
        if isinstance(legal, dict):
            legal_results.append(legal)
    return summarize_legal_rag_metrics(legal_results)


async def repair_evaluation(evaluation_id: uuid.UUID, dry_run: bool) -> int:
    return await _repair_evaluation(
        evaluation_id=evaluation_id,
        dry_run=dry_run,
        judge_model_override=None,
        judge_provider_override=None,
        timeout_seconds=None,
        judge_context_max_chars=None,
        judge_context_chunk_max_chars=None,
        source_qa_ids=None,
        rejudge_all=False,
    )


async def _repair_evaluation(
    *,
    evaluation_id: uuid.UUID,
    dry_run: bool,
    judge_model_override: str | None,
    judge_provider_override: str | None,
    timeout_seconds: float | None,
    judge_context_max_chars: int | None,
    judge_context_chunk_max_chars: int | None,
    source_qa_ids: set[int] | None,
    rejudge_all: bool,
) -> int:
    async with get_db_context() as db:
        store = get_artifact_store()
        evaluation_result = await db.execute(
            select(Evaluation)
            .where(Evaluation.id == evaluation_id)
            .options(selectinload(Evaluation.index))
        )
        evaluation = evaluation_result.scalar_one_or_none()
        if not evaluation:
            raise SystemExit(f"Evaluation not found: {evaluation_id}")

        config_snapshot = (
            evaluation.index.config_snapshot
            if evaluation.index and isinstance(evaluation.index.config_snapshot, dict)
            else {}
        )
        generation_provider = config_snapshot.get("llm_provider")
        generation_base_url = config_snapshot.get("llm_base_url")
        judge_provider = (
            judge_provider_override or evaluation.eval_judge_provider or generation_provider
        )
        judge_model = judge_model_override or evaluation.eval_judge_model or config_snapshot.get(
            "llm_model"
        )
        if not judge_model:
            raise SystemExit("Evaluation has no judge model and no index LLM model fallback.")

        base_override = generation_base_url if judge_provider == generation_provider else None
        endpoint = resolve_provider_endpoint(judge_provider, base_override)
        judge_kwargs: dict[str, int] = {}
        if judge_context_max_chars is not None:
            judge_kwargs["context_max_chars"] = judge_context_max_chars
        if judge_context_chunk_max_chars is not None:
            judge_kwargs["context_chunk_max_chars"] = judge_context_chunk_max_chars
        judge = LegalRAGBenchJudge(**judge_kwargs)

        rows_result = await db.execute(
            select(EvaluationResult)
            .where(EvaluationResult.evaluation_id == evaluation_id)
            .options(selectinload(EvaluationResult.test_case))
            .order_by(EvaluationResult.created_at.asc())
        )
        rows = rows_result.scalars().all()

        repair_rows: list[tuple[EvaluationResult, TestCase, dict[str, Any]]] = []
        for row in rows:
            raw_metrics = await _load_raw_metrics(store, db, row)
            if not _needs_rejudge(raw_metrics, rejudge_all=rejudge_all):
                continue
            if not row.test_case:
                print(f"Skipping {row.id}: missing test case")
                continue
            if not _matches_source_qa_filter(row.test_case, source_qa_ids):
                continue
            repair_rows.append((row, row.test_case, raw_metrics or {}))

        mode = "full Legal RAG rejudge" if rejudge_all else "Legal RAG judge repair"
        print(
            f"Evaluation {evaluation_id}: {len(repair_rows)} rows selected for {mode}."
        )
        if dry_run or not repair_rows:
            return len(repair_rows)

        for index, (row, test_case, raw_metrics) in enumerate(repair_rows, start=1):
            print(f"[{index}/{len(repair_rows)}] Rejudging result {row.id}")
            retrieved_context = await _load_retrieved_context(store, db, row)
            try:
                judge_result = await judge.judge(
                    question=test_case.question,
                    reference_answer=test_case.expected_answer,
                    generated_answer=row.generated_answer or "",
                    retrieved_context=retrieved_context,
                    model=str(judge_model),
                    provider=str(judge_provider) if judge_provider else None,
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                print(f"Judge failed for result {row.id}: {exc}")
                judge_result = {
                    "correct": None,
                    "grounded": None,
                    "reasoning": str(exc),
                    "parse_error": "judge_exception",
                    "model": str(judge_model),
                    "provider": str(judge_provider) if judge_provider else None,
                    "attempts": 0,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "cost_usd": 0.0,
                }

            legal = raw_metrics.get("legal_rag_bench")
            if not isinstance(legal, dict):
                legal = {}
            retrieval = (
                legal.get("retrieval") if isinstance(legal.get("retrieval"), dict) else None
            )
            legal["judge"] = judge_result
            legal["taxonomy"] = derive_taxonomy(
                retrieval_metrics=retrieval,
                judge_result=judge_result,
            )
            old_signals = (
                legal.get("success_signals")
                if isinstance(legal.get("success_signals"), dict)
                else {}
            )
            legal["success_signals"] = derive_success_signals(
                retrieval_metrics=retrieval,
                judge_result=judge_result,
                taxonomy=legal["taxonomy"],
                g_eval_score=old_signals.get("g_eval_score"),
                g_eval_threshold=settings.EVAL_G_EVAL_THRESHOLD,
            )
            raw_metrics["legal_rag_bench"] = legal

            artifact = await store.store_json(db, raw_metrics, ArtifactStore.KIND_RAW_METRICS)
            row.raw_metrics_artifact_id = artifact.id

        legal_summary = await _collect_legal_summary(store, db, evaluation_id)
        summary_metrics = dict(evaluation.summary_metrics or {})
        if legal_summary:
            summary_metrics["legal_rag_bench"] = legal_summary
        evaluation.summary_metrics = summary_metrics
        await db.commit()

        print("Repair committed.")
        return len(repair_rows)


def _matches_source_qa_filter(test_case: TestCase, source_qa_ids: set[int] | None) -> bool:
    if not source_qa_ids:
        return True
    raw_source_qa_id = (test_case.metadata_ or {}).get("source_qa_id")
    try:
        source_qa_id = int(raw_source_qa_id)
    except (TypeError, ValueError):
        return False
    return source_qa_id in source_qa_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_id", type=uuid.UUID)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many rows would be repaired.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the evaluation's stored judge model for the repair.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Override the evaluation's stored judge provider for the repair.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override per-call judge timeout.",
    )
    parser.add_argument(
        "--judge-context-max-chars",
        type=int,
        default=None,
        help="Override total retrieved-context characters visible to the judge.",
    )
    parser.add_argument(
        "--judge-context-chunk-max-chars",
        type=int,
        default=None,
        help="Override per-context-chunk characters visible to the judge.",
    )
    parser.add_argument(
        "--source-qa-id",
        type=int,
        action="append",
        default=None,
        help="Only rejudge rows whose test case metadata.source_qa_id matches. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rejudge every Legal RAG row, not only judge errors/missing taxonomy.",
    )
    args = parser.parse_args()
    asyncio.run(
        _repair_evaluation(
            evaluation_id=args.evaluation_id,
            dry_run=args.dry_run,
            judge_model_override=args.judge_model,
            judge_provider_override=args.judge_provider,
            timeout_seconds=args.timeout_seconds,
            judge_context_max_chars=args.judge_context_max_chars,
            judge_context_chunk_max_chars=args.judge_context_chunk_max_chars,
            source_qa_ids=set(args.source_qa_id) if args.source_qa_id else None,
            rejudge_all=args.all,
        )
    )


if __name__ == "__main__":
    main()
