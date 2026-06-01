"""Service for managing evaluation job checkpoints and state."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.models.evaluation_job import EvaluationJob
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobCheckpointService:
    """Service for managing evaluation job lifecycle and persistence."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the checkpoint service.

        Args:
            db_session: Database session for persistence.
        """
        self.db = db_session

    async def get_job(self, evaluation_id: uuid.UUID) -> Optional[EvaluationJob]:
        """Retrieve the job record for an evaluation.

        Args:
            evaluation_id: ID of the evaluation.

        Returns:
            The EvaluationJob model or None.
        """
        result = await self.db.execute(
            select(EvaluationJob).where(EvaluationJob.evaluation_id == evaluation_id)
        )
        return result.scalars().first()

    async def create_job(self, evaluation_id: uuid.UUID, total_test_cases: int) -> EvaluationJob:
        """Create a new job record for an evaluation.

        Args:
            evaluation_id: ID of the evaluation.
            total_test_cases: Total number of test cases to process.

        Returns:
            The created EvaluationJob model.
        """
        job = EvaluationJob(
            evaluation_id=evaluation_id,
            state="created",
            progress_current=0,
            progress_total=total_test_cases,
            last_checkpoint=0,
            checkpoint_data={},
            last_heartbeat=datetime.now(timezone.utc),
        )
        self.db.add(job)
        await self.db.flush()  # generate the UUID before commit
        await self.db.commit()
        return job

    async def update_progress(
        self,
        evaluation_id: uuid.UUID,
        current_index: int,
        state: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job progress and heartbeat.

        Args:
            evaluation_id: ID of the evaluation.
            current_index: Number of test cases completed.
            state: Optional new state for the job.
            error_message: Optional error message.
        """
        values = {
            "progress_current": current_index,
            "last_heartbeat": datetime.now(timezone.utc),
        }
        if state:
            values["state"] = state
        if error_message:
            values["error_message"] = error_message

        await self.db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.evaluation_id == evaluation_id)
            .values(**values)
        )
        await self.db.commit()

    async def save_checkpoint(
        self, evaluation_id: uuid.UUID, current_index: int, checkpoint_data: dict[str, Any]
    ) -> None:
        """Save a checkpoint for resumption.

        Args:
            evaluation_id: ID of the evaluation.
            current_index: Index of the last processed test case.
            checkpoint_data: Data needed to resume from this point.
        """
        await self.db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.evaluation_id == evaluation_id)
            .values(
                last_checkpoint=current_index,
                checkpoint_data=checkpoint_data,
                last_heartbeat=datetime.now(timezone.utc),
            )
        )
        await self.db.commit()

    async def update_evaluation_status(self, evaluation_id: uuid.UUID, status: str) -> None:
        """Update the status of the evaluation itself.

        Args:
            evaluation_id: ID of the evaluation.
            status: New status (e.g., 'running', 'completed', 'failed').
        """
        await self.db.execute(
            update(Evaluation).where(Evaluation.id == evaluation_id).values(status=status)
        )
        await self.db.commit()

    async def complete_job(
        self,
        evaluation_id: uuid.UUID,
        summary_metrics: dict[str, Any],
        pass_rate: float,
        cost_metrics: dict[str, Any] | None = None,
        performance_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Mark a job and evaluation as completed.

        Args:
            evaluation_id: ID of the evaluation.
            summary_metrics: Calculated summary metrics.
            pass_rate: Calculated pass rate.
            cost_metrics: Optional cost metrics.
            performance_metrics: Optional performance metrics.
        """
        now = datetime.now(timezone.utc)

        # Update Evaluation
        values = {
            "status": "completed",
            "completed_at": now,
            "summary_metrics": summary_metrics,
            "pass_rate": pass_rate,
        }
        if cost_metrics:
            values["cost_metrics"] = cost_metrics
        if performance_metrics:
            values["performance_metrics"] = performance_metrics

        await self.db.execute(
            update(Evaluation).where(Evaluation.id == evaluation_id).values(**values)
        )

        # Update Job
        await self.db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.evaluation_id == evaluation_id)
            .values(
                state="completed",
                last_heartbeat=now,
            )
        )
        await self.db.commit()

    async def fail_job(self, evaluation_id: uuid.UUID, error_message: str) -> None:
        """Mark a job and evaluation as failed.

        Args:
            evaluation_id: ID of the evaluation.
            error_message: Error that caused the failure.
        """
        now = datetime.now(timezone.utc)

        # Update Evaluation
        await self.db.execute(
            update(Evaluation)
            .where(Evaluation.id == evaluation_id)
            .values(
                status="failed",
                completed_at=now,
                error_message=error_message,
            )
        )

        # Update Job
        await self.db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.evaluation_id == evaluation_id)
            .values(
                state="failed",
                error_message=error_message,
                last_heartbeat=now,
            )
        )
        await self.db.commit()


def get_checkpoint_service(db_session: AsyncSession) -> JobCheckpointService:
    """Factory to get the checkpoint service."""
    return JobCheckpointService(db_session)
