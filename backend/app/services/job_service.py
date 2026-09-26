from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode, JobStatus, JobType
from app.core.errors import AppError
from app.db.models import JobRecord
from app.schemas.jobs import JobStatusResponse

logger = logging.getLogger(__name__)

# Valid state transitions. A queued job must never jump directly to COMPLETED.
_VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED},
}


class JobService:
    """Job lifecycle management with explicit state transitions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_job(
        self,
        case_id: str,
        job_type: JobType,
        *,
        inference_source: str = "mock",
        model_version: str | None = None,
    ) -> JobRecord:
        """Persist a new QUEUED job.

        `inference_source` and `model_version` are recorded from the backend
        that is actually registered in this process, so a job row never claims
        PyTorch inference for a synthetic result or vice versa.
        """
        now = datetime.now(timezone.utc)
        job = JobRecord(
            job_id=str(uuid4()),
            case_id=case_id,
            job_type=job_type.value,
            status=JobStatus.QUEUED.value,
            created_at=now,
            progress=0,
            inference_source=inference_source,
            model_version=model_version,
        )
        self.session.add(job)
        self.session.flush()
        logger.info(
            "job_created",
            extra={
                "job_id": job.job_id,
                "case_id": case_id,
                "job_type": job_type.value,
                "inference_source": inference_source,
            },
        )
        return job

    def get_job(self, job_id: str) -> JobRecord:
        """Look up a job by UUID. Raises JOB_NOT_FOUND on miss."""
        job = self.session.get(JobRecord, job_id)
        if job is None:
            raise AppError(
                ErrorCode.JOB_NOT_FOUND,
                "No job exists for the given identifier.",
                status_code=404,
            )
        return job

    def transition(
        self,
        job_id: str,
        from_status: JobStatus,
        to_status: JobStatus,
        *,
        progress: int | None = None,
    ) -> JobRecord:
        """Explicitly transition a job between valid states."""
        job = self.get_job(job_id)
        current = JobStatus(job.status)

        if current != from_status:
            raise AppError(
                ErrorCode.INVALID_JOB_STATE,
                f"Job is {current.value}, expected {from_status.value}.",
                status_code=409,
            )

        valid_targets = _VALID_TRANSITIONS.get(current, set())
        if to_status not in valid_targets:
            raise AppError(
                ErrorCode.INVALID_JOB_STATE,
                f"Transition {current.value} → {to_status.value} is not allowed.",
                status_code=409,
            )

        now = datetime.now(timezone.utc)
        job.status = to_status.value
        if to_status == JobStatus.RUNNING:
            job.started_at = now
        if to_status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = now
        if progress is not None:
            job.progress = progress
        self.session.flush()
        return job

    def complete_job(self, job_id: str) -> JobRecord:
        """Mark a job as COMPLETED with progress 100."""
        job = self.transition(
            job_id, JobStatus.RUNNING, JobStatus.COMPLETED, progress=100
        )
        logger.info(
            "job_completed",
            extra={
                "job_id": job.job_id,
                "case_id": job.case_id,
                "job_type": job.job_type,
                "inference_source": job.inference_source,
            },
        )
        return job

    def fail_job(
        self, job_id: str, error_code: str, error_message: str
    ) -> JobRecord:
        """Mark a job as FAILED with a safe user-facing error message."""
        job = self.get_job(job_id)
        current = JobStatus(job.status)

        # Allow failure from QUEUED or RUNNING
        if current not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise AppError(
                ErrorCode.INVALID_JOB_STATE,
                f"Cannot fail a job in state {current.value}.",
                status_code=409,
            )

        now = datetime.now(timezone.utc)
        job.status = JobStatus.FAILED.value
        job.completed_at = now
        if current == JobStatus.QUEUED:
            job.started_at = now
        job.error_code = error_code
        job.error_message = error_message
        self.session.flush()

        logger.warning(
            "job_failed",
            extra={
                "job_id": job.job_id,
                "case_id": job.case_id,
                "job_type": job.job_type,
                "inference_source": job.inference_source,
                "error": error_code,
            },
        )
        return job

    def update_progress(self, job_id: str, progress: int) -> None:
        """Update the progress percentage of a running job."""
        job = self.get_job(job_id)
        job.progress = min(max(progress, 0), 100)
        self.session.flush()

    def to_response(self, job: JobRecord) -> JobStatusResponse:
        """Convert a JobRecord to an API response."""
        result_id: str | None = None
        if job.result is not None:
            result_id = job.result.result_id

        return JobStatusResponse(
            job_id=job.job_id,
            case_id=job.case_id,
            job_type=job.job_type,
            status=job.status,
            progress=job.progress,
            inference_source=job.inference_source,
            model_version=job.model_version,
            result_id=result_id,
            error_code=job.error_code,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
