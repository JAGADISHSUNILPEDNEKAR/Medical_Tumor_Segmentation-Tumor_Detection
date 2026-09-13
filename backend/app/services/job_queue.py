"""Async FIFO job queue with a single worker.

This is a process-local, in-memory queue suitable for the MVP.
It prevents parallel inference execution (which would OOM on a single GPU)
by processing jobs sequentially through a single worker coroutine.

LIMITATIONS (documented, acceptable for Phase 3):
- The queue is process-local. If Docker later runs two backend replicas,
  jobs are NOT globally coordinated between them.
- Jobs in the asyncio.Queue are lost if the process crashes. The persistent
  JobRecord remains in the database as QUEUED/RUNNING and can be detected
  on restart (recovery is not implemented in Phase 3).
- No retry logic. A failed job stays FAILED and is not re-enqueued.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.constants import ErrorCode, JobStatus, JobType
from app.db.session import SessionLocal
from app.inference.base import InferenceService
from app.services.job_service import JobService
from app.services.result_service import ResultService
from app.storage.filesystem import CaseStorage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class JobQueue:
    """Application-level async FIFO queue with a single inference worker.

    Usage:
        queue = JobQueue(inference_service, upload_root)
        await queue.start()    # in FastAPI lifespan startup
        queue.enqueue(job_id, case_id, job_type, has_ground_truth)
        await queue.stop()     # in FastAPI lifespan shutdown
    """

    def __init__(
        self,
        inference_service: InferenceService,
        upload_root: Path,
    ) -> None:
        self._inference_service = inference_service
        self._upload_root = upload_root
        self._queue: asyncio.Queue[_JobItem] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the single worker coroutine."""
        self._worker_task = asyncio.create_task(self._worker(), name="job_queue_worker")
        logger.info("job_queue_started")

    async def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        if self._worker_task is not None:
            # Enqueue a sentinel to unblock the worker
            await self._queue.put(_STOP_SENTINEL)
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("job_queue_shutdown_timeout")
                self._worker_task.cancel()
            self._worker_task = None
        logger.info("job_queue_stopped")

    def enqueue(
        self,
        job_id: str,
        case_id: str,
        job_type: JobType,
        *,
        has_ground_truth: bool = False,
    ) -> None:
        """Add a job to the FIFO queue. Non-blocking."""
        item = _JobItem(
            job_id=job_id,
            case_id=case_id,
            job_type=job_type,
            has_ground_truth=has_ground_truth,
        )
        self._queue.put_nowait(item)
        logger.info(
            "job_enqueued",
            extra={
                "job_id": job_id,
                "case_id": case_id,
                "job_type": job_type.value,
                "queue_size": self._queue.qsize(),
            },
        )

    async def _worker(self) -> None:
        """Single worker that processes jobs sequentially."""
        logger.info("job_worker_started")
        while True:
            item = await self._queue.get()
            if item is _STOP_SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._execute_job(item)
            except Exception:
                # Should not happen — _execute_job catches everything.
                # But don't let the worker die.
                logger.exception(
                    "job_worker_unexpected_error",
                    extra={"job_id": item.job_id},
                )
            finally:
                self._queue.task_done()
        logger.info("job_worker_stopped")

    async def _execute_job(self, item: _JobItem) -> None:
        """Execute a single job: transition → infer → persist result."""
        # Run the blocking inference in a thread so we don't block the event loop
        await asyncio.get_event_loop().run_in_executor(
            None, self._execute_job_sync, item
        )

    def _execute_job_sync(self, item: _JobItem) -> None:
        """Synchronous job execution with its own DB session."""
        session = SessionLocal()
        try:
            job_svc = JobService(session)
            result_svc = ResultService(session)
            storage = CaseStorage(self._upload_root)

            # Transition: QUEUED → RUNNING
            try:
                job_svc.transition(
                    item.job_id, JobStatus.QUEUED, JobStatus.RUNNING, progress=10
                )
                session.commit()
            except Exception:
                logger.exception(
                    "job_transition_failed",
                    extra={"job_id": item.job_id, "transition": "QUEUED→RUNNING"},
                )
                session.rollback()
                # Try to fail the job
                try:
                    job_svc.fail_job(
                        item.job_id,
                        ErrorCode.INVALID_JOB_STATE,
                        "Failed to start job execution.",
                    )
                    session.commit()
                except Exception:
                    session.rollback()
                return

            # Execute inference
            try:
                paths = storage.paths_for(item.case_id)
                job_svc.update_progress(item.job_id, 30)
                session.commit()

                inference_result = self._inference_service.predict(
                    case_dir=paths.input_dir,
                    output_dir=paths.output_dir,
                    case_id=item.case_id,
                    has_ground_truth=item.has_ground_truth,
                )

                job_svc.update_progress(item.job_id, 80)
                session.commit()

                # Persist result
                result_svc.create_result(
                    job_id=item.job_id,
                    case_id=item.case_id,
                    inference_result=inference_result,
                )

                # Transition: RUNNING → COMPLETED
                job_svc.complete_job(item.job_id)
                session.commit()

            except Exception as exc:
                session.rollback()
                # Persist failure — safe user-facing message
                safe_message = (
                    "Mock inference failed. The synthetic segmentation could not "
                    "be generated. This is not a clinical result."
                )
                logger.exception(
                    "job_inference_failed",
                    extra={
                        "job_id": item.job_id,
                        "case_id": item.case_id,
                    },
                )
                try:
                    job_svc.fail_job(
                        item.job_id,
                        ErrorCode.INFERENCE_FAILED,
                        safe_message,
                    )
                    session.commit()
                except Exception:
                    logger.exception(
                        "job_fail_persist_error",
                        extra={"job_id": item.job_id},
                    )
                    session.rollback()

        finally:
            session.close()


class _JobItem:
    """Internal queue item."""

    __slots__ = ("job_id", "case_id", "job_type", "has_ground_truth")

    def __init__(
        self,
        job_id: str,
        case_id: str,
        job_type: JobType,
        has_ground_truth: bool,
    ) -> None:
        self.job_id = job_id
        self.case_id = case_id
        self.job_type = job_type
        self.has_ground_truth = has_ground_truth


# Sentinel value to signal worker shutdown
_STOP_SENTINEL = _JobItem(
    job_id="__STOP__",
    case_id="__STOP__",
    job_type=JobType.PREDICT,
    has_ground_truth=False,
)
