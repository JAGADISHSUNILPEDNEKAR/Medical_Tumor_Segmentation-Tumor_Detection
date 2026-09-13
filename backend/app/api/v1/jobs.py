from fastapi import APIRouter, Depends

from app.api.deps import get_job_service
from app.schemas.jobs import JobStatusResponse
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobStatusResponse:
    """Get the status of an inference job.

    Returns 200 with job details. Includes result_id when COMPLETED.
    Returns 404 JOB_NOT_FOUND for unknown job IDs.
    """
    job = service.get_job(job_id)
    return service.to_response(job)
