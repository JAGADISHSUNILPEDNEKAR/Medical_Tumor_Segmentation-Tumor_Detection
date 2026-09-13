from datetime import datetime

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    """Response for GET /api/v1/jobs/{job_id}."""

    job_id: str
    case_id: str
    job_type: str
    status: str
    progress: int
    inference_source: str
    model_version: str | None = None
    result_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
