from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_case_service, get_job_queue, get_job_service
from app.core.constants import ErrorCode, JobType, REQUIRED_MODALITIES
from app.db.session import get_session
from app.core.errors import AppError
from app.schemas.cases import PredictAcceptedResponse
from app.services.case_service import CaseService
from app.services.job_queue import JobQueue
from app.services.job_service import JobService

router = APIRouter(tags=["ingest"])


@router.post("/predict", response_model=PredictAcceptedResponse, status_code=202)
async def predict(
    t1: UploadFile | None = File(None),
    t1ce: UploadFile | None = File(None),
    t2: UploadFile | None = File(None),
    flair: UploadFile | None = File(None),
    session: Session = Depends(get_session),
    service: CaseService = Depends(get_case_service),
    job_service: JobService = Depends(get_job_service),
    queue: JobQueue = Depends(get_job_queue),
) -> PredictAcceptedResponse:
    """Submit a case for prediction (mock inference).

    Phase 3: creates a case, validates it, creates a QUEUED job,
    enqueues it for async execution, and returns 202 Accepted.
    """
    return await _ingest_and_enqueue(
        session,
        service,
        job_service,
        queue,
        {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair},
        job_type=JobType.PREDICT,
        require_seg=False,
    )


@router.post("/evaluate", response_model=PredictAcceptedResponse, status_code=202)
async def evaluate(
    t1: UploadFile | None = File(None),
    t1ce: UploadFile | None = File(None),
    t2: UploadFile | None = File(None),
    flair: UploadFile | None = File(None),
    seg: UploadFile | None = File(None),
    session: Session = Depends(get_session),
    service: CaseService = Depends(get_case_service),
    job_service: JobService = Depends(get_job_service),
    queue: JobQueue = Depends(get_job_queue),
) -> PredictAcceptedResponse:
    """Submit a case for evaluation (mock inference + seg stored).

    Phase 3: accepts ground-truth seg, creates a QUEUED job.
    Dice/HD95 are NOT computed in Phase 3.
    """
    return await _ingest_and_enqueue(
        session,
        service,
        job_service,
        queue,
        {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair, "seg": seg},
        job_type=JobType.EVALUATE,
        require_seg=True,
    )


@router.post("/cases/{case_id}/predict", response_model=PredictAcceptedResponse, status_code=202)
def predict_existing_case(
    case_id: str,
    session: Session = Depends(get_session),
    service: CaseService = Depends(get_case_service),
    job_service: JobService = Depends(get_job_service),
    queue: JobQueue = Depends(get_job_queue),
) -> PredictAcceptedResponse:
    """Start mock inference for an already uploaded and READY case."""
    case = service.get_case(case_id)
    if case.status != "READY":
        raise AppError(
            ErrorCode.INVALID_CASE_STATE,
            f"Case {case_id} is not READY.",
            status_code=400,
        )

    try:
        job = job_service.create_job(case.case_id, JobType.PREDICT)
    except Exception:
        raise AppError(
            ErrorCode.JOB_CREATION_FAILED,
            "Failed to create inference job.",
            status_code=500,
            case_id=case.case_id,
        ) from None

    session.commit()
    queue.enqueue(
        job_id=job.job_id,
        case_id=case.case_id,
        job_type=JobType.PREDICT,
        has_ground_truth=case.has_ground_truth,
    )

    return PredictAcceptedResponse(
        job_id=job.job_id,
        case_id=case.case_id,
        status=job.status,
        inference="mock",
        message="Job queued for mock inference. Poll GET /api/v1/jobs/{job_id} for status.",
    )


from sqlalchemy.orm import Session

async def _ingest_and_enqueue(
    session: Session,
    service: CaseService,
    job_service: JobService,
    queue: JobQueue,
    files: dict[str, UploadFile | None],
    *,
    job_type: JobType,
    require_seg: bool,
) -> PredictAcceptedResponse:
    """Ingest case → validate → create job → enqueue → return 202.

    Failure boundaries:
    - Case creation failure → 500 (no orphans)
    - Job creation failure → logged, case exists but no job
    - Queue enqueue failure → job stays QUEUED (detectable on restart)
    """
    # 1. Validate required modalities are present
    missing = [name for name in REQUIRED_MODALITIES if files.get(name) is None]
    if require_seg and files.get("seg") is None:
        missing.append("seg")
    if missing:
        raise AppError(
            ErrorCode.MISSING_MODALITY,
            f"Required modality '{missing[0].upper()}' is missing.",
            status_code=400,
        )

    # 2. Create case and upload files
    case = service.create_case()
    for name, upload in files.items():
        if upload is None:
            continue
        await service.add_file(case.case_id, name, upload)
        case = service.get_case(case.case_id)

    # 3. Validate case
    case = service.complete_case(case.case_id)

    # 4. Create persistent job (QUEUED)
    try:
        job = job_service.create_job(case.case_id, job_type)
    except Exception:
        raise AppError(
            ErrorCode.JOB_CREATION_FAILED,
            "Failed to create inference job.",
            status_code=500,
            case_id=case.case_id,
        ) from None

    # 5. Commit transaction so the worker can find the job, then enqueue
    session.commit()
    queue.enqueue(
        job_id=job.job_id,
        case_id=case.case_id,
        job_type=job_type,
        has_ground_truth=case.has_ground_truth,
    )

    return PredictAcceptedResponse(
        job_id=job.job_id,
        case_id=case.case_id,
        status=job.status,
        inference="mock",
        message="Job queued for mock inference. Poll GET /api/v1/jobs/{job_id} for status.",
    )
