from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import (
    get_case_service,
    get_inference_service,
    get_job_queue,
    get_job_service,
)
from app.core.constants import CaseStatus, ErrorCode, JobType, REQUIRED_MODALITIES
from app.core.errors import AppError
from app.db.session import get_session
from app.inference.base import (
    inference_source_of,
    is_available,
    model_version_of,
)
from app.schemas.cases import PredictAcceptedResponse
from app.services.case_service import CaseService
from app.services.job_queue import JobQueue
from app.services.job_service import JobService

router = APIRouter(tags=["ingest"])


def _assert_inference_available(service, case_id: str | None = None) -> None:
    """Reject the request before any upload work when no backend can run.

    A configured-but-unloadable real backend must not silently accept jobs
    that are guaranteed to fail, and must never fall back to synthetic
    output dressed up as a prediction.
    """
    if is_available(service):
        return
    reason = getattr(service, "reason", None) or (
        "Inference is not available on this server."
    )
    raise AppError(
        ErrorCode.MODEL_UNAVAILABLE,
        reason,
        status_code=503,
        case_id=case_id,
    )


def _queued_message(source: str) -> str:
    label = {
        "pytorch": "trained 3D U-Net inference",
        "mock": "mock inference",
    }.get(source, f"{source} inference")
    return (
        f"Job queued for {label}. "
        "Poll GET /api/v1/jobs/{job_id} for status."
    )


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
    inference_service=Depends(get_inference_service),
) -> PredictAcceptedResponse:
    """Submit a case for prediction.

    Creates a case, validates it, creates a QUEUED job, enqueues it for async
    execution, and returns 202 Accepted. Inference itself never runs inside
    this request — it runs on the single-worker job queue.
    """
    return await _ingest_and_enqueue(
        session,
        service,
        job_service,
        queue,
        {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair},
        job_type=JobType.PREDICT,
        require_seg=False,
        inference_service=inference_service,
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
    inference_service=Depends(get_inference_service),
) -> PredictAcceptedResponse:
    """Submit a case for evaluation (inference plus stored ground truth).

    Dice/HD95 are computed by MetricsService once the job produces a
    segmentation.
    """
    return await _ingest_and_enqueue(
        session,
        service,
        job_service,
        queue,
        {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair, "seg": seg},
        job_type=JobType.EVALUATE,
        require_seg=True,
        inference_service=inference_service,
    )


@router.post("/cases/{case_id}/predict", response_model=PredictAcceptedResponse, status_code=202)
def predict_existing_case(
    case_id: str,
    session: Session = Depends(get_session),
    service: CaseService = Depends(get_case_service),
    job_service: JobService = Depends(get_job_service),
    queue: JobQueue = Depends(get_job_queue),
    inference_service=Depends(get_inference_service),
) -> PredictAcceptedResponse:
    """Start inference for an already uploaded and READY case."""
    _assert_inference_available(inference_service, case_id)
    case = service.get_case(case_id)
    if case.status != CaseStatus.READY:
        raise AppError(
            ErrorCode.INVALID_CASE_STATE,
            (
                f"Case is {case.status} and cannot be submitted for inference. "
                "Upload all four modalities (T1, T1ce, T2, FLAIR) and complete "
                "the case so it reaches READY, then try again."
            ),
            status_code=409,
            case_id=case.case_id,
        )

    inference_source = inference_source_of(inference_service)
    try:
        job = job_service.create_job(
            case.case_id,
            JobType.PREDICT,
            inference_source=inference_source,
            model_version=model_version_of(inference_service),
        )
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
        inference=inference_source,
        message=_queued_message(inference_source),
    )


async def _ingest_and_enqueue(
    session: Session,
    service: CaseService,
    job_service: JobService,
    queue: JobQueue,
    files: dict[str, UploadFile | None],
    *,
    job_type: JobType,
    require_seg: bool,
    inference_service,
) -> PredictAcceptedResponse:
    """Ingest case → validate → create job → enqueue → return 202.

    Failure boundaries:
    - Case creation failure → 500 (no orphans)
    - Job creation failure → logged, case exists but no job
    - Queue enqueue failure → job stays QUEUED (detectable on restart)
    """
    # 0. Refuse early if no backend can run — before any bytes are stored.
    _assert_inference_available(inference_service)

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
    inference_source = inference_source_of(inference_service)
    try:
        job = job_service.create_job(
            case.case_id,
            job_type,
            inference_source=inference_source,
            model_version=model_version_of(inference_service),
        )
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
        inference=inference_source,
        message=_queued_message(inference_source),
    )
