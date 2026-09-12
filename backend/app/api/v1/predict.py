from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_case_service
from app.core.constants import ErrorCode, REQUIRED_MODALITIES
from app.core.errors import AppError
from app.schemas.cases import PredictAcceptedResponse
from app.services.case_service import CaseService

router = APIRouter(tags=["ingest"])

_PHASE2_MESSAGE = (
    "Case validated and stored. Inference is not started in Phase 2; no job was queued."
)


@router.post("/predict", response_model=PredictAcceptedResponse, status_code=201)
async def predict(
    t1: UploadFile | None = File(None),
    t1ce: UploadFile | None = File(None),
    t2: UploadFile | None = File(None),
    flair: UploadFile | None = File(None),
    service: CaseService = Depends(get_case_service),
) -> PredictAcceptedResponse:
    """PRD ingest endpoint. Phase 2 validates and stores; it does not enqueue inference."""
    return await _ingest(service, {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}, require_seg=False)


@router.post("/evaluate", response_model=PredictAcceptedResponse, status_code=201)
async def evaluate(
    t1: UploadFile | None = File(None),
    t1ce: UploadFile | None = File(None),
    t2: UploadFile | None = File(None),
    flair: UploadFile | None = File(None),
    seg: UploadFile | None = File(None),
    service: CaseService = Depends(get_case_service),
) -> PredictAcceptedResponse:
    """PRD evaluation ingest. Phase 2 stores optional/required seg; it does not compute Dice/HD95."""
    return await _ingest(
        service,
        {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair, "seg": seg},
        require_seg=True,
    )


async def _ingest(
    service: CaseService,
    files: dict[str, UploadFile | None],
    *,
    require_seg: bool,
) -> PredictAcceptedResponse:
    missing = [name for name in REQUIRED_MODALITIES if files.get(name) is None]
    if require_seg and files.get("seg") is None:
        missing.append("seg")
    if missing:
        raise AppError(
            ErrorCode.MISSING_MODALITY,
            f"Required modality '{missing[0].upper()}' is missing.",
            status_code=400,
        )

    case = service.create_case()
    for name, upload in files.items():
        if upload is None:
            continue
        await service.add_file(case.case_id, name, upload)
        case = service.get_case(case.case_id)

    case = service.complete_case(case.case_id)
    body = service.to_response(case)
    return PredictAcceptedResponse(
        **body.model_dump(),
        job_id=None,
        message=_PHASE2_MESSAGE,
    )
