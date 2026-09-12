from fastapi import APIRouter, Depends, UploadFile

from app.api.deps import get_case_service
from app.schemas.cases import CaseCreateResponse, CaseResponse
from app.services.case_service import CaseService

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseCreateResponse, status_code=201)
def create_case(service: CaseService = Depends(get_case_service)) -> CaseCreateResponse:
    case = service.create_case()
    return CaseCreateResponse(
        case_id=case.case_id,
        status=case.status,
        created_at=case.created_at,
        modalities_present=[],
        has_ground_truth=False,
    )


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, service: CaseService = Depends(get_case_service)) -> CaseResponse:
    return service.to_response(service.get_case(case_id))


@router.post("/{case_id}/files/{modality}", response_model=CaseResponse)
async def upload_case_file(
    case_id: str,
    modality: str,
    file: UploadFile,
    service: CaseService = Depends(get_case_service),
) -> CaseResponse:
    case = await service.add_file(case_id, modality, file)
    return service.to_response(case)


@router.post("/{case_id}/complete", response_model=CaseResponse)
def complete_case(case_id: str, service: CaseService = Depends(get_case_service)) -> CaseResponse:
    case = service.complete_case(case_id)
    return service.to_response(case)
