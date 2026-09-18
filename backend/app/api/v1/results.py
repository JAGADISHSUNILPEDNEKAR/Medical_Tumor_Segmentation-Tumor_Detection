from fastapi import APIRouter, Depends

from app.api.deps import get_result_service
from app.schemas.results import ResultResponse
from app.services.result_service import ResultService

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{result_id}", response_model=ResultResponse)
def get_result(
    result_id: str,
    service: ResultService = Depends(get_result_service),
) -> ResultResponse:
    """Get the result of a completed inference job.

    Returns measurements, evaluation, and provenance data.
    Does not expose raw filesystem paths.
    Returns 404 RESULT_NOT_FOUND for unknown result IDs.
    """
    result = service.get_result(result_id)
    return service.to_response(result)


@router.get("/{result_id}/measurements", response_model=dict)
def get_measurements(
    result_id: str,
    service: ResultService = Depends(get_result_service),
) -> dict:
    """Get the measurements of a completed inference job."""
    result = service.get_result(result_id)
    response = service.to_response(result)
    return response.measurements.model_dump(mode="json")


@router.get("/{result_id}/report", response_model=dict)
def get_report(
    result_id: str,
    service: ResultService = Depends(get_result_service),
) -> dict:
    """Get the structured report data for a completed inference job."""
    result = service.get_result(result_id)
    response = service.to_response(result)
    
    from app.schemas.results import ReportResponse
    report = ReportResponse(
        result_id=response.result_id,
        case_id=response.case_id,
        job_id=response.job_id,
        status=response.status,
        measurements=response.measurements,
        evaluation=response.evaluation,
        provenance=response.provenance,
        created_at=response.created_at,
    )
    return report.model_dump(mode="json")
