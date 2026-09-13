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

    Returns synthetic measurements, segmentation availability,
    and evaluation availability (not available in Phase 3).

    Does not expose raw filesystem paths.
    Returns 404 RESULT_NOT_FOUND for unknown result IDs.
    """
    result = service.get_result(result_id)
    return service.to_response(result)
