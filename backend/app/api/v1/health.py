from fastapi import APIRouter

from app.schemas.health import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["system"])

_NO_CHECKPOINT_MESSAGE = (
    "No trained checkpoint is registered. Mock inference is available for "
    "pipeline validation. Headline Dice/HD95 metrics are not available."
)


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=False,
        inference_source="mock",
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        checkpoint_id=None,
        dataset_version=None,
        trained_on_split=None,
        headline_metrics=None,
        num_classes=4,
        input_modalities=["T1", "T1ce", "T2", "FLAIR"],
        model_loaded=False,
        inference_source="mock",
        model_version=None,
        message=_NO_CHECKPOINT_MESSAGE,
    )
