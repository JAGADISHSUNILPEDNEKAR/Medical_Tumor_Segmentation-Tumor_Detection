"""Health and model-provenance endpoints.

Both report the state of the inference backend that actually loaded in this
process. `model_loaded` is true only when a real trained checkpoint was read
from disk and loaded into the network — never because a backend was merely
configured.
"""

from fastapi import APIRouter, Depends

from app.api.deps import get_inference_service
from app.inference.base import (
    describe_service,
    inference_source_of,
    model_loaded_of,
    model_version_of,
)
from app.schemas.health import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["system"])

_NO_CHECKPOINT_MESSAGE = (
    "No trained checkpoint is registered. Mock inference is available for "
    "pipeline validation. Headline Dice/HD95 metrics are not available."
)
_LOADED_MESSAGE_SUFFIX = (
    "Headline Dice/HD95 metrics are not registered for this checkpoint; the "
    "validation figures reported by the training notebook are not served here."
)


@router.get("/health", response_model=HealthResponse)
def get_health(service=Depends(get_inference_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded_of(service),
        inference_source=inference_source_of(service),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(service=Depends(get_inference_service)) -> ModelInfoResponse:
    described = describe_service(service)
    model_loaded = model_loaded_of(service)

    if model_loaded:
        architecture = described.get("architecture")
        message = f"{architecture} {_LOADED_MESSAGE_SUFFIX}".strip()
    else:
        message = str(described.get("message") or _NO_CHECKPOINT_MESSAGE)

    return ModelInfoResponse(
        checkpoint_id=described.get("checkpoint_id"),
        dataset_version=None,
        trained_on_split=None,
        # Never populated from a training-time proxy metric. A headline Dice /
        # HD95 pair belongs to a registered, validated checkpoint record, which
        # this prototype does not maintain.
        headline_metrics=None,
        num_classes=int(described.get("num_classes", 4)),
        input_modalities=list(
            described.get("input_modalities") or ["T1", "T1ce", "T2", "FLAIR"]
        ),
        model_loaded=model_loaded,
        inference_source=inference_source_of(service),
        model_version=model_version_of(service),
        message=message,
        details=described or None,
    )
