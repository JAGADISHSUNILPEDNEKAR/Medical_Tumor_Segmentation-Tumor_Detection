from app.inference.base import (
    InferenceResult,
    InferenceService,
    describe_service,
    inference_source_of,
    is_available,
    model_loaded_of,
    model_version_of,
)
from app.inference.factory import build_inference_service
from app.inference.mock import MockInferenceService
from app.inference.unavailable import (
    InferenceUnavailableError,
    UnavailableInferenceService,
)

__all__ = [
    "InferenceResult",
    "InferenceService",
    "InferenceUnavailableError",
    "MockInferenceService",
    "UnavailableInferenceService",
    "build_inference_service",
    "describe_service",
    "inference_source_of",
    "is_available",
    "model_loaded_of",
    "model_version_of",
]
