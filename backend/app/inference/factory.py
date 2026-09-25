"""Selection of the inference backend from configuration.

INFERENCE_BACKEND=mock     -> MockInferenceService      (default)
INFERENCE_BACKEND=pytorch  -> RealBraTSInferenceService (requires MODEL_PATH)

Torch is imported only on the `pytorch` path, so a mock-mode deployment — and
the existing test suite — runs on a machine with no PyTorch installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.inference.mock import MockInferenceService
from app.inference.unavailable import UnavailableInferenceService

logger = logging.getLogger(__name__)

MOCK_BACKEND = "mock"
PYTORCH_BACKEND = "pytorch"
SUPPORTED_BACKENDS = (MOCK_BACKEND, PYTORCH_BACKEND)


def build_inference_service(settings: Settings):
    """Construct the configured backend, loading its checkpoint eagerly.

    Never raises: a misconfigured or unloadable real backend yields an
    `UnavailableInferenceService` carrying the reason, so the API starts and
    reports the problem instead of crash-looping or silently degrading to mock
    inference (which would misrepresent synthetic output as a real prediction).
    """
    backend = (settings.inference_backend or MOCK_BACKEND).strip().lower()

    if backend == MOCK_BACKEND:
        logger.info("inference_backend_selected", extra={"backend": MOCK_BACKEND})
        return MockInferenceService()

    if backend != PYTORCH_BACKEND:
        reason = (
            f"The configured inference backend is not supported. "
            f"Supported values: {', '.join(SUPPORTED_BACKENDS)}."
        )
        detail = f"INFERENCE_BACKEND={settings.inference_backend!r} is not supported."
        logger.error("inference_backend_invalid", extra={"reason": detail})
        return UnavailableInferenceService(reason, detail)

    if not settings.model_registered:
        reason = (
            "INFERENCE_BACKEND=pytorch requires MODEL_PATH to point at the "
            "trained checkpoint exported by the notebook (best_model.pth). "
            "No checkpoint is registered, so real inference is unavailable."
        )
        logger.error("inference_backend_no_checkpoint", extra={"reason": reason})
        return UnavailableInferenceService(reason)

    try:
        from app.inference.pytorch_service import RealBraTSInferenceService
    except ImportError as exc:
        reason = (
            "Real inference is configured but PyTorch is not installed on this "
            "server, so no model could be loaded."
        )
        detail = f"INFERENCE_BACKEND=pytorch requires PyTorch: {exc}"
        logger.error("inference_backend_torch_missing", extra={"reason": detail})
        return UnavailableInferenceService(reason, detail)

    assert settings.model_path is not None
    service = RealBraTSInferenceService(
        Path(settings.model_path).expanduser(),
        device_preference=settings.inference_device,
        model_version=settings.model_version,
        strict_fingerprint=settings.inference_strict_fingerprint,
    )
    try:
        service.load()
    except Exception as exc:  # noqa: BLE001 - logged in full, never swallowed
        # The full diagnostic names the checkpoint path and the exact mismatch.
        # It goes to the logs; the client-facing reason stays path-free.
        detail = f"Trained checkpoint could not be loaded: {exc}"
        reason = (
            "The configured model checkpoint could not be loaded, so real "
            "inference is unavailable on this server. See the backend logs for "
            "the checkpoint diagnostic."
        )
        logger.error(
            "inference_backend_checkpoint_load_failed",
            extra={"reason": detail, "model_path": str(settings.model_path)},
        )
        return UnavailableInferenceService(reason, detail)

    logger.info(
        "inference_backend_selected",
        extra={
            "backend": PYTORCH_BACKEND,
            "checkpoint_id": service.checkpoint_id,
            "device": str(service.device),
        },
    )
    return service
