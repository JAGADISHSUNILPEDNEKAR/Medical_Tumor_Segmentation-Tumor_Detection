"""Placeholder for a real backend that was configured but could not load.

When INFERENCE_BACKEND=pytorch and the checkpoint is missing, corrupt, or
architecturally incompatible, the API still starts — a GPU or a checkpoint is
never required merely to serve `/health` — but it must not pretend inference is
available. This service reports the failure honestly and refuses to run.
"""

from __future__ import annotations

from pathlib import Path

from app.inference.base import InferenceResult


class InferenceUnavailableError(RuntimeError):
    """Raised if a job somehow reaches an unavailable inference backend."""


class UnavailableInferenceService:
    """Conforms to the InferenceService protocol; always refuses to predict."""

    inference_source = "unavailable"
    synthetic = False
    model_loaded = False
    available = False
    model_version = None

    def __init__(self, reason: str, detail: str | None = None) -> None:
        # `reason` is client-facing and must stay free of filesystem paths and
        # other server internals. `detail` is the full operator diagnostic and
        # is logged, never returned over HTTP.
        self.reason = reason
        self.detail = detail or reason

    def describe(self) -> dict[str, object]:
        return {
            "inference_source": self.inference_source,
            "model_loaded": False,
            "model_version": None,
            "checkpoint_id": None,
            "message": self.reason,
        }

    def predict(
        self,
        case_dir: Path,
        output_dir: Path,
        case_id: str,
        *,
        has_ground_truth: bool = False,
    ) -> InferenceResult:
        raise InferenceUnavailableError(self.reason)
