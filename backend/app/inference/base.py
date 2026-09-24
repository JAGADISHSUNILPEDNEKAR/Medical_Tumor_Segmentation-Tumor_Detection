"""Production inference boundary.

The web application must call this interface only. Concrete implementations
live beside it:

    MockInferenceService        — Phase 3, synthetic geometric mask
    RealBraTSInferenceService   — Phase 6, trained 3D U-Net
    UnavailableInferenceService — a configured real backend that failed to load

The `InferenceResult` contract is designed to be future-proof: additional
metadata fields (checkpoint_id, preprocessing_version, …) can be added without
breaking the API.

`InferenceService` stays a method-only Protocol so `runtime_checkable`
`isinstance` keeps working on every supported Python. The provenance
attributes implementations also carry (`inference_source`, `model_loaded`,
`available`, `synthetic`) are read through the accessor helpers below, which
fall back to conservative defaults for any implementation that omits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class InferenceResult:
    """Structured result returned by any InferenceService implementation.

    Attributes:
        segmentation_path: Relative path to the output segmentation NIfTI.
        metadata: Provenance and configuration metadata.
    """

    segmentation_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceService(Protocol):
    """Protocol for all inference implementations."""

    def predict(
        self,
        case_dir: Path,
        output_dir: Path,
        case_id: str,
        *,
        has_ground_truth: bool = False,
    ) -> InferenceResult:
        """Run segmentation for a validated case.

        Args:
            case_dir: Path to the case's input directory containing NIfTI files.
            output_dir: Path to the case's output directory for artifacts.
            case_id: Opaque case identifier for logging.
            has_ground_truth: Whether a ground-truth seg file exists.

        Returns:
            InferenceResult with segmentation path and metadata.
        """
        ...


def inference_source_of(service: object) -> str:
    """`mock`, `pytorch`, or `unavailable`. Never guessed from configuration."""
    return str(getattr(service, "inference_source", "unavailable"))


def model_loaded_of(service: object) -> bool:
    """True only when a real trained checkpoint was actually loaded."""
    return bool(getattr(service, "model_loaded", False))


def is_available(service: object) -> bool:
    """False when the configured backend could not be brought up."""
    return bool(getattr(service, "available", True))


def model_version_of(service: object) -> str | None:
    version = getattr(service, "model_version", None)
    return str(version) if version else None


def describe_service(service: object) -> dict[str, Any]:
    """Provenance payload for GET /model/info, if the service supplies one."""
    describe = getattr(service, "describe", None)
    if callable(describe):
        described = describe()
        if isinstance(described, dict):
            return described
    return {}
