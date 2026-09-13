"""Production inference boundary.

The web application must call this interface only. Concrete implementations
(MockInferenceService, RealBraTSInferenceService) are added in later phases.

The InferenceResult contract is designed to be future-proof:
additional metadata fields (checkpoint_id, preprocessing_version, etc.)
can be added without breaking the API.
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
        measurements: Deterministic geometric measurements from the mask.
        metadata: Provenance and configuration metadata.
    """

    segmentation_path: str
    measurements: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceService(Protocol):
    """Protocol for all inference implementations.

    Phase 3: MockInferenceService
    Future:  RealBraTSInferenceService
    """

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
            InferenceResult with segmentation path, measurements, and metadata.
        """
        ...
