from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SegmentationInfo(BaseModel):
    """Segmentation artifact availability."""

    available: bool


class MeasurementsInfo(BaseModel):
    """Synthetic geometric measurements."""

    synthetic: bool
    description: str | None = None
    foreground_voxels: int = 0
    foreground_volume_mm3: float = 0.0
    foreground_volume_cm3: float = 0.0
    regions: dict[str, Any] | None = None


class EvaluationInfo(BaseModel):
    """Evaluation metrics availability.

    Phase 3: Dice/HD95 are not computed. available=False.
    """

    available: bool
    dice: dict[str, float] | None = None
    hd95_mm: dict[str, float] | None = None


class ResultResponse(BaseModel):
    """Response for GET /api/v1/results/{result_id}."""

    result_id: str
    job_id: str
    case_id: str
    status: str
    inference_source: str
    model_version: str | None = None
    segmentation: SegmentationInfo
    measurements: MeasurementsInfo
    evaluation: EvaluationInfo
    created_at: datetime | None = None
