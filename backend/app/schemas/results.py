from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SegmentationInfo(BaseModel):
    """Segmentation artifact availability."""

    available: bool


class RegionStats(BaseModel):
    """Per-region geometry and statistics."""

    region: str
    label: int
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    present: bool
    bounding_box: dict[str, list[int]] | None = None
    centroid_mm: list[float] | None = None


class MeasurementsInfo(BaseModel):
    """Synthetic or real geometric measurements."""

    synthetic: bool
    description: str | None = None
    voxel_spacing_mm: list[float] | None = None
    voxel_volume_mm3: float = 0.0
    foreground_voxels: int = 0
    foreground_volume_mm3: float = 0.0
    foreground_volume_cm3: float = 0.0
    regions: list[RegionStats] | None = None


class DiceMetric(BaseModel):
    value: float
    both_empty: bool = False


class HD95Metric(BaseModel):
    value_mm: float | None
    defined: bool
    reason: str | None = None


class PerClassMetrics(BaseModel):
    class_name: str
    label: int
    dice: DiceMetric | None = None
    hd95: HD95Metric | None = None


class EvaluationInfo(BaseModel):
    """Evaluation metrics availability."""

    available: bool
    ground_truth_available: bool = False
    per_class: list[PerClassMetrics] | None = None
    mean_dice: DiceMetric | None = None
    mean_hd95: HD95Metric | None = None


class ProvenanceInfo(BaseModel):
    inference_source: str
    model_version: str | None = None
    checkpoint_id: str | None = None
    synthetic: bool = True
    inference_timestamp: datetime | None = None
    description: str | None = None


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
    provenance: ProvenanceInfo | None = None
    created_at: datetime | None = None


class ReportResponse(BaseModel):
    """Structured report data for a result."""
    
    result_id: str
    case_id: str
    job_id: str
    status: str
    measurements: MeasurementsInfo
    evaluation: EvaluationInfo
    provenance: ProvenanceInfo
    created_at: datetime
