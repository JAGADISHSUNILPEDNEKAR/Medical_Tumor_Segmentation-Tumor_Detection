from datetime import datetime

from pydantic import BaseModel, Field


class CaseCreateResponse(BaseModel):
    case_id: str
    status: str
    created_at: datetime
    modalities_present: list[str]
    has_ground_truth: bool


class ModalityFileInfo(BaseModel):
    modality: str
    original_filename: str
    size_bytes: int
    shape: list[int] | None = None
    spacing: list[float] | None = None
    warnings: list[str] = Field(default_factory=list)


class SpatialChecks(BaseModel):
    shape_consistent: bool | None = None
    affine_consistent: bool | None = None
    spacing_consistent: bool | None = None
    notes: list[str] = Field(default_factory=list)


class CaseValidationReport(BaseModel):
    modalities: dict[str, bool]
    spatial: SpatialChecks
    warnings: list[str] = Field(default_factory=list)
    ready_for_prediction: bool
    ready_for_evaluation: bool


class CaseResponse(BaseModel):
    case_id: str
    status: str
    created_at: datetime
    modalities_present: list[str]
    has_ground_truth: bool
    total_bytes: int
    files: list[ModalityFileInfo]
    validation: CaseValidationReport | None = None
    error: str | None = None
    detail: str | None = None
    inference: str = "not_started"


class PredictAcceptedResponse(CaseResponse):
    """PRD `/predict` and `/evaluate` return a validated case in Phase 2.

    A `job_id` is not issued because the async job queue is Phase 3.
    HTTP 201 is used instead of 202 for that reason.
    """

    job_id: str | None = None
    message: str
