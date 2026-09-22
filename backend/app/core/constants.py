"""Case, NIfTI validation, and job lifecycle constants.

HARD failures reject the upload/case.
WARNINGS are recorded and returned but do not block READY.
"""

from enum import StrEnum

import numpy as np

# ── Modalities ────────────────────────────────────────────────────────────────
REQUIRED_MODALITIES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
OPTIONAL_MODALITIES: tuple[str, ...] = ("seg",)
ALL_MODALITIES: tuple[str, ...] = REQUIRED_MODALITIES + OPTIONAL_MODALITIES

# ── BraTS segmentation labels ────────────────────────────────────────────────
# Raw BraTS dataset labels. Do not remap 3 → 4 at upload time.
BRATS_SEG_LABELS: frozenset[int] = frozenset({0, 1, 2, 4})

# Internal labels used during synthetic segmentation generation.
# Label 3 is used internally for ET; it is remapped to 4 on export.
INTERNAL_BACKGROUND = 0
INTERNAL_NCR = 1
INTERNAL_ED = 2
INTERNAL_ET = 3

# BraTS export labels
BRATS_BACKGROUND = 0
BRATS_NCR = 1
BRATS_ED = 2
BRATS_ET = 4

# Human-readable region names for measurements
REGION_NAMES: dict[int, str] = {
    BRATS_NCR: "NCR (Necrotic Core)",
    BRATS_ED: "ED (Peritumoral Edema)",
    BRATS_ET: "ET (Enhancing Tumor)",
}

ALLOWED_NIFTI_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")

# Controlled viewer artifacts. Keys are the only values accepted in the URL.
# Values: (folder, stored stem) — never built from raw user path segments.
VIEWER_ARTIFACTS: dict[str, tuple[str, str]] = {
    "t1": ("input", "t1"),
    "t1ce": ("input", "t1ce"),
    "t2": ("input", "t2"),
    "flair": ("input", "flair"),
    "segmentation": ("output", "segmentation"),
}

# ── Spatial tolerances ────────────────────────────────────────────────────────
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4
SPACING_RTOL = 1e-5
SPACING_ATOL = 1e-4


# ── Case states ──────────────────────────────────────────────────────────────
class CaseStatus(StrEnum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


# ── Job states ───────────────────────────────────────────────────────────────
class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Job types ────────────────────────────────────────────────────────────────
class JobType(StrEnum):
    PREDICT = "PREDICT"
    EVALUATE = "EVALUATE"


# ── Error codes ──────────────────────────────────────────────────────────────
class ErrorCode(StrEnum):
    # Case errors (Phase 2)
    CASE_NOT_FOUND = "CASE_NOT_FOUND"
    MISSING_MODALITY = "MISSING_MODALITY"
    DUPLICATE_MODALITY = "DUPLICATE_MODALITY"
    INVALID_EXTENSION = "INVALID_EXTENSION"
    INVALID_NIFTI = "INVALID_NIFTI"
    INVALID_DIMENSION = "INVALID_DIMENSION"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    AFFINE_MISMATCH = "AFFINE_MISMATCH"
    SPACING_MISMATCH = "SPACING_MISMATCH"
    INVALID_SEGMENTATION_LABEL = "INVALID_SEGMENTATION_LABEL"
    CASE_TOO_LARGE = "CASE_TOO_LARGE"
    STORAGE_ERROR = "STORAGE_ERROR"
    INVALID_MODALITY = "INVALID_MODALITY"
    CASE_NOT_MUTABLE = "CASE_NOT_MUTABLE"
    INVALID_CASE_STATE = "INVALID_CASE_STATE"

    # Job/inference errors (Phase 3)
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    RESULT_NOT_FOUND = "RESULT_NOT_FOUND"
    JOB_CREATION_FAILED = "JOB_CREATION_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RESULT_CREATION_FAILED = "RESULT_CREATION_FAILED"
    INVALID_JOB_STATE = "INVALID_JOB_STATE"

    # Measurements and Metrics (Phase 5)
    MEASUREMENT_FAILED = "MEASUREMENT_FAILED"
    METRICS_FAILED = "METRICS_FAILED"
    SEGMENTATION_UNAVAILABLE = "SEGMENTATION_UNAVAILABLE"

    # Artifact access (Phase 4)
    INVALID_ARTIFACT = "INVALID_ARTIFACT"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"


def to_brats_labels(mask: np.ndarray) -> np.ndarray:
    """Remap internal label 3 (ET) → BraTS label 4 (ET).

    This is the single, centralized location for internal → BraTS label
    conversion. Both MockInferenceService and the future
    RealBraTSInferenceService must call this function when writing exported
    segmentation artifacts.

    Internal labels: {0, 1, 2, 3}
    BraTS labels:    {0, 1, 2, 4}
    """
    out = mask.copy()
    out[mask == INTERNAL_ET] = BRATS_ET
    return out
