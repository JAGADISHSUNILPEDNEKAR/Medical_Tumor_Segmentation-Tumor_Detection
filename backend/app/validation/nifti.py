from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage

from app.core.constants import (
    AFFINE_ATOL,
    AFFINE_RTOL,
    BRATS_SEG_LABELS,
    SPACING_ATOL,
    SPACING_RTOL,
    ErrorCode,
)
from app.core.errors import AppError


@dataclass
class VolumeInfo:
    shape: tuple[int, ...]
    affine: list[list[float]]
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    warnings: list[str] = field(default_factory=list)
    unique_labels: list[int] | None = None


def inspect_nifti(path: Path, *, is_segmentation: bool, case_id: str | None = None) -> VolumeInfo:
    """Parse and hard-validate a NIfTI volume. Does not resample."""
    try:
        image = nib.load(str(path), mmap=False)
    except Exception:
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "The file could not be parsed as NIfTI.",
            status_code=422,
            case_id=case_id,
        ) from None

    if not isinstance(image, SpatialImage):
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "The file is not a spatial NIfTI image.",
            status_code=422,
            case_id=case_id,
        )

    try:
        data = np.asanyarray(image.dataobj)
    except Exception:
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "The NIfTI header was readable but voxel data could not be loaded.",
            status_code=422,
            case_id=case_id,
        ) from None

    data = np.squeeze(data)
    if data.ndim != 3:
        raise AppError(
            ErrorCode.INVALID_DIMENSION,
            f"Expected a 3D volume after squeezing singleton axes, received shape {tuple(int(x) for x in data.shape)}.",
            status_code=422,
            case_id=case_id,
        )

    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "NIfTI affine must be a finite 4×4 matrix.",
            status_code=422,
            case_id=case_id,
        )

    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    if len(zooms) != 3 or any(z <= 0 or not np.isfinite(z) for z in zooms):
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "Voxel spacing must be finite and positive on all three axes.",
            status_code=422,
            case_id=case_id,
        )

    if not np.isfinite(data).all():
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "Volume contains non-finite voxel values.",
            status_code=422,
            case_id=case_id,
        )

    warnings: list[str] = []
    if np.all(data == 0):
        warnings.append("Volume is entirely zero. Technically valid, but unusual for MRI.")

    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, rtol=1e-3, atol=1e-3):
        warnings.append("qform and sform differ; the loader affine was used. No resampling was applied.")

    unique_labels: list[int] | None = None
    if is_segmentation:
        unique_labels = sorted({int(v) for v in np.unique(data)})
        unexpected = [label for label in unique_labels if label not in BRATS_SEG_LABELS]
        if unexpected:
            raise AppError(
                ErrorCode.INVALID_SEGMENTATION_LABEL,
                (
                    f"Segmentation contains labels {unexpected} that are not BraTS labels "
                    f"{{0, 1, 2, 4}}. Label 3 is not accepted as an alias for enhancing tumor."
                ),
                status_code=422,
                case_id=case_id,
            )

    orientation = tuple(str(code) for code in nib.aff2axcodes(affine))
    return VolumeInfo(
        shape=tuple(int(x) for x in data.shape),
        affine=affine.tolist(),
        spacing=zooms,
        orientation=(orientation[0], orientation[1], orientation[2]),
        warnings=warnings,
        unique_labels=unique_labels,
    )


def assert_spatially_compatible(
    volumes: dict[str, VolumeInfo],
    case_id: str | None = None,
) -> list[str]:
    """Compare shape, affine, and spacing. Returns notes for the validation report."""
    if not volumes:
        return []

    reference_name, reference = next(iter(volumes.items()))
    notes: list[str] = [f"Reference volume: {reference_name.upper()}."]

    for name, volume in volumes.items():
        if volume.shape != reference.shape:
            raise AppError(
                ErrorCode.SHAPE_MISMATCH,
                (
                    f"Modality {name.upper()} shape {volume.shape} does not match "
                    f"{reference_name.upper()} shape {reference.shape}."
                ),
                status_code=422,
                case_id=case_id,
            )
        if not np.allclose(
            np.asarray(volume.spacing),
            np.asarray(reference.spacing),
            rtol=SPACING_RTOL,
            atol=SPACING_ATOL,
        ):
            raise AppError(
                ErrorCode.SPACING_MISMATCH,
                (
                    f"Modality {name.upper()} spacing {volume.spacing} does not match "
                    f"{reference_name.upper()} spacing {reference.spacing}."
                ),
                status_code=422,
                case_id=case_id,
            )
        if not np.allclose(
            np.asarray(volume.affine),
            np.asarray(reference.affine),
            rtol=AFFINE_RTOL,
            atol=AFFINE_ATOL,
        ):
            raise AppError(
                ErrorCode.AFFINE_MISMATCH,
                (
                    f"Modality {name.upper()} affine does not match {reference_name.upper()}. "
                    "Volumes must already be co-registered. This service does not resample on upload."
                ),
                status_code=422,
                case_id=case_id,
            )
        if volume.orientation != reference.orientation:
            raise AppError(
                ErrorCode.AFFINE_MISMATCH,
                (
                    f"Modality {name.upper()} orientation {volume.orientation} does not match "
                    f"{reference_name.upper()} orientation {reference.orientation}."
                ),
                status_code=422,
                case_id=case_id,
            )

    notes.append("Shape, voxel spacing, affine, and orientation are consistent across uploaded volumes.")
    return notes
