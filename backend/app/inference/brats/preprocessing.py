"""Preprocessing, transcribed from notebook Section 8.

Training and inference must apply byte-identical preprocessing, so the two
functions the notebook defines are reproduced here unchanged in behaviour:

  * `foreground_zscore_normalize` — per-case, per-modality z-score computed
    over the non-zero voxels only, leaving background voxels at exactly 0.
    BraTS volumes are skull-stripped, so "non-zero" is the foreground
    definition the notebook used (Sections 5.3 and 8.3).
  * `compute_foreground_bbox` — bounding box of the union of the four raw
    modalities' non-zero voxels, expanded by a 4-voxel margin and clipped to
    the volume.

Deliberately absent, because the notebook does not do them: resampling
(Section 5.1 measured the dataset as uniform 1.0mm), reorientation
(Section 8.1 measured a uniform L/P/S orientation), intensity clipping,
percentile normalization, global dataset statistics, and skull stripping.

The one production-only step is `np.squeeze` on each loaded volume, which
drops trailing singleton axes so a (X, Y, Z, 1) upload is handled the same way
the Phase 2 validator and MockInferenceService already handle it. It is a
no-op for the 3D volumes the notebook trained on and changes no voxel value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np

from app.inference.brats.config import (
    MODALITY_ORDER,
    BratsPreprocessingConfig,
)


@dataclass
class PreprocessedCase:
    """Everything the inference and export stages need from preprocessing."""

    image: np.ndarray
    """(4, D, H, W) float32, channels ordered [T1, T1ce, T2, FLAIR]."""

    original_shape: tuple[int, ...]
    """Spatial shape before cropping — the shape the output must return to."""

    affine: np.ndarray
    """4x4 affine of the reference (T1) volume."""

    spacing: tuple[float, float, float]
    """Voxel spacing in mm, from the reference volume's header."""

    crop_bbox: tuple[slice, slice, slice] | None
    """Slices into the original volume, or None when cropping was skipped."""

    norm_stats: dict[str, dict[str, float]]
    """Per-modality foreground mean/std. Provenance only, never reused."""


def foreground_zscore_normalize(
    volume: np.ndarray,
    background_value: float = 0.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Per-volume foreground-only z-score (notebook 8.3).

    Background voxels stay at exactly 0 in the normalized output: 0 remains
    the "no signal" sentinel for cropping and for sliding-window padding.
    """
    fg_mask = volume != background_value
    if not fg_mask.any():
        return np.zeros_like(volume, dtype=np.float32), 0.0, 1.0
    fg = volume[fg_mask]
    mean = float(fg.mean())
    std = max(float(fg.std()), eps)
    normalized = volume.astype(np.float32).copy()
    normalized[fg_mask] = (fg - mean) / std
    return normalized, mean, std


def compute_foreground_bbox(
    union_fg_mask: np.ndarray, margin: int = 4
) -> tuple[slice, slice, slice]:
    """Bounding box of a boolean mask, expanded by `margin` and clipped."""
    coords = np.argwhere(union_fg_mask)
    if coords.size == 0:
        raise ValueError("Foreground mask is empty — cannot compute a bounding box.")
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + margin, union_fg_mask.shape)
    bbox = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    return bbox  # type: ignore[return-value]


def load_and_preprocess_case(
    modality_paths: Mapping[str, Path],
    config: BratsPreprocessingConfig,
) -> PreprocessedCase:
    """Load the four modalities, normalize, and crop (notebook 8.3).

    `modality_paths` must contain the keys in `MODALITY_ORDER`; the stacking
    order is taken from `MODALITY_ORDER`, never from the mapping's own order
    and never alphabetically.
    """
    missing = [name for name in MODALITY_ORDER if name not in modality_paths]
    if missing:
        raise ValueError(
            f"Missing modality input(s) for preprocessing: {missing}. "
            f"Required order: {list(MODALITY_ORDER)}."
        )

    raw_volumes: list[np.ndarray] = []
    affine: np.ndarray | None = None
    spacing: tuple[float, float, float] | None = None
    original_shape: tuple[int, ...] | None = None

    for modality in MODALITY_ORDER:
        img = nib.load(str(modality_paths[modality]))
        if config.orientation_uniform_across_dataset is False:
            img = nib.as_closest_canonical(img)
        data = np.squeeze(np.asanyarray(img.dataobj, dtype=np.float32))
        if affine is None:
            affine = np.asarray(img.affine, dtype=np.float64)
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            original_shape = tuple(int(s) for s in data.shape)
        elif tuple(int(s) for s in data.shape) != original_shape:
            raise ValueError(
                f"Modality '{modality}' has shape {data.shape}, expected "
                f"{original_shape} (matching the reference modality "
                f"'{MODALITY_ORDER[0]}')."
            )
        raw_volumes.append(data)

    assert affine is not None and spacing is not None and original_shape is not None

    norm_stats: dict[str, dict[str, float]] = {}
    normalized: list[np.ndarray] = []
    for modality, volume in zip(MODALITY_ORDER, raw_volumes):
        norm_vol, mean, std = foreground_zscore_normalize(
            volume, config.background_value
        )
        normalized.append(norm_vol)
        norm_stats[modality] = {"mean": mean, "std": std}
    image = np.stack(normalized, axis=0)

    crop_bbox: tuple[slice, slice, slice] | None = None
    if config.crop_to_foreground:
        union_fg = np.any(
            np.stack(raw_volumes, axis=0) != config.background_value, axis=0
        )
        if union_fg.any():
            crop_bbox = compute_foreground_bbox(union_fg, config.crop_margin_voxels)
            image = image[(slice(None),) + crop_bbox]

    return PreprocessedCase(
        image=image,
        original_shape=original_shape,
        affine=affine,
        spacing=spacing,
        crop_bbox=crop_bbox,
        norm_stats=norm_stats,
    )
