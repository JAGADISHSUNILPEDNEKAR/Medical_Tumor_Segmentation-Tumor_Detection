"""Post-processing, transcribed from notebook Sections 20.1 and 21.1.

Two operations, both from the notebook and nothing else. No clinical
heuristics, no extra thresholds, no region merging.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from app.inference.brats.config import LABEL_MAP_INTERNAL_TO_RAW

CONNECTED_COMPONENT_CLASSES: tuple[int, ...] = (1, 2, 3)


def connected_component_filter(
    pred_internal: np.ndarray,
    min_voxels: int = 50,
) -> tuple[np.ndarray, dict[int, int]]:
    """Remove connected components smaller than `min_voxels` (notebook 20.1).

    Applied independently to each foreground class (1=NCR, 2=ED, 3=ET) rather
    than to the union mask, so anatomically separate regions of different tumor
    types are never bridged into one component. Background is never removed.

    Returns the cleaned mask and the voxel count removed per internal class.
    """
    cleaned = pred_internal.copy()
    removed: dict[int, int] = {}

    for class_id in CONNECTED_COMPONENT_CLASSES:
        binary_mask = pred_internal == class_id
        if not binary_mask.any():
            removed[class_id] = 0
            continue

        labeled_array, num_features = ndi.label(binary_mask)
        if num_features == 0:
            removed[class_id] = 0
            continue

        component_sizes = np.bincount(labeled_array.ravel())
        remove_mask = (labeled_array != 0) & (
            component_sizes[labeled_array] < min_voxels
        )
        removed[class_id] = int(remove_mask.sum())
        cleaned[remove_mask] = 0

    return cleaned, removed


def internal_to_brats_labels(pred_internal: np.ndarray) -> np.ndarray:
    """Restore raw BraTS IDs from the network's contiguous IDs (3 → 4).

    The result is asserted to contain only {0, 1, 2, 4}; an internal 3 leaking
    into an exported mask is a correctness bug, not a cosmetic one.
    """
    pred_raw = pred_internal.copy()
    pred_raw[pred_internal == 3] = LABEL_MAP_INTERNAL_TO_RAW[3]

    observed = {int(v) for v in np.unique(pred_raw)}
    unexpected = observed - {0, 1, 2, 4}
    if unexpected:
        raise RuntimeError(
            f"Label restoration produced non-BraTS labels {sorted(unexpected)}. "
            f"Exported masks must contain only {{0, 1, 2, 4}}."
        )
    return pred_raw


def restore_prediction_to_original_space(
    pred_cropped: np.ndarray,
    original_shape: tuple[int, ...],
    crop_bbox: tuple[slice, slice, slice] | None,
) -> np.ndarray:
    """Embed a cropped prediction back into the original volume (notebook 21.1).

    This is the exact inverse of the foreground-bbox crop applied during
    preprocessing. Everything outside the crop is background (0).
    """
    if crop_bbox is None:
        if tuple(pred_cropped.shape) != tuple(original_shape):
            raise ValueError(
                f"Prediction shape {pred_cropped.shape} does not match the "
                f"original shape {tuple(original_shape)} and no crop box was "
                f"recorded — the inverse transform is undefined."
            )
        return pred_cropped
    full = np.zeros(original_shape, dtype=pred_cropped.dtype)
    full[crop_bbox] = pred_cropped
    return full
