"""Phase 6 — post-processing and label mapping.

The label-mapping tests are the important ones here: exporting internal label 3
instead of BraTS label 4 would be invisible in a viewer but would silently zero
out every ET metric downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.constants import BRATS_SEG_LABELS, to_brats_labels
from app.inference.brats.config import (
    LABEL_MAP_INTERNAL_TO_RAW,
    LABEL_MAP_RAW_TO_INTERNAL,
    RAW_LABELS,
)
from app.inference.brats.postprocess import (
    connected_component_filter,
    internal_to_brats_labels,
    restore_prediction_to_original_space,
)


# ── Label mapping ────────────────────────────────────────────────────────────
def test_label_maps_are_a_bijection_matching_the_notebook() -> None:
    assert RAW_LABELS == (0, 1, 2, 4)
    assert LABEL_MAP_RAW_TO_INTERNAL == {0: 0, 1: 1, 2: 2, 4: 3}
    assert LABEL_MAP_INTERNAL_TO_RAW == {0: 0, 1: 1, 2: 2, 3: 4}
    for raw in RAW_LABELS:
        assert LABEL_MAP_INTERNAL_TO_RAW[LABEL_MAP_RAW_TO_INTERNAL[raw]] == raw


def test_internal_three_becomes_brats_four() -> None:
    internal = np.array([0, 1, 2, 3, 3, 0], dtype=np.uint8)
    raw = internal_to_brats_labels(internal)
    assert raw.tolist() == [0, 1, 2, 4, 4, 0]
    assert 3 not in np.unique(raw)


def test_exported_labels_are_exactly_the_brats_set() -> None:
    internal = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    raw = internal_to_brats_labels(internal)
    assert set(np.unique(raw).tolist()).issubset(BRATS_SEG_LABELS)


def test_non_tumour_classes_are_untouched_by_the_mapping() -> None:
    rng = np.random.default_rng(0)
    internal = rng.integers(0, 4, size=(8, 8, 8)).astype(np.uint8)
    raw = internal_to_brats_labels(internal)
    for value in (0, 1, 2):
        assert np.array_equal(internal == value, raw == value)
    assert np.array_equal(internal == 3, raw == 4)


def test_mapping_agrees_with_the_shared_constants_helper() -> None:
    """RealBraTSInferenceService and MockInferenceService must agree."""
    rng = np.random.default_rng(1)
    internal = rng.integers(0, 4, size=(6, 6, 6)).astype(np.int16)
    assert np.array_equal(
        internal_to_brats_labels(internal.astype(np.uint8)),
        to_brats_labels(internal).astype(np.uint8),
    )


def test_unexpected_label_in_prediction_is_rejected() -> None:
    bad = np.array([0, 1, 5], dtype=np.uint8)
    with pytest.raises(RuntimeError, match="non-BraTS labels"):
        internal_to_brats_labels(bad)


# ── Connected-component filtering ────────────────────────────────────────────
def test_small_components_are_removed_and_large_ones_kept() -> None:
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 1  # 216 voxels, kept
    mask[15, 15, 15] = 1  # 1 voxel, removed

    cleaned, removed = connected_component_filter(mask, min_voxels=50)

    assert cleaned[15, 15, 15] == 0
    assert int((cleaned == 1).sum()) == 216
    assert removed[1] == 1


def test_filtering_is_applied_per_class_not_on_the_union() -> None:
    """Two small components of different classes must not merge into one big one."""
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1  # 27 voxels
    mask[5:8, 2:5, 2:5] = 2  # 27 voxels, adjacent to the first

    cleaned, removed = connected_component_filter(mask, min_voxels=50)

    assert int(cleaned.sum()) == 0
    assert removed[1] == 27
    assert removed[2] == 27


def test_background_is_never_removed() -> None:
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    cleaned, removed = connected_component_filter(mask, min_voxels=50)
    assert np.array_equal(cleaned, mask)
    assert removed == {1: 0, 2: 0, 3: 0}


def test_filter_never_introduces_new_labels() -> None:
    rng = np.random.default_rng(3)
    mask = rng.integers(0, 4, size=(16, 16, 16)).astype(np.uint8)
    cleaned, _ = connected_component_filter(mask, min_voxels=10)
    assert set(np.unique(cleaned).tolist()).issubset({0, 1, 2, 3})
    assert set(np.unique(cleaned).tolist()).issubset(set(np.unique(mask).tolist()))


def test_filter_only_ever_removes_voxels() -> None:
    rng = np.random.default_rng(4)
    mask = rng.integers(0, 4, size=(16, 16, 16)).astype(np.uint8)
    cleaned, _ = connected_component_filter(mask, min_voxels=25)
    assert np.all((cleaned == mask) | (cleaned == 0))


def test_removed_counts_match_the_actual_difference() -> None:
    rng = np.random.default_rng(5)
    mask = rng.integers(0, 4, size=(20, 20, 20)).astype(np.uint8)
    cleaned, removed = connected_component_filter(mask, min_voxels=30)
    for class_id in (1, 2, 3):
        before = int((mask == class_id).sum())
        after = int((cleaned == class_id).sum())
        assert removed[class_id] == before - after


def test_filter_does_not_mutate_its_input() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1, 1, 1] = 1
    original = mask.copy()
    connected_component_filter(mask, min_voxels=50)
    assert np.array_equal(mask, original)


# ── Inverse crop ─────────────────────────────────────────────────────────────
def test_uncrop_restores_the_original_geometry() -> None:
    original_shape = (20, 22, 18)
    bbox = (slice(3, 15), slice(4, 18), slice(2, 12))
    cropped = np.ones((12, 14, 10), dtype=np.uint8) * 2

    full = restore_prediction_to_original_space(cropped, original_shape, bbox)

    assert full.shape == original_shape
    assert np.array_equal(full[bbox], cropped)
    outside = full.copy()
    outside[bbox] = 0
    assert np.all(outside == 0)


def test_uncrop_is_a_noop_when_no_crop_was_applied() -> None:
    prediction = np.ones((6, 6, 6), dtype=np.uint8)
    result = restore_prediction_to_original_space(prediction, (6, 6, 6), None)
    assert result is prediction


def test_uncrop_rejects_an_undefined_inverse() -> None:
    prediction = np.ones((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="inverse transform is undefined"):
        restore_prediction_to_original_space(prediction, (8, 8, 8), None)


def test_uncrop_preserves_dtype_and_label_set() -> None:
    bbox = (slice(1, 4), slice(1, 4), slice(1, 4))
    cropped = np.array([0, 1, 2, 4] * 6 + [4, 4, 4], dtype=np.uint8).reshape(3, 3, 3)
    full = restore_prediction_to_original_space(cropped, (6, 6, 6), bbox)
    assert full.dtype == np.uint8
    assert set(np.unique(full).tolist()).issubset(BRATS_SEG_LABELS)
