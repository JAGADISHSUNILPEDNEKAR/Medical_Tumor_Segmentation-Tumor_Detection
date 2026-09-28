"""Phase 6 — preprocessing parity.

Asserts the properties the notebook asserted in its own Section 8.5 / Section 16
sanity checks: foreground mean 0 / std 1 per modality, background pinned at
exactly 0, a crop that is strictly smaller than the input, and — the one that
silently ruins a model if it is wrong — channel ordering.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("torch")

from app.inference.brats.config import MODALITY_ORDER, TRAINING_CONFIG  # noqa: E402
from app.inference.brats.preprocessing import (  # noqa: E402
    compute_foreground_bbox,
    foreground_zscore_normalize,
    load_and_preprocess_case,
)
from tests.phase6_helpers import write_case  # noqa: E402

PREP = TRAINING_CONFIG.preprocessing


def test_channel_order_is_t1_t1ce_t2_flair_not_alphabetical() -> None:
    assert MODALITY_ORDER == ("t1", "t1ce", "t2", "flair")
    assert MODALITY_ORDER != tuple(sorted(MODALITY_ORDER))


def test_channels_are_stacked_in_notebook_order(tmp_path: Path) -> None:
    """Each modality is written with a distinct intensity scale, so a permuted
    stack would be detectable by the per-channel foreground statistics."""
    shape = (40, 42, 38)
    paths = write_case(tmp_path / "case", shape=shape)

    raw_means = {}
    for modality, path in paths.items():
        data = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
        raw_means[modality] = float(data[data != 0].mean())

    # Passing the mapping in reverse order must not change the stacking order.
    shuffled = {m: paths[m] for m in reversed(MODALITY_ORDER)}
    case = load_and_preprocess_case(shuffled, PREP)

    assert case.image.shape[0] == 4
    for index, modality in enumerate(MODALITY_ORDER):
        assert case.norm_stats[modality]["mean"] == pytest.approx(
            raw_means[modality], rel=1e-6
        )
        channel = case.image[index]
        foreground = channel[channel != 0.0]
        assert foreground.mean() == pytest.approx(0.0, abs=1e-4)


def test_missing_modality_is_rejected(tmp_path: Path) -> None:
    paths = write_case(tmp_path / "case", modalities=("t1", "t1ce", "t2"))
    with pytest.raises(ValueError, match="Missing modality"):
        load_and_preprocess_case(paths, PREP)


def test_mismatched_shapes_are_rejected(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    paths = write_case(case_dir, shape=(40, 42, 38))
    odd = np.zeros((40, 42, 20), dtype=np.float32)
    odd[5:35, 5:35, 5:15] = 100.0
    nib.save(nib.Nifti1Image(odd, np.eye(4)), str(case_dir / "t2.nii.gz"))

    with pytest.raises(ValueError, match="expected"):
        load_and_preprocess_case(paths, PREP)


def test_foreground_zscore_leaves_background_at_exactly_zero() -> None:
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:8, 2:8, 2:8] = np.linspace(100.0, 900.0, 6**3).reshape(6, 6, 6)

    normalized, mean, std = foreground_zscore_normalize(volume, 0.0)

    background = normalized[volume == 0.0]
    assert np.all(background == 0.0)
    foreground = normalized[volume != 0.0]
    assert foreground.mean() == pytest.approx(0.0, abs=1e-5)
    assert foreground.std() == pytest.approx(1.0, abs=1e-5)
    assert mean == pytest.approx(float(volume[volume != 0].mean()))
    assert std == pytest.approx(float(volume[volume != 0].std()))


def test_foreground_zscore_on_all_zero_volume_is_safe() -> None:
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    normalized, mean, std = foreground_zscore_normalize(volume, 0.0)
    assert np.all(normalized == 0.0)
    assert (mean, std) == (0.0, 1.0)


def test_normalization_is_per_case_not_global(tmp_path: Path) -> None:
    """Two cases at different intensity scales must both land at mean 0 / std 1."""
    dim = (36, 36, 34)
    quiet = write_case(tmp_path / "quiet", shape=dim)
    loud_dir = tmp_path / "loud"
    loud_dir.mkdir(parents=True)
    loud: dict[str, Path] = {}
    for modality, path in quiet.items():
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj, dtype=np.float32) * 37.0
        out = loud_dir / f"{modality}.nii.gz"
        nib.save(nib.Nifti1Image(data, image.affine), str(out))
        loud[modality] = out

    for paths in (quiet, loud):
        case = load_and_preprocess_case(paths, PREP)
        for index in range(4):
            channel = case.image[index]
            foreground = channel[channel != 0.0]
            assert foreground.mean() == pytest.approx(0.0, abs=1e-4)
            assert foreground.std() == pytest.approx(1.0, abs=1e-4)


def test_foreground_bbox_applies_the_configured_margin() -> None:
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[8:12, 9:13, 7:11] = True

    bbox = compute_foreground_bbox(mask, margin=4)

    assert [(s.start, s.stop) for s in bbox] == [(4, 16), (5, 17), (3, 15)]


def test_foreground_bbox_is_clipped_at_the_volume_edges() -> None:
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:3, 7:10, 0:10] = True

    bbox = compute_foreground_bbox(mask, margin=4)

    assert [(s.start, s.stop) for s in bbox] == [(0, 7), (3, 10), (0, 10)]


def test_empty_foreground_bbox_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_foreground_bbox(np.zeros((5, 5, 5), dtype=bool), margin=4)


def test_crop_is_smaller_than_input_and_recorded(tmp_path: Path) -> None:
    shape = (48, 50, 44)
    paths = write_case(tmp_path / "case", shape=shape)

    case = load_and_preprocess_case(paths, PREP)

    assert case.original_shape == shape
    assert case.crop_bbox is not None
    assert tuple(case.image.shape[1:]) < shape
    for axis, sl in enumerate(case.crop_bbox):
        assert 0 <= sl.start < sl.stop <= shape[axis]
    assert tuple(sl.stop - sl.start for sl in case.crop_bbox) == tuple(
        case.image.shape[1:]
    )


def test_geometry_is_preserved_exactly(tmp_path: Path) -> None:
    """No resampling, no reorientation — the notebook measured neither as needed."""
    shape = (40, 44, 36)
    spacing = (1.0, 1.0, 1.0)
    paths = write_case(tmp_path / "case", shape=shape, spacing=spacing)
    reference = nib.load(str(paths["t1"]))

    case = load_and_preprocess_case(paths, PREP)

    assert np.array_equal(case.affine, reference.affine)
    assert case.spacing == spacing
    assert case.original_shape == shape


def test_preprocessing_is_deterministic(tmp_path: Path) -> None:
    paths = write_case(tmp_path / "case", shape=(36, 38, 34))

    first = load_and_preprocess_case(paths, PREP)
    second = load_and_preprocess_case(paths, PREP)

    assert np.array_equal(first.image, second.image)
    assert first.crop_bbox == second.crop_bbox
    assert first.norm_stats == second.norm_stats


def test_output_dtype_is_float32(tmp_path: Path) -> None:
    paths = write_case(tmp_path / "case", shape=(36, 38, 34))
    case = load_and_preprocess_case(paths, PREP)
    assert case.image.dtype == np.float32
