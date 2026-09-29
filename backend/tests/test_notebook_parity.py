"""Phase 6 — notebook vs production parity, as a regression test.

`scripts/notebook_reference.py` is a verbatim transcription of the notebook's
inference path that shares no code with the application. Running both over the
same input with the same weights and comparing every stage is what turns
"we reimplemented it carefully" into a checkable claim.

The full-scale version of this comparison (240x240x155 volume, 128^3 patches)
lives in `scripts/notebook_parity_check.py --full`; this module runs the same
comparison at a reduced width so it can sit in the ordinary test suite.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import notebook_reference as ref  # noqa: E402

from app.inference.brats.config import TRAINING_CONFIG  # noqa: E402
from app.inference.brats.model import build_model  # noqa: E402
from app.inference.brats.postprocess import (  # noqa: E402
    connected_component_filter,
    internal_to_brats_labels,
    restore_prediction_to_original_space,
)
from app.inference.brats.preprocessing import load_and_preprocess_case  # noqa: E402
from app.inference.brats.sliding_window import (  # noqa: E402
    make_gaussian_weight_kernel,
    sliding_window_infer,
)
from tests.phase6_helpers import write_case  # noqa: E402

DEVICE = torch.device("cpu")
PATCH = (32, 32, 32)
BASE_CHANNELS = 4
SHAPE = (60, 64, 56)


@pytest.fixture(scope="module")
def parity_run(tmp_path_factory) -> dict:
    """Run both implementations once; the tests below assert on the outputs."""
    tmp_path = tmp_path_factory.mktemp("parity")
    modality_paths = write_case(tmp_path / "case", shape=SHAPE)

    prod_config = replace(
        TRAINING_CONFIG,
        model=replace(TRAINING_CONFIG.model, base_channels=BASE_CHANNELS),
        patch=replace(TRAINING_CONFIG.patch, selected_size=PATCH),
    )
    ref_config = ref.make_config(patch_size=PATCH, base_channels=BASE_CHANNELS)

    torch.manual_seed(4321)
    state_dict = build_model(prod_config.model).state_dict()

    prod_model = build_model(prod_config.model)
    prod_model.load_state_dict(state_dict, strict=True)
    prod_model.to(DEVICE).eval()

    ref_model = ref.UNet3D(ref_config.model)
    ref_model.load_state_dict(state_dict, strict=True)
    ref_model.to(DEVICE).eval()

    ref_row = {f"{m}_path": str(p) for m, p in modality_paths.items()}
    ref_row["seg_path"] = None
    ref_out = ref.infer_case_full(ref_row, ref_model, ref_config, DEVICE)
    ref_clean = ref.connected_component_filter(ref_out["pred_internal"], min_voxels=50)
    ref_raw = ref_clean.copy()
    ref_raw[ref_clean == 3] = 4
    ref_full = ref.restore_prediction_to_original_space(
        ref_raw, ref_out["original_shape"], ref_out["crop_bbox"]
    )

    prod_case = load_and_preprocess_case(modality_paths, prod_config.preprocessing)
    prod_prob, stats = sliding_window_infer(
        prod_case.image,
        prod_model,
        prod_config.model,
        prod_config.patch,
        DEVICE,
        weight_kernel_fn=lambda size: make_gaussian_weight_kernel(
            size, prod_config.patch.gaussian_sigma_scale
        ),
    )
    prod_internal = np.argmax(prod_prob, axis=0).astype(np.uint8)
    prod_clean, _ = connected_component_filter(prod_internal, min_voxels=50)
    prod_full = restore_prediction_to_original_space(
        internal_to_brats_labels(prod_clean),
        prod_case.original_shape,
        prod_case.crop_bbox,
    )

    return {
        "ref": ref_out,
        "ref_clean": ref_clean,
        "ref_full": ref_full,
        "prod_case": prod_case,
        "prod_prob": prod_prob,
        "prod_internal": prod_internal,
        "prod_clean": prod_clean,
        "prod_full": prod_full,
        "stats": stats,
    }


def test_input_shape_matches(parity_run) -> None:
    assert parity_run["ref"]["original_shape"] == parity_run["prod_case"].original_shape


def test_crop_bbox_matches(parity_run) -> None:
    ref_bbox = [(s.start, s.stop) for s in parity_run["ref"]["crop_bbox"]]
    prod_bbox = [(s.start, s.stop) for s in parity_run["prod_case"].crop_bbox]
    assert ref_bbox == prod_bbox


def test_preprocessed_volume_is_bit_identical(parity_run) -> None:
    assert np.array_equal(parity_run["ref"]["image"], parity_run["prod_case"].image)


def test_normalization_statistics_match_exactly(parity_run) -> None:
    for modality in ("t1", "t1ce", "t2", "flair"):
        ref_stats = parity_run["ref"]["norm_stats"][modality]
        prod_stats = parity_run["prod_case"].norm_stats[modality]
        assert ref_stats["mean"] == prod_stats["mean"]
        assert ref_stats["std"] == prod_stats["std"]


def test_model_output_shape_matches(parity_run) -> None:
    assert parity_run["ref"]["prob_map"].shape == parity_run["prod_prob"].shape


def test_probability_maps_agree_within_float32_tolerance(parity_run) -> None:
    """Tolerance, not exact equality: float32 accumulation order may differ."""
    max_abs = float(
        np.max(np.abs(parity_run["ref"]["prob_map"] - parity_run["prod_prob"]))
    )
    assert max_abs <= 1e-6, f"max |Δ| = {max_abs:.3e}"


def test_predicted_labels_are_identical(parity_run) -> None:
    assert np.array_equal(parity_run["ref"]["pred_internal"], parity_run["prod_internal"])


def test_post_processed_labels_are_identical(parity_run) -> None:
    assert np.array_equal(parity_run["ref_clean"], parity_run["prod_clean"])


def test_exported_brats_masks_are_identical(parity_run) -> None:
    assert np.array_equal(parity_run["ref_full"], parity_run["prod_full"])


def test_foreground_voxel_counts_match(parity_run) -> None:
    assert int(np.count_nonzero(parity_run["ref_full"])) == int(
        np.count_nonzero(parity_run["prod_full"])
    )


def test_per_region_voxel_counts_match(parity_run) -> None:
    for label in (1, 2, 4):
        assert int((parity_run["ref_full"] == label).sum()) == int(
            (parity_run["prod_full"] == label).sum()
        )


def test_output_geometry_matches_the_input(parity_run) -> None:
    assert parity_run["prod_full"].shape == tuple(parity_run["ref"]["original_shape"])
    assert np.array_equal(parity_run["ref"]["affine"], parity_run["prod_case"].affine)


def test_exported_labels_are_the_brats_set(parity_run) -> None:
    labels = set(int(v) for v in np.unique(parity_run["prod_full"]))
    assert labels.issubset({0, 1, 2, 4})


def test_sliding_window_stride_is_half_the_patch(parity_run) -> None:
    assert parity_run["stats"].stride == (16, 16, 16)
    assert parity_run["stats"].num_patches > 1, "parity must cover overlapping tiles"
