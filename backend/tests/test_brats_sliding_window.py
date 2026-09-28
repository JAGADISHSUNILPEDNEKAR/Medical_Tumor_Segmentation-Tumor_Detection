"""Phase 6 — sliding-window tiling and Gaussian stitching.

The cases the master prompt calls out explicitly are each covered: a single
patch, overlapping patches, edge patches, a volume smaller than the patch, and
a volume exactly equal to the patch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.inference.brats.model import build_model  # noqa: E402
from app.inference.brats.sliding_window import (  # noqa: E402
    _compute_axis_starts,
    iter_sliding_window_coords,
    make_gaussian_weight_kernel,
    make_uniform_weight_kernel,
    pad_image_end_only,
    sliding_window_infer,
)
from tests.phase6_helpers import small_config  # noqa: E402

DEVICE = torch.device("cpu")
PATCH = (32, 32, 32)


# ── Gaussian kernel (notebook 19.1 assertions) ───────────────────────────────
def test_gaussian_kernel_is_positive_centred_and_symmetric() -> None:
    kernel = make_gaussian_weight_kernel((8, 8, 8))
    assert kernel.min() > 0
    assert kernel[4, 4, 4] == kernel.max()
    assert np.allclose(kernel[0, :, :], kernel[-1, :, :])
    assert np.allclose(kernel[:, 0, :], kernel[:, -1, :])
    assert kernel.dtype == np.float32


def test_gaussian_kernel_decays_from_centre_to_edge() -> None:
    kernel = make_gaussian_weight_kernel(PATCH)
    centre = kernel[16, 16, 16]
    edge = kernel[0, 16, 16]
    assert centre > edge
    assert edge >= 1e-4  # clipped floor, never exactly zero


def test_gaussian_kernel_sigma_scale_matches_notebook_formula() -> None:
    dim = 16
    sigma_scale = 0.125
    kernel = make_gaussian_weight_kernel((dim, 1, 1), sigma_scale=sigma_scale)
    sigma = sigma_scale * dim
    centre = (dim - 1) / 2.0
    expected = np.exp(-0.5 * ((np.arange(dim) - centre) / sigma) ** 2)
    expected = np.clip(expected, 1e-4, None)
    assert np.allclose(kernel[:, 0, 0], expected, atol=1e-6)


# ── Tiling ───────────────────────────────────────────────────────────────────
def test_axis_starts_for_volume_smaller_than_patch() -> None:
    assert _compute_axis_starts(20, 32, 16) == [0]


def test_axis_starts_for_volume_equal_to_patch() -> None:
    assert _compute_axis_starts(32, 32, 16) == [0]


def test_axis_starts_flush_the_final_tile_to_the_edge() -> None:
    """A trailing tile is added so the far edge is never left uncovered."""
    starts = _compute_axis_starts(100, 32, 16)
    assert starts[0] == 0
    assert starts[-1] == 100 - 32
    assert all(s % 16 == 0 for s in starts[:-1])


def test_stride_is_half_the_patch_at_overlap_one_half() -> None:
    coords = list(iter_sliding_window_coords((64, 32, 32), PATCH, 0.5))
    starts = sorted({c[0].start for c in coords})
    assert starts == [0, 16, 32]


def test_tiling_covers_every_voxel() -> None:
    shape = (70, 45, 33)
    coverage = np.zeros(shape, dtype=np.int32)
    padded = tuple(max(s, p) for s, p in zip(shape, PATCH))
    coverage = np.zeros(padded, dtype=np.int32)
    for coords in iter_sliding_window_coords(padded, PATCH, 0.5):
        coverage[coords] += 1
    assert coverage.min() >= 1


# ── Padding ──────────────────────────────────────────────────────────────────
def test_end_only_padding_is_reversible_by_a_plain_crop() -> None:
    image = np.arange(4 * 5 * 6 * 7, dtype=np.float32).reshape(4, 5, 6, 7)
    padded = pad_image_end_only(image, (8, 8, 8))
    assert padded.shape == (4, 8, 8, 8)
    assert np.array_equal(padded[:, :5, :6, :7], image)
    assert np.all(padded[:, 5:, :, :] == 0.0)


def test_padding_is_a_noop_when_already_large_enough() -> None:
    image = np.ones((4, 40, 40, 40), dtype=np.float32)
    assert pad_image_end_only(image, PATCH) is image


# ── Stitching ────────────────────────────────────────────────────────────────
def _constant_logit_model(num_classes: int = 4):
    """A stand-in that returns the same logits everywhere.

    With a position-independent model, correct Gaussian normalization must
    reproduce that constant exactly at every voxel — any seam or weighting bug
    shows up immediately as a deviation.
    """

    class Constant(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(
                (x.shape[0], num_classes, *x.shape[2:]), dtype=torch.float32
            )
            out[:, 1] = 2.0
            return out

    return Constant().eval()


@pytest.mark.parametrize(
    "shape,label",
    [
        ((32, 32, 32), "volume exactly equal to patch"),
        ((20, 25, 18), "volume smaller than patch"),
        ((48, 32, 32), "two overlapping patches on one axis"),
        ((70, 45, 33), "edge patches on all three axes"),
    ],
)
def test_gaussian_stitching_is_seamless(shape, label) -> None:
    config = small_config(patch_size=PATCH)
    model = _constant_logit_model()
    image = np.random.default_rng(0).normal(size=(4, *shape)).astype(np.float32)

    prob_map, stats = sliding_window_infer(
        image, model, config.model, config.patch, DEVICE
    )

    assert prob_map.shape == (4, *shape), label
    expected = float(torch.softmax(torch.tensor([0.0, 2.0, 0.0, 0.0]), dim=0)[1])
    assert np.allclose(prob_map[1], expected, atol=1e-6), label
    # No seam: the spread across the whole volume is numerical noise only.
    assert float(prob_map[1].max() - prob_map[1].min()) < 1e-6, label
    assert stats.num_patches >= 1


def test_single_patch_volume_evaluates_exactly_one_patch() -> None:
    config = small_config(patch_size=PATCH)
    model = _constant_logit_model()
    image = np.zeros((4, 32, 32, 32), dtype=np.float32)

    _, stats = sliding_window_infer(image, model, config.model, config.patch, DEVICE)

    assert stats.num_patches == 1
    assert stats.stride == (16, 16, 16)


def test_overlapping_volume_evaluates_multiple_patches() -> None:
    config = small_config(patch_size=PATCH)
    model = _constant_logit_model()
    image = np.zeros((4, 64, 32, 32), dtype=np.float32)

    _, stats = sliding_window_infer(image, model, config.model, config.patch, DEVICE)

    assert stats.num_patches == 3


def test_output_is_a_probability_distribution_over_classes() -> None:
    config = small_config(patch_size=PATCH)
    model = build_model(config.model).eval()
    image = np.random.default_rng(1).normal(size=(4, 40, 36, 34)).astype(np.float32)

    prob_map, _ = sliding_window_infer(
        image, model, config.model, config.patch, DEVICE
    )

    assert prob_map.shape == (4, 40, 36, 34)
    assert prob_map.min() >= 0.0
    assert prob_map.max() <= 1.0 + 1e-6
    assert np.allclose(prob_map.sum(axis=0), 1.0, atol=1e-5)


def test_argmax_labels_are_within_the_class_range() -> None:
    config = small_config(patch_size=PATCH)
    model = build_model(config.model).eval()
    image = np.random.default_rng(2).normal(size=(4, 36, 36, 36)).astype(np.float32)

    prob_map, _ = sliding_window_infer(
        image, model, config.model, config.patch, DEVICE
    )
    labels = np.argmax(prob_map, axis=0)

    assert set(np.unique(labels).tolist()).issubset({0, 1, 2, 3})


def test_inference_is_deterministic() -> None:
    config = small_config(patch_size=PATCH)
    torch.manual_seed(11)
    model = build_model(config.model).eval()
    image = np.random.default_rng(3).normal(size=(4, 40, 36, 34)).astype(np.float32)

    first, _ = sliding_window_infer(image, model, config.model, config.patch, DEVICE)
    second, _ = sliding_window_infer(image, model, config.model, config.patch, DEVICE)

    assert np.array_equal(first, second)


def test_uniform_kernel_remains_available_for_diagnostics() -> None:
    kernel = make_uniform_weight_kernel((4, 4, 4))
    assert np.all(kernel == 1.0)
    assert kernel.dtype == np.float32


def test_inference_does_not_retain_autograd_graph() -> None:
    config = small_config(patch_size=PATCH)
    model = build_model(config.model).eval()
    image = np.zeros((4, 32, 32, 32), dtype=np.float32)

    sliding_window_infer(image, model, config.model, config.patch, DEVICE)

    for parameter in model.parameters():
        assert parameter.grad is None
