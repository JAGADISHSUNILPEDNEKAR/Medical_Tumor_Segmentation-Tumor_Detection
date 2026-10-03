"""Gaussian-weighted sliding-window inference (notebook Sections 16.1 + 19.1).

The notebook defines the tiling and accumulation once (Section 16.1) with a
pluggable `weight_kernel_fn`, and Stage G passes the Gaussian kernel from
Section 19.1 without touching the tiling logic. Both are reproduced here.

Geometry, taken from the notebook and not assumed:
  * patch size 128 x 128 x 128
  * overlap 0.5, so stride = round(128 * (1 - 0.5)) = 64 per axis
  * a volume shorter than the patch on an axis yields a single tile at 0
  * the last tile on each axis is flushed to the far edge, so there is never
    a coverage gap
  * padding is applied at the END of each axis only, which makes it reversible
    by a plain crop back to the original shape
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from app.inference.brats.config import BratsModelConfig, BratsPatchConfig

WeightKernelFn = Callable[[tuple[int, int, int]], np.ndarray]


def make_uniform_weight_kernel(patch_size: tuple[int, int, int]) -> np.ndarray:
    """Flat weighting. Retained for tests that isolate tiling from blending."""
    return np.ones(patch_size, dtype=np.float32)


def make_gaussian_weight_kernel(
    patch_size: tuple[int, int, int], sigma_scale: float = 0.125
) -> np.ndarray:
    """Separable Gaussian peaked at the patch centre (notebook 19.1).

    Centre predictions outweigh edge predictions, which is what removes seams
    at tile boundaries. The kernel is clipped to a small positive floor so the
    accumulated weight map can never be exactly zero.
    """
    kernel = np.ones(patch_size, dtype=np.float32)
    for axis, dim in enumerate(patch_size):
        sigma = sigma_scale * dim
        center = (dim - 1) / 2.0
        coords = np.arange(dim, dtype=np.float32)
        gauss_1d = np.exp(-0.5 * ((coords - center) / sigma) ** 2)
        shape = [1] * len(patch_size)
        shape[axis] = dim
        kernel *= gauss_1d.reshape(shape)
    kernel = np.clip(kernel, a_min=1e-4, a_max=None)
    return kernel.astype(np.float32)


def _compute_axis_starts(dim_size: int, patch_size: int, stride: int) -> list[int]:
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] != dim_size - patch_size:
        starts.append(dim_size - patch_size)
    return starts


def iter_sliding_window_coords(
    spatial_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    overlap: float,
) -> Iterator[tuple[slice, slice, slice]]:
    stride = tuple(max(1, int(round(p * (1 - overlap)))) for p in patch_size)
    axis_starts = [
        _compute_axis_starts(d, p, s)
        for d, p, s in zip(spatial_shape, patch_size, stride)
    ]
    for starts in itertools.product(*axis_starts):
        yield tuple(slice(s, s + p) for s, p in zip(starts, patch_size))  # type: ignore[misc]


def pad_image_end_only(
    image: np.ndarray, min_size: tuple[int, int, int]
) -> np.ndarray:
    """Zero-pad the END of each spatial axis up to `min_size`.

    End-only (rather than symmetric) padding is what makes the padding
    trivially reversible by cropping back to [0:D, 0:H, 0:W] afterwards.
    """
    pads = [(0, max(m - dim, 0)) for dim, m in zip(image.shape[1:], min_size)]
    if all(p == (0, 0) for p in pads):
        return image
    return np.pad(image, [(0, 0)] + pads, mode="constant", constant_values=0.0)


@dataclass(frozen=True)
class SlidingWindowStats:
    """Measured facts about one sliding-window pass."""

    num_patches: int
    padded_shape: tuple[int, ...]
    stride: tuple[int, int, int]


def sliding_window_infer(
    image: np.ndarray,
    model: nn.Module,
    model_config: BratsModelConfig,
    patch_config: BratsPatchConfig,
    device: torch.device,
    weight_kernel_fn: WeightKernelFn | None = None,
) -> tuple[np.ndarray, SlidingWindowStats]:
    """Run tiled inference over a preprocessed volume.

    Args:
        image: (C_in, D, H, W) float32, already preprocessed.

    Returns:
        (prob_map, stats) where prob_map is (num_classes, D, H, W) at the
        ORIGINAL (unpadded) spatial shape. The caller applies argmax.
    """
    patch_size = patch_config.selected_size
    overlap = patch_config.sliding_window_overlap
    num_classes = model_config.num_classes
    original_shape = image.shape[1:]

    if weight_kernel_fn is None:
        def weight_kernel_fn(size: tuple[int, int, int]) -> np.ndarray:
            return make_gaussian_weight_kernel(size, patch_config.gaussian_sigma_scale)

    padded_image = pad_image_end_only(image, patch_size)
    padded_shape = padded_image.shape[1:]
    weight_kernel = weight_kernel_fn(patch_size)
    prob_map = np.zeros((num_classes, *padded_shape), dtype=np.float32)
    weight_map = np.zeros(padded_shape, dtype=np.float32)

    stride = tuple(max(1, int(round(p * (1 - overlap)))) for p in patch_size)
    num_patches = 0

    model.eval()
    with torch.inference_mode():
        for coords in iter_sliding_window_coords(padded_shape, patch_size, overlap):
            patch = np.ascontiguousarray(padded_image[(slice(None),) + coords])
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            logits = model(patch_t)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            prob_map[(slice(None),) + coords] += probs * weight_kernel
            weight_map[coords] += weight_kernel
            num_patches += 1
            # Drop device tensors before the next tile so peak memory stays at
            # one patch, not one patch per tile.
            del patch_t, logits, probs
            
            if device.type == "cpu":
                gc.collect()

    if not (weight_map > 0).all():
        raise RuntimeError(
            "Sliding-window tiling left a zero-coverage voxel — stride/edge bug."
        )
    prob_map /= np.clip(weight_map, 1e-8, None)
    crop = tuple(slice(0, d) for d in original_shape)
    stats = SlidingWindowStats(
        num_patches=num_patches,
        padded_shape=tuple(int(s) for s in padded_shape),
        stride=stride,  # type: ignore[arg-type]
    )
    return prob_map[(slice(None),) + crop], stats
