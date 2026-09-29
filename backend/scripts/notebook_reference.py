"""Verbatim transcription of the training notebook's inference path.

This module exists for ONE purpose: to be the independent side of the
notebook-vs-production parity comparison in `notebook_parity_check.py`. It
imports nothing from `app.` on purpose — if it shared code with the production
implementation, the comparison would prove nothing.

Source cells in `notebook/notebookd7ed0558bb (2).ipynb`:

    Section  8.3   foreground_zscore_normalize, compute_foreground_bbox,
                   load_and_preprocess_case
    Section 12.2   ConvBlock3D, Down3D, Up3D
    Section 12.3   UNet3D
    Section 16.1   _compute_axis_starts, iter_sliding_window_coords,
                   pad_image_end_only, sliding_window_infer,
                   make_uniform_weight_kernel
    Section 19.1   make_gaussian_weight_kernel
    Section 19.2   infer_case_full
    Section 20.1   connected_component_filter
    Section 21.1   restore_prediction_to_original_space

Deliberate, minimal adaptations (each one is mechanical, none change a value):

  * `load_and_preprocess_case` takes a plain dict of modality paths instead of
    a pandas Series, so the script runs without pandas.
  * Notebook globals (`CONFIG`, `RAW_LABELS`, `LABEL_MAP_*`) are module
    constants here, set to the values the notebook actually held during the
    training run that produced the checkpoint.
  * `np.squeeze` is NOT applied, matching the notebook exactly. The production
    implementation does squeeze; parity fixtures are 3D so the two agree.

DO NOT import this from application code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch import nn

# ── Notebook globals (Sections 2.1, 8.2, 9.1) ────────────────────────────────
RAW_LABELS = (0, 1, 2, 4)
LABEL_MAP_RAW_TO_INTERNAL = {0: 0, 1: 1, 2: 2, 4: 3}
LABEL_MAP_INTERNAL_TO_RAW = {v: k for k, v in LABEL_MAP_RAW_TO_INTERNAL.items()}


@dataclass
class ModelConfig:
    in_channels: int = 4
    num_classes: int = 4
    num_stages: int = 5
    base_channels: int = 32
    max_channels: int = 320
    norm: str = "instance"
    activation: str = "leaky_relu"
    negative_slope: float = 0.01
    downsample_mode: str = "strided_conv"
    upsample_mode: str = "transposed_conv"


@dataclass
class PatchConfig:
    selected_size: tuple = (128, 128, 128)
    sliding_window_overlap: float = 0.5


@dataclass
class PreprocessingConfig:
    normalize_mode: str = "foreground_zscore_per_case"
    background_value: float = 0.0
    crop_to_foreground: bool = True
    crop_margin_voxels: int = 4
    canonical_orientation: tuple = ("L", "P", "S")
    orientation_uniform_across_dataset: bool = True
    sagittal_axis_index: int = 0


@dataclass
class ExperimentConfig:
    model: ModelConfig
    patch: PatchConfig
    preprocessing: PreprocessingConfig


def make_config(
    patch_size: tuple = (128, 128, 128), base_channels: int = 32
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(base_channels=base_channels),
        patch=PatchConfig(selected_size=tuple(patch_size)),
        preprocessing=PreprocessingConfig(),
    )


# ── Section 8.3 — preprocessing ──────────────────────────────────────────────
def foreground_zscore_normalize(volume, background_value=0.0, eps=1e-8):
    fg_mask = volume != background_value
    if not fg_mask.any():
        return np.zeros_like(volume, dtype=np.float32), 0.0, 1.0
    fg = volume[fg_mask]
    mean = float(fg.mean())
    std = max(float(fg.std()), eps)
    normalized = volume.astype(np.float32).copy()
    normalized[fg_mask] = (fg - mean) / std
    return normalized, mean, std


def compute_foreground_bbox(union_fg_mask, margin=4):
    coords = np.argwhere(union_fg_mask)
    if coords.size == 0:
        raise ValueError("Foreground mask is empty — cannot compute a bounding box.")
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + margin, union_fg_mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def load_and_preprocess_case(
    case_row: Mapping[str, object], config: ExperimentConfig, load_seg: bool = True
) -> dict:
    modality_order = ["t1", "t1ce", "t2", "flair"]
    raw_volumes, affine, original_shape = [], None, None
    for mod in modality_order:
        img = nib.load(str(case_row[f"{mod}_path"]))
        if config.preprocessing.orientation_uniform_across_dataset is False:
            img = nib.as_closest_canonical(img)
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        if affine is None:
            affine, original_shape = img.affine, data.shape
        raw_volumes.append(data)

    norm_stats, normalized = {}, []
    for mod, vol in zip(modality_order, raw_volumes):
        norm_vol, mean, std = foreground_zscore_normalize(
            vol, config.preprocessing.background_value
        )
        normalized.append(norm_vol)
        norm_stats[mod] = {"mean": mean, "std": std}
    image = np.stack(normalized, axis=0)

    seg_internal = None
    if load_seg and case_row.get("seg_path"):
        seg_img = nib.load(str(case_row["seg_path"]))
        if config.preprocessing.orientation_uniform_across_dataset is False:
            seg_img = nib.as_closest_canonical(seg_img)
        seg_raw = np.asanyarray(seg_img.dataobj).astype(np.uint8)
        unexpected = set(np.unique(seg_raw)) - set(RAW_LABELS)
        assert not unexpected, f"unexpected raw seg labels {unexpected}"
        seg_internal = np.zeros_like(seg_raw, dtype=np.int64)
        for raw, internal in LABEL_MAP_RAW_TO_INTERNAL.items():
            seg_internal[seg_raw == raw] = internal

    crop_bbox = None
    if config.preprocessing.crop_to_foreground:
        union_fg = np.any(
            np.stack(raw_volumes, axis=0) != config.preprocessing.background_value,
            axis=0,
        )
        if union_fg.any():
            crop_bbox = compute_foreground_bbox(
                union_fg, config.preprocessing.crop_margin_voxels
            )
            image = image[(slice(None),) + crop_bbox]
            if seg_internal is not None:
                seg_internal = seg_internal[crop_bbox]

    return {
        "image": image,
        "seg_internal": seg_internal,
        "original_shape": original_shape,
        "affine": affine,
        "crop_bbox": crop_bbox,
        "norm_stats": norm_stats,
    }


# ── Sections 12.2 / 12.3 — architecture ──────────────────────────────────────
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)


class Up3D(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        mode="transposed_conv",
        negative_slope=0.01,
    ):
        super().__init__()
        if mode == "transposed_conv":
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        elif mode == "trilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, in_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {mode}")
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels, negative_slope)

    def forward(self, x, skip):
        x = self.up(x)
        diff = [skip.shape[i] - x.shape[i] for i in (2, 3, 4)]
        if any(diff):
            x = F.pad(
                x,
                [
                    diff[2] // 2,
                    diff[2] - diff[2] // 2,
                    diff[1] // 2,
                    diff[1] - diff[1] // 2,
                    diff[0] // 2,
                    diff[0] - diff[0] // 2,
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3D(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = [
            min(config.base_channels * (2**i), config.max_channels)
            for i in range(config.num_stages)
        ]
        self.channels = channels

        self.encoder_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        in_ch = config.in_channels
        for i, ch in enumerate(channels):
            self.encoder_blocks.append(ConvBlock3D(in_ch, ch, config.negative_slope))
            if i < len(channels) - 1:
                self.downs.append(Down3D(ch))
            in_ch = ch

        self.decoder_blocks = nn.ModuleList(
            [
                Up3D(
                    channels[i],
                    channels[i - 1],
                    channels[i - 1],
                    config.upsample_mode,
                    config.negative_slope,
                )
                for i in range(len(channels) - 1, 0, -1)
            ]
        )
        self.final_conv = nn.Conv3d(channels[0], config.num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.downs[i](x)
        for dec_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = dec_block(x, skip)
        return self.final_conv(x)


# ── Sections 16.1 / 19.1 — sliding window ────────────────────────────────────
def make_uniform_weight_kernel(patch_size):
    return np.ones(patch_size, dtype=np.float32)


def make_gaussian_weight_kernel(patch_size, sigma_scale: float = 0.125) -> np.ndarray:
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


def _compute_axis_starts(dim_size, patch_size, stride):
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] != dim_size - patch_size:
        starts.append(dim_size - patch_size)
    return starts


def iter_sliding_window_coords(spatial_shape, patch_size, overlap):
    stride = tuple(max(1, int(round(p * (1 - overlap)))) for p in patch_size)
    axis_starts = [
        _compute_axis_starts(d, p, s)
        for d, p, s in zip(spatial_shape, patch_size, stride)
    ]
    for starts in itertools.product(*axis_starts):
        yield tuple(slice(s, s + p) for s, p in zip(starts, patch_size))


def pad_image_end_only(image: np.ndarray, min_size):
    pads = [(0, max(m - dim, 0)) for dim, m in zip(image.shape[1:], min_size)]
    if all(p == (0, 0) for p in pads):
        return image
    return np.pad(image, [(0, 0)] + pads, mode="constant", constant_values=0.0)


def sliding_window_infer(
    image, model, config: ExperimentConfig, device, weight_kernel_fn=make_uniform_weight_kernel
):
    patch_size = config.patch.selected_size
    overlap = config.patch.sliding_window_overlap
    num_classes = config.model.num_classes
    original_shape = image.shape[1:]

    padded_image = pad_image_end_only(image, patch_size)
    padded_shape = padded_image.shape[1:]
    weight_kernel = weight_kernel_fn(patch_size)
    prob_map = np.zeros((num_classes, *padded_shape), dtype=np.float32)
    weight_map = np.zeros(padded_shape, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for coords in iter_sliding_window_coords(padded_shape, patch_size, overlap):
            patch_t = (
                torch.from_numpy(padded_image[(slice(None),) + coords])
                .float()
                .unsqueeze(0)
                .to(device)
            )
            probs = torch.softmax(model(patch_t), dim=1).squeeze(0).cpu().numpy()
            prob_map[(slice(None),) + coords] += probs * weight_kernel
            weight_map[coords] += weight_kernel

    assert (weight_map > 0).all(), "Sliding-window tiling left a zero-coverage voxel."
    prob_map /= np.clip(weight_map, 1e-8, None)
    crop = tuple(slice(0, d) for d in original_shape)
    return prob_map[(slice(None),) + crop]


# ── Section 20.1 — post-processing ───────────────────────────────────────────
def connected_component_filter(pred_internal: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    cleaned = pred_internal.copy()
    for class_id in (1, 2, 3):
        binary_mask = pred_internal == class_id
        if not binary_mask.any():
            continue
        labeled_array, num_features = ndi.label(binary_mask)
        if num_features == 0:
            continue
        component_sizes = np.bincount(labeled_array.ravel())
        remove_mask = (labeled_array != 0) & (component_sizes[labeled_array] < min_voxels)
        cleaned[remove_mask] = 0
    return cleaned


# ── Section 21.1 — inverse crop ──────────────────────────────────────────────
def restore_prediction_to_original_space(pred_cropped, original_shape, crop_bbox):
    if crop_bbox is None:
        return pred_cropped
    full = np.zeros(original_shape, dtype=pred_cropped.dtype)
    full[crop_bbox] = pred_cropped
    return full


# ── Section 19.2 — end-to-end per-case inference ─────────────────────────────
def infer_case_full(case_row, model, config: ExperimentConfig, device) -> dict:
    preprocessed = load_and_preprocess_case(case_row, config)
    prob_map = sliding_window_infer(
        preprocessed["image"],
        model,
        config,
        device,
        weight_kernel_fn=make_gaussian_weight_kernel,
    )
    pred_internal = np.argmax(prob_map, axis=0).astype(np.uint8)

    pred_raw = pred_internal.copy()
    pred_raw[pred_internal == 3] = 4
    assert 3 not in np.unique(pred_raw)
    assert set(np.unique(pred_raw)).issubset(set(RAW_LABELS))

    return {
        "prob_map": prob_map,
        "pred_internal": pred_internal,
        "pred_raw": pred_raw,
        "seg_internal": preprocessed["seg_internal"],
        "affine": preprocessed["affine"],
        "original_shape": preprocessed["original_shape"],
        "crop_bbox": preprocessed["crop_bbox"],
        "norm_stats": preprocessed["norm_stats"],
        "image": preprocessed["image"],
    }
