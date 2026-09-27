"""Shared Phase 6 fixtures and helpers.

Imported by the Phase 6 test modules. Everything here builds SYNTHETIC data:
random weights and geometric phantom volumes. No test in this suite makes a
claim about segmentation quality, and no number produced here is a metric.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 6 requires PyTorch")

from app.inference.brats.config import (  # noqa: E402
    TRAINING_CONFIG,
    BratsInferenceConfig,
    checkpoint_compat_fingerprint,
)
from app.inference.brats.model import build_model  # noqa: E402

MODALITIES = ("t1", "t1ce", "t2", "flair")

# A deliberately small architecture/patch pairing so the whole suite runs on a
# laptop CPU in seconds. Every code path exercised is the production one; only
# the width and the patch size shrink.
#
# 32 is the SMALLEST legal patch size for this architecture, not an arbitrary
# choice: 5 stages means 4 strided-conv halvings, and InstanceNorm3d cannot
# normalize a bottleneck that has collapsed to a single spatial element. The
# trained patch size of 128 clears this by a factor of 4.
TEST_BASE_CHANNELS = 4
TEST_PATCH_SIZE = (32, 32, 32)
MIN_PATCH_SIZE_PER_AXIS = 32


def small_config(
    *,
    base_channels: int = TEST_BASE_CHANNELS,
    patch_size: tuple[int, int, int] = TEST_PATCH_SIZE,
    cc_min_voxels: int | None = None,
) -> BratsInferenceConfig:
    config = replace(
        TRAINING_CONFIG,
        model=replace(TRAINING_CONFIG.model, base_channels=base_channels),
        patch=replace(TRAINING_CONFIG.patch, selected_size=tuple(patch_size)),
    )
    if cc_min_voxels is not None:
        config = replace(
            config,
            postprocessing=replace(
                config.postprocessing, connected_component_min_voxels=cc_min_voxels
            ),
        )
    return config


def write_synthetic_checkpoint(
    path: Path,
    config: BratsInferenceConfig,
    *,
    seed: int = 0,
    epoch: int = 3,
    fingerprint: str | None = None,
    include_fingerprint: bool = True,
    bare_state_dict: bool = False,
) -> Path:
    """Write a checkpoint in the notebook's save format. Weights are random."""
    torch.manual_seed(seed)
    model = build_model(config.model)
    if bare_state_dict:
        torch.save(model.state_dict(), path)
        return path

    payload: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "epoch": epoch,
        "global_step": epoch * 100,
        "best_val_metric": None,
        "saved_at": "2026-09-15T06:16:56",
        "synthetic_fixture": True,
    }
    if include_fingerprint:
        payload["compat_fingerprint"] = fingerprint or checkpoint_compat_fingerprint(
            config
        )
    torch.save(payload, path)
    return path


def phantom_volume(
    shape: tuple[int, int, int], *, scale: float, seed: int
) -> np.ndarray:
    """Geometric phantom: an ellipsoid of signal inside a zero background.

    The zero rim matters — it is what gives the foreground-bbox crop something
    to crop, and what makes the foreground z-score a non-trivial operation.
    """
    rng = np.random.default_rng(seed)
    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float32)
    center = np.array([s / 2.0 for s in shape], dtype=np.float32)
    radius = np.array([s * 0.35 for s in shape], dtype=np.float32)
    inside = sum(((grid[i] - center[i]) / radius[i]) ** 2 for i in range(3)) <= 1.0
    volume = np.zeros(shape, dtype=np.float32)
    noise = rng.normal(loc=scale, scale=scale * 0.2, size=shape)
    volume[inside] = np.abs(noise[inside]) + 1.0
    return volume


def write_case(
    case_dir: Path,
    *,
    shape: tuple[int, int, int] = (60, 64, 56),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_seg: bool = False,
    modalities: tuple[str, ...] = MODALITIES,
    affine: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write a synthetic 4-modality case laid out the way CaseStorage does."""
    case_dir.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([*spacing, 1.0]).astype(np.float64)
        affine[:3, 3] = [-float(shape[0]) / 2, -float(shape[1]) / 2, -float(shape[2]) / 2]

    paths: dict[str, Path] = {}
    for index, modality in enumerate(modalities):
        volume = phantom_volume(shape, scale=300.0 + 200.0 * index, seed=index + 1)
        image = nib.Nifti1Image(volume, affine)
        image.header.set_zooms(spacing)
        path = case_dir / f"{modality}.nii.gz"
        nib.save(image, str(path))
        paths[modality] = path

    if include_seg:
        seg = np.zeros(shape, dtype=np.uint8)
        c = [s // 2 for s in shape]
        seg[c[0] - 4 : c[0] + 4, c[1] - 4 : c[1] + 4, c[2] - 3 : c[2] + 3] = 2
        seg[c[0] - 2 : c[0] + 2, c[1] - 2 : c[1] + 2, c[2] - 1 : c[2] + 1] = 1
        seg[c[0] - 1 : c[0] + 1, c[1] - 1 : c[1] + 1, c[2] : c[2] + 1] = 4
        image = nib.Nifti1Image(seg, affine)
        image.header.set_zooms(spacing)
        path = case_dir / "seg.nii.gz"
        nib.save(image, str(path))
        paths["seg"] = path

    return paths


@pytest.fixture
def brats_config() -> BratsInferenceConfig:
    return small_config()


@pytest.fixture
def synthetic_case(tmp_path: Path) -> dict[str, Path]:
    return write_case(tmp_path / "case" / "input")


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path, brats_config: BratsInferenceConfig) -> Path:
    return write_synthetic_checkpoint(tmp_path / "synthetic.pth", brats_config)
