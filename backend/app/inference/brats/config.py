"""Frozen training configuration transcribed from the research notebook.

Every value in this module is a transcription of what the supplied training
notebook (`notebook/notebookd7ed0558bb (2).ipynb`) actually used for the run
that produced `best_model.pth`. Nothing here is a default borrowed from a
generic 3D U-Net tutorial, and nothing here is tunable through the
environment: a value that drifts from training silently breaks inference, so
the only safe place for these is source control.

Provenance (notebook section → value):

    2.1 / 2.2   label map, in_channels, num_classes, num_stages,
                base_channels, max_channels, norm, activation,
                negative_slope, downsample_mode, upsample_mode
    5.1         dataset geometry is uniform → resampling_required = False
    8.1         canonical orientation ('L', 'P', 'S'), uniform across dataset
    8.2 / 8.3   foreground z-score normalization, foreground-bbox crop,
                crop margin 4 voxels, background sentinel 0.0
    9.1 / 9.2   selected patch size (128, 128, 128)
    10.1        sagittal axis index 0 (derived from ('L','P','S'))
    12.1        base_channels stayed at 32 (GPU run; the CPU derate to 16 in
                the notebook's cell did not fire — channel schedule printed
                as [32, 64, 128, 256, 320], 18,774,756 parameters)
    15.4        checkpoint compatibility fingerprint
    19.1        Gaussian weight kernel, sigma_scale 0.125
    16.1        sliding-window overlap 0.5, end-only padding
    20.1        connected-component filter, min_voxels 50
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# ── Label semantics (notebook Section 2.1) ───────────────────────────────────
# Fixed by BraTS, not a design choice. The network is trained on the internal
# contiguous IDs; exported masks must be restored to the raw BraTS IDs.
RAW_LABELS: tuple[int, ...] = (0, 1, 2, 4)
CLASS_NAMES: tuple[str, ...] = ("background", "NCR", "ED", "ET")
LABEL_MAP_RAW_TO_INTERNAL: dict[int, int] = {0: 0, 1: 1, 2: 2, 4: 3}
LABEL_MAP_INTERNAL_TO_RAW: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 4}

# ── Modality channel order (notebook Section 8.3) ────────────────────────────
# `load_and_preprocess_case` stacks in exactly this order. Never sorted.
MODALITY_ORDER: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class BratsModelConfig:
    """Mirror of the notebook's `ModelConfig` at training time.

    Field names and values must match the notebook exactly: they are hashed
    into the checkpoint compatibility fingerprint (notebook Section 15.4).
    """

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

    @property
    def channel_schedule(self) -> tuple[int, ...]:
        """[32, 64, 128, 256, 320] for the trained configuration."""
        return tuple(
            min(self.base_channels * (2**i), self.max_channels)
            for i in range(self.num_stages)
        )


@dataclass(frozen=True)
class BratsPreprocessingConfig:
    """Mirror of the notebook's `PreprocessingConfig` at training time."""

    normalize_mode: str = "foreground_zscore_per_case"
    background_value: float = 0.0
    crop_to_foreground: bool = True
    crop_margin_voxels: int = 4
    canonical_orientation: tuple[str, str, str] = ("L", "P", "S")
    orientation_uniform_across_dataset: bool = True
    sagittal_axis_index: int = 0


@dataclass(frozen=True)
class BratsPatchConfig:
    """Patch and sliding-window geometry (notebook Sections 9.1, 16.1, 19.1)."""

    selected_size: tuple[int, int, int] = (128, 128, 128)
    sliding_window_overlap: float = 0.5
    gaussian_sigma_scale: float = 0.125


@dataclass(frozen=True)
class BratsPostprocessingConfig:
    """Connected-component filtering (notebook Section 20.1)."""

    connected_component_min_voxels: int = 50


@dataclass(frozen=True)
class BratsInferenceConfig:
    """The complete frozen training→inference contract."""

    model: BratsModelConfig = field(default_factory=BratsModelConfig)
    preprocessing: BratsPreprocessingConfig = field(
        default_factory=BratsPreprocessingConfig
    )
    patch: BratsPatchConfig = field(default_factory=BratsPatchConfig)
    postprocessing: BratsPostprocessingConfig = field(
        default_factory=BratsPostprocessingConfig
    )


TRAINING_CONFIG = BratsInferenceConfig()


def checkpoint_compat_fingerprint(config: BratsInferenceConfig) -> str:
    """Reproduce the notebook's Section 15.4 compatibility fingerprint.

    The notebook hashes exactly three things — the model shape, the patch size
    the network's input geometry was built around, and the preprocessing
    convention the weights were trained against — and stores the result in
    every checkpoint under `compat_fingerprint`. Training hyperparameters are
    deliberately excluded: they may differ across a resumed run without
    invalidating the weights.

    The serialization (``sort_keys=True``, ``default=str``, 12-char SHA-256
    prefix) is reproduced byte-for-byte so the digest computed here is
    directly comparable to the one stored in the checkpoint.
    """
    compat_subset = {
        "model": asdict(config.model),
        "patch_selected_size": config.patch.selected_size,
        "preprocessing": asdict(config.preprocessing),
    }
    serialized = json.dumps(compat_subset, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


# Fingerprint implied by TRAINING_CONFIG. Recorded for documentation and
# diagnostics only — the authoritative comparison is always made against the
# value actually stored inside the checkpoint being loaded.
EXPECTED_COMPAT_FINGERPRINT = checkpoint_compat_fingerprint(TRAINING_CONFIG)

ARCHITECTURE_DESCRIPTION = (
    "UNet3D (custom 3D U-Net, notebook Section 12.3) — 5 encoder stages, "
    "channel schedule [32, 64, 128, 256, 320], two 3x3x3 Conv3d per block with "
    "InstanceNorm3d(affine=True) + LeakyReLU(0.01), strided-conv downsampling, "
    "transposed-conv upsampling with skip concatenation, 1x1x1 output conv to "
    "4 classes. Not nnU-Net, not MONAI, not SegResNet (PRD ADR-001)."
)

PREPROCESSING_DESCRIPTION = (
    "Per-case foreground-only (voxel != 0) z-score per modality with the "
    "background held at exactly 0, followed by a union-foreground bounding-box "
    "crop with a 4-voxel margin. No resampling and no reorientation: the "
    "notebook measured the dataset as geometrically uniform "
    "(240x240x155 @ 1.0mm, orientation L/P/S)."
)
