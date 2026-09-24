"""Notebook-parity BraTS inference components.

Importing this package pulls in PyTorch. The mock backend must stay importable
on a machine without torch installed, so `app.inference.factory` imports this
lazily and nothing under `app.api` imports it at module level.
"""

from app.inference.brats.config import (
    ARCHITECTURE_DESCRIPTION,
    CLASS_NAMES,
    EXPECTED_COMPAT_FINGERPRINT,
    LABEL_MAP_INTERNAL_TO_RAW,
    LABEL_MAP_RAW_TO_INTERNAL,
    MODALITY_ORDER,
    PREPROCESSING_DESCRIPTION,
    RAW_LABELS,
    TRAINING_CONFIG,
    BratsInferenceConfig,
    BratsModelConfig,
    BratsPatchConfig,
    BratsPostprocessingConfig,
    BratsPreprocessingConfig,
    checkpoint_compat_fingerprint,
)

__all__ = [
    "ARCHITECTURE_DESCRIPTION",
    "CLASS_NAMES",
    "EXPECTED_COMPAT_FINGERPRINT",
    "LABEL_MAP_INTERNAL_TO_RAW",
    "LABEL_MAP_RAW_TO_INTERNAL",
    "MODALITY_ORDER",
    "PREPROCESSING_DESCRIPTION",
    "RAW_LABELS",
    "TRAINING_CONFIG",
    "BratsInferenceConfig",
    "BratsModelConfig",
    "BratsPatchConfig",
    "BratsPostprocessingConfig",
    "BratsPreprocessingConfig",
    "checkpoint_compat_fingerprint",
]
