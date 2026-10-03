"""Real PyTorch inference against the trained 3D U-Net.

This service satisfies the same `InferenceService` protocol as
`MockInferenceService` and returns the same `InferenceResult` contract, so
nothing downstream — MeasurementService, MetricsService, ResultService, the
API schemas, the viewer, the report — needs to know a real model ran. The only
observable difference is provenance: `inference_source="pytorch"` and
`synthetic=False`.

Pipeline (each step is the notebook's, see `app/inference/brats/`):

    validate (Phase 2 validators)
        -> load + foreground z-score + foreground-bbox crop
        -> 4-channel tensor, channel order [T1, T1ce, T2, FLAIR]
        -> Gaussian-weighted sliding-window inference at 128^3, overlap 0.5
        -> argmax over 4 classes
        -> connected-component filter (min 50 voxels, per class)
        -> internal {0,1,2,3} -> BraTS {0,1,2,4}
        -> un-crop back into the original volume geometry
        -> segmentation.nii.gz with the input's affine, spacing and form codes
"""

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from app.core.constants import ALLOWED_NIFTI_EXTENSIONS, REQUIRED_MODALITIES
from app.inference.base import InferenceResult
from app.inference.brats.checkpoint import (
    CheckpointError,
    CheckpointInfo,
    load_brats_checkpoint,
)
from app.inference.brats.config import (
    ARCHITECTURE_DESCRIPTION,
    LABEL_MAP_INTERNAL_TO_RAW,
    MODALITY_ORDER,
    PREPROCESSING_DESCRIPTION,
    BratsInferenceConfig,
    TRAINING_CONFIG,
    checkpoint_compat_fingerprint,
)
from app.inference.brats.model import build_model, count_parameters
from app.inference.brats.postprocess import (
    connected_component_filter,
    internal_to_brats_labels,
    restore_prediction_to_original_space,
)
from app.inference.brats.preprocessing import load_and_preprocess_case
from app.inference.brats.sliding_window import (
    make_gaussian_weight_kernel,
    sliding_window_infer,
)
from app.validation.nifti import assert_spatially_compatible, inspect_nifti

logger = logging.getLogger(__name__)

SEGMENTATION_FILENAME = "segmentation.nii.gz"

_RESULT_DESCRIPTION = (
    "Segmentation produced by the trained 3D U-Net (custom UNet3D) using "
    "Gaussian-weighted sliding-window inference. Research / decision-support "
    "prototype output — not a diagnosis and not clinically validated."
)


def resolve_device(preference: str) -> torch.device:
    """Resolve the configured device preference to a real torch device.

    'auto' selects CUDA when available and falls back to CPU. A GPU is never
    required for the API to start, and CPU is never silently reported as GPU.
    """
    normalized = (preference or "auto").strip().lower()
    if normalized in {"auto", ""}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise CheckpointError(
            f"INFERENCE_DEVICE={preference!r} requests CUDA but torch reports no "
            f"CUDA device on this host. Use INFERENCE_DEVICE=auto or cpu."
        )
    return torch.device(normalized)


class RealBraTSInferenceService:
    """Trained 3D U-Net inference. Conforms to the `InferenceService` protocol."""

    inference_source = "pytorch"
    synthetic = False

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device_preference: str = "auto",
        model_version: str | None = None,
        strict_fingerprint: bool = True,
        config: BratsInferenceConfig | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_preference = device_preference
        self.config = config or TRAINING_CONFIG
        self.strict_fingerprint = strict_fingerprint
        self._configured_model_version = model_version

        self._device: torch.device | None = None
        self._model: torch.nn.Module | None = None
        self._checkpoint_info: CheckpointInfo | None = None
        self._model_load_seconds: float | None = None
        self._parameter_count: int | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Build the network and load the checkpoint once, at startup.

        Raises `CheckpointError` on any failure. The model is loaded exactly
        once per process and reused by every job; jobs never reconstruct it.
        """
        started = time.perf_counter()
        device = resolve_device(self.device_preference)
        model = build_model(self.config.model)
        info = load_brats_checkpoint(
            self.checkpoint_path,
            model,
            self.config,
            device=device,
            strict_fingerprint=self.strict_fingerprint,
        )
        model.to(device)
        model.eval()

        self._device = device
        self._model = model
        self._checkpoint_info = info
        self._parameter_count = count_parameters(model)
        self._model_load_seconds = time.perf_counter() - started

        logger.info(
            "real_inference_model_loaded",
            extra={
                "inference_source": self.inference_source,
                "device": str(device),
                "checkpoint_id": info.sha256_12,
                "parameters": self._parameter_count,
                "model_load_seconds": round(self._model_load_seconds, 3),
            },
        )

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def model_loaded(self) -> bool:
        """True only when a real checkpoint was actually loaded."""
        return self._model is not None

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def checkpoint_id(self) -> str | None:
        return self._checkpoint_info.sha256_12 if self._checkpoint_info else None

    @property
    def model_version(self) -> str | None:
        """Configured MODEL_VERSION, else a value derived from the checkpoint.

        The derived form is not invented: it is the checkpoint file's SHA-256
        prefix plus the training epoch recorded inside the checkpoint.
        """
        if self._configured_model_version:
            return self._configured_model_version
        info = self._checkpoint_info
        if info is None:
            return None
        if info.epoch is not None:
            return f"unet3d-ep{info.epoch}-{info.sha256_12}"
        return f"unet3d-{info.sha256_12}"

    def describe(self) -> dict[str, Any]:
        """Provenance for GET /model/info. Contains only measured values."""
        info = self._checkpoint_info
        description: dict[str, Any] = {
            "inference_source": self.inference_source,
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "checkpoint_id": self.checkpoint_id,
            "architecture": ARCHITECTURE_DESCRIPTION,
            "num_classes": self.config.model.num_classes,
            "input_modalities": [m.upper() for m in MODALITY_ORDER],
            "channel_order": list(MODALITY_ORDER),
            "patch_size": list(self.config.patch.selected_size),
            "sliding_window_overlap": self.config.patch.sliding_window_overlap,
            "device": str(self._device) if self._device else None,
            "parameters": self._parameter_count,
            "preprocessing": PREPROCESSING_DESCRIPTION,
            "service_compat_fingerprint": checkpoint_compat_fingerprint(self.config),
        }
        if self._device is not None and self._device.type == "cuda":
            description["gpu_name"] = torch.cuda.get_device_name(self._device)
            description["gpu_memory_total_bytes"] = torch.cuda.get_device_properties(self._device).total_memory
        
        if self._model_load_seconds is not None:
            description["model_load_seconds"] = round(self._model_load_seconds, 3)
        if info is not None:
            description["checkpoint"] = {
                "format": info.format,
                "size_bytes": info.size_bytes,
                "epoch": info.epoch,
                "global_step": info.global_step,
                "saved_at": info.saved_at,
                "compat_fingerprint": info.compat_fingerprint,
                "fingerprint_verified": info.fingerprint_verified,
                # Training-time proxy selection metric recorded by the
                # notebook. NOT a validated headline Dice — reported under its
                # own name so it can never be mistaken for one.
                "proxy_val_mean_soft_dice": info.best_val_metric,
                "synthetic_fixture": info.synthetic_fixture,
            }
        return description

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        case_dir: Path,
        output_dir: Path,
        case_id: str,
        *,
        has_ground_truth: bool = False,
    ) -> InferenceResult:
        if self._model is None or self._device is None:
            raise CheckpointError(
                "RealBraTSInferenceService.predict() was called before the "
                "checkpoint was loaded. Call load() during application startup."
            )

        logger.info(
            "real_inference_started",
            extra={"case_id": case_id, "inference_source": self.inference_source},
        )
        wall_start = time.perf_counter()

        modality_paths = self._resolve_modalities(case_dir, case_id)
        reference_info = self._validate_inputs(modality_paths, case_id)

        # 1. Preprocess — notebook Section 8.3.
        preprocess_start = time.perf_counter()
        case = load_and_preprocess_case(modality_paths, self.config.preprocessing)
        preprocess_seconds = time.perf_counter() - preprocess_start

        # 2. Gaussian-weighted sliding window — notebook Sections 16.1 + 19.
        # Reset the CUDA peak counter first: without this, the value reported
        # below is the process-wide high-water mark rather than this case's.
        self._reset_peak_gpu_stats()
        inference_start = time.perf_counter()
        prob_map, window_stats = sliding_window_infer(
            case.image,
            self._model,
            self.config.model,
            self.config.patch,
            self._device,
            weight_kernel_fn=lambda size: make_gaussian_weight_kernel(
                size, self.config.patch.gaussian_sigma_scale
            ),
        )
        pred_internal = np.argmax(prob_map, axis=0).astype(np.uint8)
        del prob_map
        inference_seconds = time.perf_counter() - inference_start

        peak_gpu_bytes = self._peak_gpu_bytes()

        # 3. Connected-component filtering — notebook Section 20.1.
        min_voxels = self.config.postprocessing.connected_component_min_voxels
        pred_clean, removed_voxels = connected_component_filter(
            pred_internal, min_voxels=min_voxels
        )
        del pred_internal

        # 4. Internal {0,1,2,3} -> BraTS {0,1,2,4} — notebook Section 19.2.
        pred_raw = internal_to_brats_labels(pred_clean)
        del pred_clean

        # 5. Invert the foreground-bbox crop — notebook Section 21.1.
        pred_full = restore_prediction_to_original_space(
            pred_raw, case.original_shape, case.crop_bbox
        )
        del pred_raw

        # 6. Write the artifact in the input's geometry.
        seg_path = self._write_segmentation(
            pred_full,
            output_dir=output_dir,
            reference_path=modality_paths[MODALITY_ORDER[0]],
            affine=case.affine,
            spacing=case.spacing,
        )

        total_seconds = time.perf_counter() - wall_start
        metadata = self._build_metadata(
            case=case,
            window_stats=window_stats,
            removed_voxels=removed_voxels,
            min_voxels=min_voxels,
            pred_full=pred_full,
            reference_orientation=reference_info,
            has_ground_truth=has_ground_truth,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
            total_seconds=total_seconds,
            peak_gpu_bytes=peak_gpu_bytes,
        )

        (output_dir / "result.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        logger.info(
            "real_inference_completed",
            extra={
                "case_id": case_id,
                "inference_source": self.inference_source,
                "output_shape": list(int(s) for s in pred_full.shape),
                "num_patches": window_stats.num_patches,
                "inference_seconds": round(inference_seconds, 3),
            },
        )
        self._release_device_memory()
        
        # Free memory aggressively on CPU (where empty_cache is a no-op)
        if self._device is None or self._device.type == "cpu":
            gc.collect()

        return InferenceResult(
            segmentation_path=seg_path.name,
            metadata=metadata,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_modalities(self, case_dir: Path, case_id: str) -> dict[str, Path]:
        """Locate the four stored modality files. Order comes from the notebook."""
        paths: dict[str, Path] = {}
        for modality in REQUIRED_MODALITIES:
            for extension in ALLOWED_NIFTI_EXTENSIONS:
                candidate = case_dir / f"{modality}{extension}"
                if candidate.is_file():
                    paths[modality] = candidate
                    break
            else:
                raise RuntimeError(
                    f"Expected modality file for '{modality}' not found in case "
                    f"directory. Case: {case_id}"
                )
        return {modality: paths[modality] for modality in MODALITY_ORDER}

    def _validate_inputs(self, modality_paths: dict[str, Path], case_id: str):
        """Re-run the Phase 2 validators rather than duplicating their rules.

        The case passed validation at upload; re-checking here catches a file
        that changed or was truncated since, and it is the same code path, so
        error codes and messages stay consistent across the application.
        """
        volumes = {
            modality: inspect_nifti(path, is_segmentation=False, case_id=case_id)
            for modality, path in modality_paths.items()
        }
        assert_spatially_compatible(volumes, case_id=case_id)
        return volumes[MODALITY_ORDER[0]]

    def _write_segmentation(
        self,
        mask: np.ndarray,
        *,
        output_dir: Path,
        reference_path: Path,
        affine: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> Path:
        """Write segmentation.nii.gz in the reference volume's geometry."""
        output_dir.mkdir(parents=True, exist_ok=True)
        seg_path = output_dir / SEGMENTATION_FILENAME

        seg_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        seg_img.set_data_dtype(np.uint8)
        seg_img.header.set_zooms(tuple(float(z) for z in spacing))

        # Carry the input's qform/sform codes so the exported mask sits in the
        # same declared coordinate system as the modalities, not an
        # unqualified identity frame.
        reference_header = nib.load(str(reference_path)).header
        qform, qcode = reference_header.get_qform(coded=True)
        sform, scode = reference_header.get_sform(coded=True)
        if qcode:
            seg_img.set_qform(qform, code=int(qcode))
        if scode:
            seg_img.set_sform(sform, code=int(scode))

        nib.save(seg_img, str(seg_path))
        return seg_path

    def _reset_peak_gpu_stats(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

    def _peak_gpu_bytes(self) -> int | None:
        """Peak device memory for THIS case, measured since the reset above."""
        if self._device is not None and self._device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self._device))
        return None

    def _release_device_memory(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_metadata(
        self,
        *,
        case,
        window_stats,
        removed_voxels: dict[int, int],
        min_voxels: int,
        pred_full: np.ndarray,
        reference_orientation,
        has_ground_truth: bool,
        preprocess_seconds: float,
        inference_seconds: float,
        total_seconds: float,
        peak_gpu_bytes: int | None,
    ) -> dict[str, Any]:
        info = self._checkpoint_info
        observed_labels = sorted(int(v) for v in np.unique(pred_full))
        class_names = {1: "NCR", 2: "ED", 3: "ET"}

        metadata: dict[str, Any] = {
            "inference_source": self.inference_source,
            "synthetic": False,
            "model_version": self.model_version,
            "checkpoint_id": self.checkpoint_id,
            "architecture": ARCHITECTURE_DESCRIPTION,
            "model_parameters": self._parameter_count,
            "preprocessing_version": self.config.preprocessing.normalize_mode,
            "preprocessing": PREPROCESSING_DESCRIPTION,
            "input_modalities": list(MODALITY_ORDER),
            "channel_order": list(MODALITY_ORDER),
            "input_shape": list(int(s) for s in case.original_shape),
            "input_spacing": [float(z) for z in case.spacing],
            "input_orientation": list(reference_orientation.orientation),
            "cropped_shape": list(int(s) for s in case.image.shape[1:]),
            "crop_bbox": None
            if case.crop_bbox is None
            else [[int(s.start), int(s.stop)] for s in case.crop_bbox],
            "normalization_stats": case.norm_stats,
            "patch_size": list(self.config.patch.selected_size),
            "sliding_window_overlap": self.config.patch.sliding_window_overlap,
            "sliding_window_stride": list(window_stats.stride),
            "sliding_window_weighting": "gaussian",
            "gaussian_sigma_scale": self.config.patch.gaussian_sigma_scale,
            "num_patches": window_stats.num_patches,
            "postprocessing": {
                "connected_component_min_voxels": min_voxels,
                "removed_voxels_by_class": {
                    class_names[k]: int(v) for k, v in removed_voxels.items()
                },
            },
            "label_mapping": {
                "internal_to_brats": {
                    str(k): v for k, v in LABEL_MAP_INTERNAL_TO_RAW.items()
                }
            },
            "output_shape": list(int(s) for s in pred_full.shape),
            "output_labels": observed_labels,
            "device": str(self._device),
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "preprocess_seconds": round(preprocess_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "has_ground_truth": has_ground_truth,
            "description": _RESULT_DESCRIPTION,
        }
        if peak_gpu_bytes is not None:
            metadata["peak_gpu_memory_bytes"] = peak_gpu_bytes
        if info is not None:
            metadata["checkpoint_epoch"] = info.epoch
            metadata["checkpoint_saved_at"] = info.saved_at
            metadata["checkpoint_fingerprint_verified"] = info.fingerprint_verified
            if info.synthetic_fixture:
                # The checkpoint self-identifies as randomly initialized. Say so
                # on every result derived from it rather than letting a fixture
                # masquerade as the trained model.
                metadata["checkpoint_is_synthetic_fixture"] = True
                metadata["description"] = (
                    "Segmentation produced with a SYNTHETIC, randomly "
                    "initialized checkpoint fixture. Not a trained model, not a "
                    "prediction, and not usable for any quality claim."
                )
        return metadata
