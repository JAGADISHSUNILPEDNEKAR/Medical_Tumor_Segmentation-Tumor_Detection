"""Checkpoint loading for the trained 3D U-Net.

The notebook writes checkpoints through `save_checkpoint` (Section 15.4) as a
dict payload:

    {
        "model_state_dict":     ...,
        "optimizer_state_dict": ...,
        "scaler_state_dict":    ...,
        "epoch":                int,
        "global_step":          int,
        "best_val_metric":      float | None,
        "compat_fingerprint":   str,
        "torch_rng_state":      ...,
        "numpy_rng_state":      ...,
        "saved_at":             isoformat str,
    }

A bare `model.state_dict()` and the common `{"state_dict": ...}` wrapper are
also accepted, because a checkpoint may reasonably be re-exported in either
form. Every other shape is rejected loudly.

Nothing in this module hides an incompatibility: the state dict is always
loaded with `strict=True`, and a compatibility-fingerprint mismatch is an
error by default.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.inference.brats.config import (
    BratsInferenceConfig,
    checkpoint_compat_fingerprint,
)

logger = logging.getLogger(__name__)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreadable, or incompatible."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Provenance read out of the checkpoint file itself. Never invented."""

    path: Path
    sha256_12: str
    size_bytes: int
    format: str
    epoch: int | None = None
    global_step: int | None = None
    best_val_metric: float | None = None
    compat_fingerprint: str | None = None
    saved_at: str | None = None
    fingerprint_verified: bool = False
    synthetic_fixture: bool = False
    """True when the file self-identifies as a randomly-initialized test
    fixture (written by scripts/make_synthetic_checkpoint.py), so results
    derived from it can never be presented as trained-model output."""


def file_digest_12(path: Path) -> str:
    """Stable 12-hex-character identifier for a checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _load_payload(path: Path, device: torch.device) -> Any:
    """Read the checkpoint, preferring the safe `weights_only=True` path.

    The notebook's payload carries a NumPy RNG state, which `weights_only=True`
    refuses to unpickle. Falling back is safe here specifically because the
    checkpoint is operator-supplied through MODEL_PATH — it is never a file an
    API client can upload — but the fallback is logged rather than silent.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as safe_exc:  # noqa: BLE001 - reason is reported below
        logger.info(
            "checkpoint_weights_only_load_failed_retrying",
            extra={"checkpoint": path.name, "reason": type(safe_exc).__name__},
        )
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001 - re-raised as CheckpointError
            raise CheckpointError(
                f"Checkpoint at '{path.name}' could not be deserialized by "
                f"torch.load ({type(exc).__name__}: {exc}). The file is missing, "
                f"truncated, or not a PyTorch checkpoint."
            ) from exc


def _extract_state_dict(payload: Any, path: Path) -> tuple[dict[str, Any], str]:
    """Return (state_dict, detected format label)."""
    if not isinstance(payload, dict):
        raise CheckpointError(
            f"Checkpoint at '{path.name}' deserialized to "
            f"{type(payload).__name__}, not a dict. Expected either a training "
            f"payload containing 'model_state_dict' or a bare state_dict."
        )

    for key in ("model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate, key

    # A bare `model.state_dict()`: every value is a tensor.
    if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload, "bare_state_dict"

    raise CheckpointError(
        f"Checkpoint at '{path.name}' is a dict but contains no recognizable "
        f"model weights. Expected a 'model_state_dict' key (the format written "
        f"by the training notebook), a 'state_dict' key, or a bare state_dict. "
        f"Top-level keys present: {sorted(str(k) for k in payload)[:12]}"
    )


def _describe_load_failure(exc: Exception, path: Path) -> str:
    return (
        f"Checkpoint at '{path.name}' does not match the reconstructed UNet3D "
        f"architecture. The weights were NOT loaded — inference with a partially "
        f"initialized network would silently produce meaningless segmentations.\n"
        f"torch.load_state_dict(strict=True) reported:\n{exc}"
    )


def load_brats_checkpoint(
    path: Path,
    model: nn.Module,
    config: BratsInferenceConfig,
    *,
    device: torch.device,
    strict_fingerprint: bool = True,
) -> CheckpointInfo:
    """Load trained weights into `model` in place.

    Raises `CheckpointError` — never returns a partially loaded model — when
    the file is missing or corrupt, when the checkpoint was written under a
    different model/patch/preprocessing configuration, or when the state dict
    does not match the architecture key-for-key and shape-for-shape.
    """
    if not path.is_file():
        raise CheckpointError(
            f"No checkpoint file at '{path}'. Set MODEL_PATH to the trained "
            f"checkpoint exported by the notebook (best_model.pth), or run with "
            f"INFERENCE_BACKEND=mock."
        )

    payload = _load_payload(path, device)
    state_dict, detected_format = _extract_state_dict(payload, path)

    stored_fingerprint = (
        payload.get("compat_fingerprint") if isinstance(payload, dict) else None
    )
    expected_fingerprint = checkpoint_compat_fingerprint(config)
    fingerprint_verified = False

    if isinstance(stored_fingerprint, str):
        if stored_fingerprint == expected_fingerprint:
            fingerprint_verified = True
        elif strict_fingerprint:
            raise CheckpointError(
                f"Checkpoint at '{path.name}' was saved under a different "
                f"model/patch/preprocessing configuration than this service "
                f"reconstructs (checkpoint fingerprint {stored_fingerprint!r}, "
                f"service fingerprint {expected_fingerprint!r}). Refusing to load: "
                f"the weights may fit the layer shapes while the preprocessing "
                f"they were trained against no longer matches. Reconcile "
                f"app/inference/brats/config.py with the notebook's CONFIG, or "
                f"set INFERENCE_STRICT_FINGERPRINT=false if the difference is "
                f"understood and intentional."
            )
        else:
            logger.warning(
                "checkpoint_fingerprint_mismatch_allowed",
                extra={
                    "checkpoint": path.name,
                    "checkpoint_fingerprint": stored_fingerprint,
                    "service_fingerprint": expected_fingerprint,
                },
            )
    else:
        logger.warning(
            "checkpoint_fingerprint_absent",
            extra={"checkpoint": path.name, "format": detected_format},
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(_describe_load_failure(exc, path)) from exc

    def _maybe(key: str) -> Any:
        value = payload.get(key) if isinstance(payload, dict) else None
        return value

    epoch = _maybe("epoch")
    global_step = _maybe("global_step")
    best_val_metric = _maybe("best_val_metric")
    saved_at = _maybe("saved_at")

    info = CheckpointInfo(
        path=path,
        sha256_12=file_digest_12(path),
        size_bytes=path.stat().st_size,
        format=detected_format,
        epoch=int(epoch) if isinstance(epoch, (int, float)) else None,
        global_step=int(global_step) if isinstance(global_step, (int, float)) else None,
        best_val_metric=float(best_val_metric)
        if isinstance(best_val_metric, (int, float))
        else None,
        compat_fingerprint=stored_fingerprint
        if isinstance(stored_fingerprint, str)
        else None,
        saved_at=str(saved_at) if saved_at is not None else None,
        fingerprint_verified=fingerprint_verified,
        synthetic_fixture=bool(payload.get("synthetic_fixture", False))
        if isinstance(payload, dict)
        else False,
    )
    logger.info(
        "checkpoint_loaded",
        extra={
            "checkpoint": path.name,
            "checkpoint_id": info.sha256_12,
            "format": detected_format,
            "epoch": info.epoch,
            "fingerprint_verified": fingerprint_verified,
        },
    )
    return info
