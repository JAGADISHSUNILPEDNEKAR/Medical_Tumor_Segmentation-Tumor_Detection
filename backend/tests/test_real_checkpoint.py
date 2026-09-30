"""Phase 6 — assertions against the REAL trained checkpoint.

These tests run only when the trained checkpoint is present; they skip
otherwise, so the suite stays green on a machine that does not have it.

Resolution order:
    1. $BRATS_CHECKPOINT
    2. $MODEL_PATH
    3. <repo>/models/best_model.pth

Nothing here runs a forward pass. Loading the weights is enough to prove that
the reconstructed architecture matches the trained one key-for-key and
shape-for-shape; a full 128^3 forward needs far more RAM than a typical
development machine has (see docs/Phase6_Real_Inference.md §9).

No metric is asserted or reported. `best_val_metric` is checked only for its
presence and type, never against a threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.inference.brats.checkpoint import load_brats_checkpoint  # noqa: E402
from app.inference.brats.config import (  # noqa: E402
    EXPECTED_COMPAT_FINGERPRINT,
    TRAINING_CONFIG,
)
from app.inference.brats.model import build_model, count_parameters  # noqa: E402
from app.inference.pytorch_service import RealBraTSInferenceService  # noqa: E402

# Printed by the notebook's own Section 12.3 output.
NOTEBOOK_PARAMETER_COUNT = 18_774_756
DEVICE = torch.device("cpu")


def _checkpoint_path() -> Path | None:
    for candidate in (os.environ.get("BRATS_CHECKPOINT"), os.environ.get("MODEL_PATH")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    default = Path(__file__).resolve().parents[2] / "models" / "best_model.pth"
    return default if default.is_file() else None


CHECKPOINT = _checkpoint_path()
requires_checkpoint = pytest.mark.skipif(
    CHECKPOINT is None,
    reason="Trained checkpoint not present (set BRATS_CHECKPOINT or MODEL_PATH)",
)


@requires_checkpoint
def test_real_checkpoint_loads_strictly_into_the_reconstructed_architecture() -> None:
    """The decisive test: the trained weights fit the production network exactly.

    `strict=True` means every key and every tensor shape must correspond. A
    single renamed layer, a changed channel width, or a bias that should not
    exist would fail here.
    """
    model = build_model(TRAINING_CONFIG.model)

    info = load_brats_checkpoint(CHECKPOINT, model, TRAINING_CONFIG, device=DEVICE)

    assert info.format == "model_state_dict"
    assert count_parameters(model) == NOTEBOOK_PARAMETER_COUNT


@requires_checkpoint
def test_real_checkpoint_fingerprint_matches_the_transcribed_config() -> None:
    """The checkpoint's own recorded fingerprint confirms the transcription.

    The notebook hashed its live `ModelConfig`, patch size and
    `PreprocessingConfig` into the checkpoint at save time. Reproducing that
    digest from `app/inference/brats/config.py` means every field name and
    value in the transcription matches what training actually used — not just
    the ones that happen to affect tensor shapes.
    """
    model = build_model(TRAINING_CONFIG.model)

    info = load_brats_checkpoint(CHECKPOINT, model, TRAINING_CONFIG, device=DEVICE)

    assert info.compat_fingerprint == EXPECTED_COMPAT_FINGERPRINT
    assert info.fingerprint_verified is True


@requires_checkpoint
def test_real_checkpoint_is_not_a_synthetic_fixture() -> None:
    model = build_model(TRAINING_CONFIG.model)
    info = load_brats_checkpoint(CHECKPOINT, model, TRAINING_CONFIG, device=DEVICE)
    assert info.synthetic_fixture is False


@requires_checkpoint
def test_real_checkpoint_carries_training_provenance() -> None:
    """Provenance is read from the file, never invented."""
    model = build_model(TRAINING_CONFIG.model)

    info = load_brats_checkpoint(CHECKPOINT, model, TRAINING_CONFIG, device=DEVICE)

    assert isinstance(info.epoch, int)
    assert isinstance(info.global_step, int)
    assert info.saved_at
    assert len(info.sha256_12) == 12
    # Present and numeric — deliberately NOT compared against any threshold.
    assert isinstance(info.best_val_metric, float)


@requires_checkpoint
def test_service_reports_a_loaded_real_model() -> None:
    service = RealBraTSInferenceService(CHECKPOINT, device_preference="cpu")
    service.load()

    assert service.model_loaded is True
    assert service.available is True
    assert service.inference_source == "pytorch"
    assert service.synthetic is False

    described = service.describe()
    assert described["parameters"] == NOTEBOOK_PARAMETER_COUNT
    assert described["patch_size"] == [128, 128, 128]
    assert described["channel_order"] == ["t1", "t1ce", "t2", "flair"]
    assert described["service_compat_fingerprint"] == EXPECTED_COMPAT_FINGERPRINT
    assert described["checkpoint"]["fingerprint_verified"] is True
    assert described["checkpoint"]["synthetic_fixture"] is False
    # Derived from the checkpoint's digest and epoch, not fabricated.
    assert service.checkpoint_id in service.model_version


@requires_checkpoint
def test_real_checkpoint_is_refused_by_a_mismatched_architecture() -> None:
    """Guard against a false-positive: the strict load is genuinely strict."""
    from dataclasses import replace

    from app.inference.brats.checkpoint import CheckpointError

    wrong = replace(
        TRAINING_CONFIG, model=replace(TRAINING_CONFIG.model, base_channels=16)
    )
    model = build_model(wrong.model)

    with pytest.raises(CheckpointError):
        load_brats_checkpoint(
            CHECKPOINT, model, wrong, device=DEVICE, strict_fingerprint=False
        )
