"""Phase 6 — checkpoint loading.

Covers the four failure modes that must never be silent: missing file, corrupt
file, wrong architecture, and a fingerprint that says the weights were trained
under a different preprocessing/patch configuration.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.inference.brats.checkpoint import (  # noqa: E402
    CheckpointError,
    file_digest_12,
    load_brats_checkpoint,
)
from app.inference.brats.config import (  # noqa: E402
    EXPECTED_COMPAT_FINGERPRINT,
    TRAINING_CONFIG,
    checkpoint_compat_fingerprint,
)
from app.inference.brats.model import build_model  # noqa: E402
from tests.phase6_helpers import small_config, write_synthetic_checkpoint  # noqa: E402

DEVICE = torch.device("cpu")


def test_fingerprint_is_deterministic_and_config_sensitive() -> None:
    config = TRAINING_CONFIG
    assert checkpoint_compat_fingerprint(config) == EXPECTED_COMPAT_FINGERPRINT
    assert checkpoint_compat_fingerprint(config) == checkpoint_compat_fingerprint(config)

    wider = replace(config, model=replace(config.model, base_channels=64))
    assert checkpoint_compat_fingerprint(wider) != EXPECTED_COMPAT_FINGERPRINT

    other_patch = replace(config, patch=replace(config.patch, selected_size=(96, 96, 96)))
    assert checkpoint_compat_fingerprint(other_patch) != EXPECTED_COMPAT_FINGERPRINT

    other_prep = replace(
        config,
        preprocessing=replace(config.preprocessing, crop_margin_voxels=8),
    )
    assert checkpoint_compat_fingerprint(other_prep) != EXPECTED_COMPAT_FINGERPRINT


def test_valid_notebook_format_checkpoint_loads(tmp_path: Path) -> None:
    config = small_config()
    path = write_synthetic_checkpoint(tmp_path / "ok.pth", config, seed=1, epoch=19)
    model = build_model(config.model)

    info = load_brats_checkpoint(path, model, config, device=DEVICE)

    assert info.format == "model_state_dict"
    assert info.epoch == 19
    assert info.fingerprint_verified is True
    assert info.synthetic_fixture is True
    assert info.sha256_12 == file_digest_12(path)
    assert len(info.sha256_12) == 12


def test_loaded_weights_are_the_checkpoint_weights(tmp_path: Path) -> None:
    """Loading must actually replace the randomly initialized parameters."""
    config = small_config()
    path = write_synthetic_checkpoint(tmp_path / "ok.pth", config, seed=5)
    saved = torch.load(path, map_location=DEVICE, weights_only=False)["model_state_dict"]

    model = build_model(config.model)
    before = model.final_conv.weight.detach().clone()
    load_brats_checkpoint(path, model, config, device=DEVICE)

    assert not torch.equal(before, model.final_conv.weight)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved[key]), key


def test_bare_state_dict_checkpoint_is_accepted(tmp_path: Path) -> None:
    config = small_config()
    path = write_synthetic_checkpoint(
        tmp_path / "bare.pth", config, seed=2, bare_state_dict=True
    )
    model = build_model(config.model)

    info = load_brats_checkpoint(path, model, config, device=DEVICE)

    assert info.format == "bare_state_dict"
    assert info.fingerprint_verified is False
    assert info.epoch is None


def test_missing_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = small_config()
    model = build_model(config.model)
    with pytest.raises(CheckpointError, match="No checkpoint file"):
        load_brats_checkpoint(tmp_path / "absent.pth", model, config, device=DEVICE)


def test_corrupt_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = small_config()
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"this is not a torch checkpoint, it is just some bytes")
    model = build_model(config.model)

    with pytest.raises(CheckpointError, match="could not be deserialized"):
        load_brats_checkpoint(path, model, config, device=DEVICE)


def test_truncated_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = small_config()
    full = write_synthetic_checkpoint(tmp_path / "full.pth", config)
    truncated = tmp_path / "truncated.pth"
    truncated.write_bytes(full.read_bytes()[: len(full.read_bytes()) // 2])
    model = build_model(config.model)

    with pytest.raises(CheckpointError):
        load_brats_checkpoint(truncated, model, config, device=DEVICE)


def test_payload_without_weights_fails_clearly(tmp_path: Path) -> None:
    config = small_config()
    path = tmp_path / "no_weights.pth"
    torch.save({"epoch": 3, "notes": "nothing useful here"}, path)
    model = build_model(config.model)

    with pytest.raises(CheckpointError, match="no recognizable"):
        load_brats_checkpoint(path, model, config, device=DEVICE)


def test_architecture_mismatch_fails_and_never_uses_strict_false(tmp_path: Path) -> None:
    """A checkpoint from a different width must be refused, not partially loaded."""
    saved_config = small_config(base_channels=4)
    path = write_synthetic_checkpoint(tmp_path / "narrow.pth", saved_config)

    loading_config = small_config(base_channels=8)
    model = build_model(loading_config.model)
    before = model.final_conv.weight.detach().clone()

    with pytest.raises(CheckpointError) as excinfo:
        load_brats_checkpoint(
            path, model, loading_config, device=DEVICE, strict_fingerprint=False
        )

    message = str(excinfo.value)
    assert "does not match the reconstructed UNet3D architecture" in message
    assert "size mismatch" in message or "shape" in message.lower()
    # The model must be left untouched: no half-loaded network may escape.
    assert torch.equal(before, model.final_conv.weight)


def test_missing_and_unexpected_keys_are_reported(tmp_path: Path) -> None:
    config = small_config()
    model = build_model(config.model)
    state = model.state_dict()
    state.pop("final_conv.weight")
    state["not_a_real_layer.weight"] = torch.zeros(1)
    path = tmp_path / "keys.pth"
    torch.save({"model_state_dict": state}, path)

    fresh = build_model(config.model)
    with pytest.raises(CheckpointError) as excinfo:
        load_brats_checkpoint(path, fresh, config, device=DEVICE)

    message = str(excinfo.value)
    assert "final_conv.weight" in message
    assert "not_a_real_layer.weight" in message


def test_fingerprint_mismatch_is_refused_by_default(tmp_path: Path) -> None:
    config = small_config()
    path = write_synthetic_checkpoint(
        tmp_path / "wrong_fp.pth", config, fingerprint="deadbeefcafe"
    )
    model = build_model(config.model)

    with pytest.raises(CheckpointError, match="different model/patch/preprocessing"):
        load_brats_checkpoint(path, model, config, device=DEVICE)


def test_fingerprint_mismatch_can_be_overridden_explicitly(tmp_path: Path) -> None:
    """The escape hatch relaxes the fingerprint only — never state_dict strictness."""
    config = small_config()
    path = write_synthetic_checkpoint(
        tmp_path / "wrong_fp.pth", config, fingerprint="deadbeefcafe"
    )
    model = build_model(config.model)

    info = load_brats_checkpoint(
        path, model, config, device=DEVICE, strict_fingerprint=False
    )

    assert info.compat_fingerprint == "deadbeefcafe"
    assert info.fingerprint_verified is False


def test_checkpoint_without_fingerprint_loads_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = small_config()
    path = write_synthetic_checkpoint(
        tmp_path / "no_fp.pth", config, include_fingerprint=False
    )
    model = build_model(config.model)

    with caplog.at_level("WARNING"):
        info = load_brats_checkpoint(path, model, config, device=DEVICE)

    assert info.compat_fingerprint is None
    assert info.fingerprint_verified is False
    assert "checkpoint_fingerprint_absent" in caplog.text


def test_fingerprint_catches_a_change_strict_state_dict_loading_cannot(
    tmp_path: Path,
) -> None:
    """The case that justifies the fingerprint existing at all.

    UNet3D is fully convolutional, so weights trained at one patch size load
    without complaint at another — `strict=True` sees identical keys and
    identical shapes. Only the recorded fingerprint notices that the patch
    geometry (and with it the sliding-window behaviour the weights were tuned
    around) has changed. The same holds for preprocessing fields such as
    `crop_margin_voxels`, which never touch a tensor shape.
    """
    from dataclasses import replace

    saved_config = small_config(patch_size=(32, 32, 32))
    path = write_synthetic_checkpoint(tmp_path / "patch32.pth", saved_config)

    other_patch = replace(
        saved_config, patch=replace(saved_config.patch, selected_size=(64, 64, 64))
    )

    # The weights themselves are perfectly compatible.
    model = build_model(other_patch.model)
    state = torch.load(path, map_location=DEVICE, weights_only=False)["model_state_dict"]
    model.load_state_dict(state, strict=True)  # no error

    # But the loader refuses, because the configuration no longer matches.
    fresh = build_model(other_patch.model)
    with pytest.raises(CheckpointError, match="different model/patch/preprocessing"):
        load_brats_checkpoint(path, fresh, other_patch, device=DEVICE)

    # And the same is true for a preprocessing field with no shape consequence.
    other_margin = replace(
        saved_config,
        preprocessing=replace(saved_config.preprocessing, crop_margin_voxels=16),
    )
    fresh = build_model(other_margin.model)
    with pytest.raises(CheckpointError, match="different model/patch/preprocessing"):
        load_brats_checkpoint(path, fresh, other_margin, device=DEVICE)
