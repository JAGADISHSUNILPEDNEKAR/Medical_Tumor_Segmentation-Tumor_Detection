"""Phase 6 — backend selection and startup behaviour.

INFERENCE_BACKEND picks the implementation. The invariants that matter:
mock stays fully functional, a real backend that cannot load never silently
degrades to mock, and /health never claims a model is loaded when it is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.inference.factory import build_inference_service
from app.inference.mock import MockInferenceService
from app.inference.unavailable import (
    InferenceUnavailableError,
    UnavailableInferenceService,
)


def make_settings(**overrides) -> Settings:
    base = {
        "inference_backend": "mock",
        "model_path": None,
        "model_version": None,
        "inference_device": "cpu",
    }
    base.update(overrides)
    return Settings(**base)


def test_default_backend_is_mock() -> None:
    assert Settings().inference_backend == "mock"


def test_mock_backend_builds_the_mock_service() -> None:
    service = build_inference_service(make_settings(inference_backend="mock"))
    assert isinstance(service, MockInferenceService)
    assert service.inference_source == "mock"
    assert service.model_loaded is False
    assert service.available is True


def test_backend_selection_is_case_insensitive() -> None:
    service = build_inference_service(make_settings(inference_backend="MOCK"))
    assert isinstance(service, MockInferenceService)


def test_unknown_backend_is_unavailable_not_silently_mock() -> None:
    service = build_inference_service(make_settings(inference_backend="tensorflow"))
    assert isinstance(service, UnavailableInferenceService)
    assert "not supported" in service.reason
    # The client-facing reason must not echo server configuration values.
    assert "tensorflow" not in service.reason
    assert "tensorflow" in service.detail
    assert service.inference_source == "unavailable"


def test_pytorch_backend_without_a_checkpoint_is_unavailable() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc:
        make_settings(inference_backend="pytorch", model_path=None)
    assert "requires MODEL_PATH" in str(exc.value)


def test_pytorch_backend_with_a_missing_checkpoint_is_unavailable(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc:
        make_settings(
            inference_backend="pytorch", model_path=str(tmp_path / "nope.pth")
        )
    assert "does not exist" in str(exc.value)


def test_pytorch_backend_with_a_corrupt_checkpoint_is_unavailable(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"definitely not a checkpoint")

    service = build_inference_service(
        make_settings(inference_backend="pytorch", model_path=str(corrupt))
    )

    assert isinstance(service, UnavailableInferenceService)
    assert service.model_loaded is False


def test_unavailable_service_refuses_to_predict(tmp_path: Path) -> None:
    service = UnavailableInferenceService("no checkpoint registered")
    with pytest.raises(InferenceUnavailableError, match="no checkpoint registered"):
        service.predict(tmp_path, tmp_path, "case-1")


def test_unavailable_reason_never_leaks_the_checkpoint_path(tmp_path: Path) -> None:
    """MODEL_PATH is a server path; it must not reach an HTTP client."""
    pytest.importorskip("torch")
    secret_path = tmp_path / "very" / "private" / "best_model.pth"
    # Create the fake secret path so config validation passes
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.touch()

    # But make it unreadable/corrupt so the backend fails to load
    service = build_inference_service(
        make_settings(inference_backend="pytorch", model_path=str(secret_path))
    )

    assert isinstance(service, UnavailableInferenceService)
    assert str(secret_path) not in service.reason
    assert "private" not in service.reason
    # The operator diagnostic keeps the detail.
    assert "best_model.pth" in service.detail


def test_pytorch_backend_with_an_incompatible_checkpoint_is_unavailable(
    tmp_path: Path,
) -> None:
    """A checkpoint from a different architecture must not load."""
    pytest.importorskip("torch")
    from tests.phase6_helpers import small_config, write_synthetic_checkpoint

    narrow = write_synthetic_checkpoint(tmp_path / "narrow.pth", small_config())

    service = build_inference_service(
        make_settings(inference_backend="pytorch", model_path=str(narrow))
    )

    assert isinstance(service, UnavailableInferenceService)
    assert service.model_loaded is False


@pytest.mark.slow
def test_pytorch_backend_loads_a_full_size_trained_geometry_checkpoint(
    tmp_path: Path,
) -> None:
    """End-to-end factory success at the real trained width (18.7M parameters).

    No forward pass is run — this asserts that a checkpoint written in the
    notebook's format at the trained configuration is accepted, the fingerprint
    matches, and the service reports itself loaded.
    """
    pytest.importorskip("torch")
    from app.inference.brats.config import (
        EXPECTED_COMPAT_FINGERPRINT,
        TRAINING_CONFIG,
    )
    from tests.phase6_helpers import write_synthetic_checkpoint

    checkpoint = write_synthetic_checkpoint(
        tmp_path / "full.pth", TRAINING_CONFIG, epoch=19
    )

    service = build_inference_service(
        make_settings(inference_backend="pytorch", model_path=str(checkpoint))
    )

    assert not isinstance(service, UnavailableInferenceService)
    assert service.inference_source == "pytorch"
    assert service.model_loaded is True
    assert service.available is True
    described = service.describe()
    assert described["parameters"] == 18_774_756
    assert described["patch_size"] == [128, 128, 128]
    assert described["service_compat_fingerprint"] == EXPECTED_COMPAT_FINGERPRINT
    assert described["checkpoint"]["fingerprint_verified"] is True
