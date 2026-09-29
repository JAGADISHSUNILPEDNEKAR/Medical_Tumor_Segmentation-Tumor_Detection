"""Phase 6 — RealBraTSInferenceService.

Covers the output artifact contract (shape, affine, spacing, labels, storage
location), provenance honesty, and device handling.

Every checkpoint used here is a SYNTHETIC fixture with random weights. These
tests assert pipeline correctness; none of them assert or imply segmentation
quality.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.core.errors import AppError  # noqa: E402
from app.inference.base import InferenceResult, InferenceService  # noqa: E402
from app.inference.brats.checkpoint import CheckpointError  # noqa: E402
from app.inference.pytorch_service import (  # noqa: E402
    RealBraTSInferenceService,
    resolve_device,
)
from tests.phase6_helpers import (  # noqa: E402
    small_config,
    write_case,
    write_synthetic_checkpoint,
)

CASE_SHAPE = (60, 64, 56)


@pytest.fixture
def loaded_service(tmp_path: Path) -> RealBraTSInferenceService:
    config = small_config()
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)
    service = RealBraTSInferenceService(checkpoint, device_preference="cpu", config=config)
    service.load()
    return service


@pytest.fixture
def case_dirs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "case" / "input"
    output_dir = tmp_path / "case" / "output"
    write_case(input_dir, shape=CASE_SHAPE)
    return input_dir, output_dir


# ── Protocol conformance and provenance ──────────────────────────────────────
def test_service_satisfies_the_inference_protocol(loaded_service) -> None:
    assert isinstance(loaded_service, InferenceService)


def test_provenance_reports_pytorch_and_not_synthetic(loaded_service) -> None:
    assert loaded_service.inference_source == "pytorch"
    assert loaded_service.synthetic is False
    assert loaded_service.model_loaded is True
    assert loaded_service.available is True


def test_model_loaded_is_false_before_load(tmp_path: Path) -> None:
    config = small_config()
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)
    service = RealBraTSInferenceService(checkpoint, device_preference="cpu", config=config)

    assert service.model_loaded is False
    assert service.available is False
    assert service.checkpoint_id is None


def test_predict_before_load_fails_loudly(tmp_path: Path, case_dirs) -> None:
    config = small_config()
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)
    service = RealBraTSInferenceService(checkpoint, device_preference="cpu", config=config)
    input_dir, output_dir = case_dirs

    with pytest.raises(CheckpointError, match="before the checkpoint was loaded"):
        service.predict(input_dir, output_dir, "case-1")


def test_model_version_is_derived_from_the_checkpoint_not_invented(
    loaded_service,
) -> None:
    version = loaded_service.model_version
    assert version is not None
    assert loaded_service.checkpoint_id in version
    assert version.startswith("unet3d-ep")


def test_configured_model_version_wins(tmp_path: Path) -> None:
    config = small_config()
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)
    service = RealBraTSInferenceService(
        checkpoint,
        device_preference="cpu",
        model_version="unet3d-v1-ep180",
        config=config,
    )
    service.load()
    assert service.model_version == "unet3d-v1-ep180"


def test_model_is_loaded_once_and_reused(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    model_identity = id(loaded_service._model)

    loaded_service.predict(input_dir, output_dir, "case-1")
    loaded_service.predict(input_dir, output_dir, "case-1")

    assert id(loaded_service._model) == model_identity


# ── Device handling ──────────────────────────────────────────────────────────
def test_auto_device_resolves_without_a_gpu() -> None:
    device = resolve_device("auto")
    assert device.type in {"cuda", "cpu"}
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_explicit_cpu_is_honoured() -> None:
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present on this host")
def test_requesting_cuda_without_cuda_fails_clearly() -> None:
    with pytest.raises(CheckpointError, match="no CUDA device"):
        resolve_device("cuda")


def test_model_is_in_eval_mode_after_load(loaded_service) -> None:
    assert loaded_service._model.training is False


# ── Output artifact ──────────────────────────────────────────────────────────
def test_predict_returns_the_inference_result_contract(loaded_service, case_dirs):
    input_dir, output_dir = case_dirs

    result = loaded_service.predict(input_dir, output_dir, "case-1")

    assert isinstance(result, InferenceResult)
    assert result.segmentation_path == "segmentation.nii.gz"
    # A filename, never an absolute path leaked to the API layer.
    assert Path(result.segmentation_path).name == result.segmentation_path
    assert not Path(result.segmentation_path).is_absolute()


def test_segmentation_is_written_inside_the_case_output_directory(
    loaded_service, case_dirs
) -> None:
    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")
    written = output_dir / result.segmentation_path
    assert written.is_file()
    assert written.parent == output_dir


def test_output_shape_matches_the_input_volume(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")

    seg = nib.load(str(output_dir / result.segmentation_path))
    assert seg.shape == CASE_SHAPE


def test_output_affine_and_spacing_match_the_input(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    reference = nib.load(str(input_dir / "t1.nii.gz"))

    result = loaded_service.predict(input_dir, output_dir, "case-1")
    seg = nib.load(str(output_dir / result.segmentation_path))

    assert np.allclose(seg.affine, reference.affine)
    assert not np.allclose(seg.affine, np.eye(4)), "must not fall back to identity"
    assert tuple(seg.header.get_zooms()[:3]) == pytest.approx(
        tuple(reference.header.get_zooms()[:3])
    )
    assert nib.aff2axcodes(seg.affine) == nib.aff2axcodes(reference.affine)


def test_output_contains_only_brats_labels(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")

    data = np.asanyarray(nib.load(str(output_dir / result.segmentation_path)).dataobj)
    labels = set(int(v) for v in np.unique(data))

    assert labels.issubset({0, 1, 2, 4})
    assert 3 not in labels


def test_output_passes_the_phase_2_segmentation_validator(
    loaded_service, case_dirs
) -> None:
    """The artifact the viewer loads must satisfy the same rules as an upload."""
    from app.validation.nifti import inspect_nifti

    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")

    info = inspect_nifti(
        output_dir / result.segmentation_path, is_segmentation=True, case_id="case-1"
    )

    assert info.shape == CASE_SHAPE
    assert set(info.unique_labels or []).issubset({0, 1, 2, 4})


def test_result_json_sidecar_is_written(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    loaded_service.predict(input_dir, output_dir, "case-1")

    payload = json.loads((output_dir / "result.json").read_text())
    assert payload["inference_source"] == "pytorch"
    assert payload["synthetic"] is False


def test_metadata_records_measured_pipeline_facts(loaded_service, case_dirs) -> None:
    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")
    metadata = result.metadata

    assert metadata["inference_source"] == "pytorch"
    assert metadata["synthetic"] is False
    assert metadata["channel_order"] == ["t1", "t1ce", "t2", "flair"]
    assert metadata["input_shape"] == list(CASE_SHAPE)
    assert metadata["output_shape"] == list(CASE_SHAPE)
    assert metadata["patch_size"] == [32, 32, 32]
    assert metadata["sliding_window_weighting"] == "gaussian"
    assert metadata["sliding_window_overlap"] == 0.5
    assert metadata["num_patches"] >= 1
    assert metadata["postprocessing"]["connected_component_min_voxels"] == 50
    assert metadata["label_mapping"]["internal_to_brats"]["3"] == 4
    assert set(metadata["output_labels"]).issubset({0, 1, 2, 4})
    assert metadata["device"] == "cpu"
    assert metadata["inference_seconds"] > 0
    assert metadata["checkpoint_id"] == loaded_service.checkpoint_id
    assert metadata["model_parameters"] > 0


def test_synthetic_fixture_checkpoint_is_flagged_in_metadata(
    loaded_service, case_dirs
) -> None:
    """A random-weight fixture must never be presentable as trained output."""
    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")

    assert result.metadata["checkpoint_is_synthetic_fixture"] is True
    assert "SYNTHETIC" in result.metadata["description"]


def test_inference_is_deterministic(loaded_service, tmp_path: Path) -> None:
    input_dir = tmp_path / "case" / "input"
    write_case(input_dir, shape=CASE_SHAPE)
    first_out = tmp_path / "out_a"
    second_out = tmp_path / "out_b"

    a = loaded_service.predict(input_dir, first_out, "case-1")
    b = loaded_service.predict(input_dir, second_out, "case-1")

    first = np.asanyarray(nib.load(str(first_out / a.segmentation_path)).dataobj)
    second = np.asanyarray(nib.load(str(second_out / b.segmentation_path)).dataobj)
    assert np.array_equal(first, second)


# ── Input validation ─────────────────────────────────────────────────────────
def test_missing_modality_is_rejected(loaded_service, tmp_path: Path) -> None:
    input_dir = tmp_path / "partial" / "input"
    write_case(input_dir, shape=CASE_SHAPE, modalities=("t1", "t1ce", "t2"))

    with pytest.raises(RuntimeError, match="flair"):
        loaded_service.predict(input_dir, tmp_path / "out", "case-1")


def test_mismatched_geometry_is_rejected_by_the_phase_2_validator(
    loaded_service, tmp_path: Path
) -> None:
    input_dir = tmp_path / "bad" / "input"
    write_case(input_dir, shape=CASE_SHAPE)
    odd = np.zeros((40, 40, 40), dtype=np.float32)
    odd[5:35, 5:35, 5:35] = 100.0
    nib.save(nib.Nifti1Image(odd, np.eye(4)), str(input_dir / "t2.nii.gz"))

    with pytest.raises(AppError) as excinfo:
        loaded_service.predict(input_dir, tmp_path / "out", "case-1")
    assert excinfo.value.error == "SHAPE_MISMATCH"


def test_invalid_nifti_is_rejected(loaded_service, tmp_path: Path) -> None:
    input_dir = tmp_path / "broken" / "input"
    write_case(input_dir, shape=CASE_SHAPE)
    (input_dir / "t1ce.nii.gz").write_bytes(b"not a nifti file at all")

    with pytest.raises(AppError) as excinfo:
        loaded_service.predict(input_dir, tmp_path / "out", "case-1")
    assert excinfo.value.error == "INVALID_NIFTI"


def test_non_finite_voxels_are_rejected(loaded_service, tmp_path: Path) -> None:
    input_dir = tmp_path / "nan" / "input"
    write_case(input_dir, shape=CASE_SHAPE)
    reference = nib.load(str(input_dir / "t1.nii.gz"))
    data = np.asanyarray(reference.dataobj, dtype=np.float32).copy()
    data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, reference.affine), str(input_dir / "t1.nii.gz"))

    with pytest.raises(AppError) as excinfo:
        loaded_service.predict(input_dir, tmp_path / "out", "case-1")
    assert excinfo.value.error == "INVALID_NIFTI"


# ── Small-volume handling ────────────────────────────────────────────────────
def test_volume_smaller_than_the_patch_still_produces_full_geometry(
    tmp_path: Path,
) -> None:
    config = small_config(patch_size=(32, 32, 32))
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)
    service = RealBraTSInferenceService(checkpoint, device_preference="cpu", config=config)
    service.load()

    small_shape = (24, 26, 22)
    input_dir = tmp_path / "small" / "input"
    write_case(input_dir, shape=small_shape)
    output_dir = tmp_path / "small" / "output"

    result = service.predict(input_dir, output_dir, "case-small")

    seg = nib.load(str(output_dir / result.segmentation_path))
    assert seg.shape == small_shape
    assert result.metadata["num_patches"] == 1


def test_measurements_consume_the_real_segmentation(loaded_service, case_dirs) -> None:
    """The Phase 5 measurement service must read the artifact unchanged."""
    from app.services.measurement_service import MeasurementService

    input_dir, output_dir = case_dirs
    result = loaded_service.predict(input_dir, output_dir, "case-1")

    seg_img = nib.load(str(output_dir / result.segmentation_path))
    seg_data = np.asanyarray(seg_img.dataobj)
    spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:3])

    measurements = MeasurementService().compute_measurements(
        segmentation=seg_data,
        spacing=spacing,
        is_synthetic=result.metadata["synthetic"],
        description=result.metadata["description"],
    )

    assert measurements.synthetic is False
    assert measurements.voxel_spacing_mm == list(spacing)
    assert {r.label for r in measurements.regions or []} == {1, 2, 4}
    assert measurements.foreground_voxels == int(np.count_nonzero(seg_data))
