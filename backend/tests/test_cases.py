from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np

from tests.conftest import write_nifti

REQUIRED = ("t1", "t1ce", "t2", "flair")


def _file_tuple(path: Path, filename: str | None = None) -> tuple[str, BytesIO, str]:
    data = path.read_bytes()
    name = filename if filename is not None else path.name
    return (name, BytesIO(data), "application/octet-stream")


def _upload(client, case_id: str, modality: str, path: Path, filename: str | None = None):
    return client.post(
        f"/api/v1/cases/{case_id}/files/{modality}",
        files={"file": _file_tuple(path, filename)},
    )


def _create(client):
    response = client.post("/api/v1/cases")
    assert response.status_code == 201
    return response.json()["case_id"]


def test_create_case(client) -> None:
    response = client.post("/api/v1/cases")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "CREATED"
    assert body["modalities_present"] == []
    assert len(body["case_id"]) == 36


def test_unknown_case(client) -> None:
    missing = str(uuid4())
    response = client.get(f"/api/v1/cases/{missing}")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "CASE_NOT_FOUND"
    assert body["case_id"] == missing
    assert "Traceback" not in body["detail"]


def test_upload_valid_nifti(client, tmp_path: Path) -> None:
    case_id = _create(client)
    path = write_nifti(tmp_path / "t1.nii.gz")
    response = _upload(client, case_id, "t1", path)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UPLOADING"
    assert "t1" in body["modalities_present"]
    assert body["files"][0]["shape"] == [8, 8, 8]


def test_upload_invalid_file(client, tmp_path: Path) -> None:
    case_id = _create(client)
    junk = tmp_path / "t1.nii.gz"
    junk.write_bytes(b"this is not nifti")
    response = _upload(client, case_id, "t1", junk)
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_NIFTI"


def test_wrong_extension(client, tmp_path: Path) -> None:
    case_id = _create(client)
    path = write_nifti(tmp_path / "t1.nii.gz")
    response = _upload(client, case_id, "t1", path, filename="scan.png")
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_EXTENSION"


def test_missing_modality(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in ("t1", "t1ce", "flair"):
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    response = client.post(f"/api/v1/cases/{case_id}/complete")
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "MISSING_MODALITY"
    assert "T2" in body["detail"]
    assert client.get(f"/api/v1/cases/{case_id}").json()["status"] == "UPLOADING"


def test_duplicate_modality(client, tmp_path: Path) -> None:
    case_id = _create(client)
    path = write_nifti(tmp_path / "t1.nii.gz")
    assert _upload(client, case_id, "t1", path).status_code == 200
    response = _upload(client, case_id, "t1", path)
    assert response.status_code == 400
    assert response.json()["error"] == "DUPLICATE_MODALITY"


def test_shape_mismatch(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        shape = (8, 8, 7) if name == "flair" else (8, 8, 8)
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz", shape=shape))
    response = client.post(f"/api/v1/cases/{case_id}/complete")
    assert response.status_code == 422
    assert response.json()["error"] == "SHAPE_MISMATCH"
    assert client.get(f"/api/v1/cases/{case_id}").json()["status"] == "FAILED"


def test_affine_mismatch(client, tmp_path: Path) -> None:
    case_id = _create(client)
    shifted = np.eye(4)
    shifted[0, 3] = 12.0
    for name in REQUIRED:
        affine = shifted if name == "t2" else np.eye(4)
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz", affine=affine))
    response = client.post(f"/api/v1/cases/{case_id}/complete")
    assert response.status_code == 422
    assert response.json()["error"] == "AFFINE_MISMATCH"


def test_spacing_mismatch(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        zooms = (2.0, 2.0, 2.0) if name == "t1ce" else (1.0, 1.0, 1.0)
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz", zooms=zooms))
    response = client.post(f"/api/v1/cases/{case_id}/complete")
    assert response.status_code == 422
    assert response.json()["error"] == "SPACING_MISMATCH"


def test_invalid_segmentation_labels(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    bad = write_nifti(tmp_path / "seg.nii.gz", segmentation=True, extra_label=3)
    response = _upload(client, case_id, "seg", bad)
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_SEGMENTATION_LABEL"
    assert "3" in response.json()["detail"]


def test_case_size_limit(client, tmp_path: Path, monkeypatch, app) -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    get_settings.cache_clear()
    from app.main import create_app
    from fastapi.testclient import TestClient

    small_app = create_app()
    with TestClient(small_app) as limited:
        case_id = _create(limited)
        huge = write_nifti(tmp_path / "t1.nii.gz", shape=(80, 80, 80))
        response = _upload(limited, case_id, "t1", huge)
        assert response.status_code == 413
        assert response.json()["error"] == "CASE_TOO_LARGE"


def test_valid_complete_case(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    response = client.post(f"/api/v1/cases/{case_id}/complete")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "READY"
    assert body["validation"]["ready_for_prediction"] is True
    assert body["validation"]["ready_for_evaluation"] is False
    assert body["validation"]["spatial"]["shape_consistent"] is True
    assert body["inference"] == "not_started"


def test_optional_seg_absent(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    body = client.post(f"/api/v1/cases/{case_id}/complete").json()
    assert body["has_ground_truth"] is False
    assert body["validation"]["modalities"]["seg"] is False


def test_optional_seg_present(client, tmp_path: Path) -> None:
    case_id = _create(client)
    for name in REQUIRED:
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    _upload(client, case_id, "seg", write_nifti(tmp_path / "seg.nii.gz", segmentation=True))
    body = client.post(f"/api/v1/cases/{case_id}/complete").json()
    assert body["status"] == "READY"
    assert body["has_ground_truth"] is True
    assert body["validation"]["ready_for_evaluation"] is True


def test_path_traversal_attempt(client, tmp_path: Path) -> None:
    case_id = _create(client)
    path = write_nifti(tmp_path / "t1.nii.gz")
    response = _upload(client, case_id, "t1", path, filename="../../etc/passwd.nii.gz")
    assert response.status_code == 200
    from app.core.config import get_settings

    upload_root = Path(get_settings().upload_dir).resolve()
    expected = upload_root / "cases" / case_id / "input" / "t1.nii.gz"
    assert expected.is_file()
    assert not (upload_root / "etc" / "passwd.nii.gz").exists()
    assert "../" not in expected.as_posix()


def test_safe_storage_location(client, tmp_path: Path) -> None:
    case_id = _create(client)
    path = write_nifti(tmp_path / "patient_BraTS2021_00000_t1.nii.gz")
    _upload(client, case_id, "t1", path, filename="patient_BraTS2021_00000_t1.nii.gz")
    from app.core.config import get_settings

    stored = Path(get_settings().upload_dir).resolve() / "cases" / case_id / "input" / "t1.nii.gz"
    assert stored.is_file()
    metadata = Path(get_settings().upload_dir).resolve() / "cases" / case_id / "metadata" / "case.json"
    assert metadata.is_file()
    assert "BraTS2021_00000" not in stored.name


def test_case_state_transitions(client, tmp_path: Path) -> None:
    case_id = _create(client)
    assert client.get(f"/api/v1/cases/{case_id}").json()["status"] == "CREATED"
    _upload(client, case_id, "t1", write_nifti(tmp_path / "t1.nii.gz"))
    assert client.get(f"/api/v1/cases/{case_id}").json()["status"] == "UPLOADING"
    for name in ("t1ce", "t2", "flair"):
        _upload(client, case_id, name, write_nifti(tmp_path / f"{name}.nii.gz"))
    assert client.post(f"/api/v1/cases/{case_id}/complete").json()["status"] == "READY"


def test_predict_missing_t2(client, tmp_path: Path) -> None:
    files = {
        name: _file_tuple(write_nifti(tmp_path / f"{name}.nii.gz"))
        for name in ("t1", "t1ce", "flair")
    }
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 400
    assert response.json()["error"] == "MISSING_MODALITY"
    assert "T2" in response.json()["detail"]


def test_predict_valid_does_not_start_inference(client, tmp_path: Path) -> None:
    files = {name: _file_tuple(write_nifti(tmp_path / f"{name}.nii.gz")) for name in REQUIRED}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "READY"
    assert body["job_id"] is None
    assert body["inference"] == "not_started"


def test_evaluate_requires_seg(client, tmp_path: Path) -> None:
    files = {name: _file_tuple(write_nifti(tmp_path / f"{name}.nii.gz")) for name in REQUIRED}
    response = client.post("/api/v1/evaluate", files=files)
    assert response.status_code == 400
    assert response.json()["error"] == "MISSING_MODALITY"
    assert "SEG" in response.json()["detail"]
