from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from tests.conftest import create_ready_case_files, wait_for_job, write_nifti

REQUIRED = ("t1", "t1ce", "t2", "flair")


def _create_ready_case(client: TestClient, tmp_path: Path) -> str:
    case_id = client.post("/api/v1/cases").json()["case_id"]
    for name in REQUIRED:
        path = write_nifti(tmp_path / f"{name}.nii.gz")
        response = client.post(
            f"/api/v1/cases/{case_id}/files/{name}",
            files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
        )
        assert response.status_code == 200
    assert client.post(f"/api/v1/cases/{case_id}/complete").status_code == 200
    return case_id


def test_artifact_known_case_returns_nifti_bytes(client: TestClient, tmp_path: Path) -> None:
    case_id = _create_ready_case(client, tmp_path)
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/flair")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/gzip")
    stored = Path(client.app.state.job_queue._upload_root) / "cases" / case_id / "input" / "flair.nii.gz"
    assert response.content == stored.read_bytes()
    assert "uploads" not in response.text.lower() if response.headers.get("content-type", "").startswith("application/json") else True


def test_artifact_unknown_case_404(client: TestClient) -> None:
    missing = str(uuid4())
    response = client.get(f"/api/v1/cases/{missing}/artifacts/t1")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "CASE_NOT_FOUND"
    assert "Traceback" not in body["detail"]


def test_artifact_unknown_name_rejected(client: TestClient, tmp_path: Path) -> None:
    case_id = _create_ready_case(client, tmp_path)
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/secret")
    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_ARTIFACT"


def test_artifact_path_traversal_rejected(client: TestClient, tmp_path: Path) -> None:
    case_id = _create_ready_case(client, tmp_path)
    for name in ("%2e%2e", "../etc/passwd", "..%2Fetc", "t1/../../etc/passwd", "/etc/passwd"):
        response = client.get(f"/api/v1/cases/{case_id}/artifacts/{name}")
        assert response.status_code in {400, 404, 422}
        if response.status_code == 400:
            assert response.json()["error"] == "INVALID_ARTIFACT"
        body = response.json()
        assert "Traceback" not in str(body)


def test_missing_segmentation_artifact(client: TestClient, tmp_path: Path) -> None:
    case_id = _create_ready_case(client, tmp_path)
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/segmentation")
    assert response.status_code == 404
    assert response.json()["error"] == "ARTIFACT_NOT_FOUND"


def test_segmentation_artifact_after_mock_inference(client: TestClient, tmp_path: Path) -> None:
    files = create_ready_case_files(client, tmp_path)
    predict = client.post("/api/v1/predict", files=files)
    case_id = predict.json()["case_id"]
    wait_for_job(client, predict.json()["job_id"])
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/segmentation")
    assert response.status_code == 200
    assert len(response.content) > 100


def test_artifact_endpoint_does_not_expose_database_paths(client: TestClient, tmp_path: Path) -> None:
    case_id = _create_ready_case(client, tmp_path)
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/t1")
    assert response.status_code == 200
    assert b"sqlite" not in response.content[:64]
    disposition = response.headers.get("content-disposition", "")
    assert ".." not in disposition
    assert "t1.nii.gz" in disposition
