from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.constants import JobStatus
from tests.conftest import create_ready_case_files, wait_for_job


def test_predict_creates_job_and_returns_202(client: TestClient, tmp_path: Path) -> None:
    files = create_ready_case_files(client, tmp_path)
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "QUEUED"
    assert len(body["job_id"]) == 36
    assert body["inference"] == "mock"


def test_evaluate_creates_job_and_returns_202(client: TestClient, tmp_path: Path) -> None:
    files = create_ready_case_files(client, tmp_path, include_seg=True)
    response = client.post("/api/v1/evaluate", files=files)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "QUEUED"
    assert len(body["job_id"]) == 36
    assert body["inference"] == "mock"


def test_job_lifecycle_to_completion(client: TestClient, tmp_path: Path) -> None:
    """Test full queued -> completed lifecycle with result generation."""
    files = create_ready_case_files(client, tmp_path)
    predict_resp = client.post("/api/v1/predict", files=files)
    job_id = predict_resp.json()["job_id"]
    case_id = predict_resp.json()["case_id"]

    # Poll until done
    final_state = wait_for_job(client, job_id)
    assert final_state["status"] == "COMPLETED"
    assert final_state["progress"] == 100
    assert final_state["inference_source"] == "mock"
    assert final_state["result_id"] is not None

    # Fetch results
    result_id = final_state["result_id"]
    result_resp = client.get(f"/api/v1/results/{result_id}")
    assert result_resp.status_code == 200
    result_data = result_resp.json()

    assert result_data["job_id"] == job_id
    assert result_data["case_id"] == case_id
    assert result_data["status"] == "COMPLETED"
    assert result_data["inference_source"] == "mock"
    assert result_data["segmentation"]["available"] is True
    assert result_data["measurements"]["synthetic"] is True
    assert "foreground_volume_cm3" in result_data["measurements"]

    # Phase 3: evaluation is not available
    assert result_data["evaluation"]["available"] is False


def test_job_unknown(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error"] == "JOB_NOT_FOUND"


def test_result_unknown(client: TestClient) -> None:
    response = client.get("/api/v1/results/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error"] == "RESULT_NOT_FOUND"


def test_mock_inference_deterministic_and_safe_output(client: TestClient, tmp_path: Path) -> None:
    """Verify mock inference generates valid NIfTI artifacts on disk."""
    files = create_ready_case_files(client, tmp_path, shape=(10, 10, 10))
    predict_resp = client.post("/api/v1/predict", files=files)
    job_id = predict_resp.json()["job_id"]
    case_id = predict_resp.json()["case_id"]

    wait_for_job(client, job_id)

    # Check filesystem
    from app.core.config import get_settings
    upload_root = Path(get_settings().upload_dir).resolve()
    output_dir = upload_root / "cases" / case_id / "output"
    
    seg_path = output_dir / "segmentation.nii.gz"
    assert seg_path.exists(), "Segmentation artifact was not written"
    
    meta_path = output_dir / "result.json"
    assert meta_path.exists(), "Result metadata was not written"

    import nibabel as nib
    import numpy as np
    
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)
    
    assert data.shape == (10, 10, 10)
    
    # Check that ONLY BraTS labels exist in the output (0, 1, 2, 4)
    unique_labels = set(np.unique(data))
    assert unique_labels.issubset({0, 1, 2, 4})
    # Ensure internal label 3 was remapped or doesn't exist
    assert 3 not in unique_labels


def test_predict_failure_handling(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    """Test that if inference raises an exception, the job gracefully fails."""
    # We patch MockInferenceService to raise an error
    from app.inference.mock import MockInferenceService
    
    def failing_predict(*args, **kwargs):
        raise RuntimeError("Simulated failure")
        
    monkeypatch.setattr(MockInferenceService, "predict", failing_predict)

    files = create_ready_case_files(client, tmp_path)
    predict_resp = client.post("/api/v1/predict", files=files)
    job_id = predict_resp.json()["job_id"]

    final_state = wait_for_job(client, job_id)
    assert final_state["status"] == "FAILED"
    assert final_state["error_code"] == "INFERENCE_FAILED"
    assert "Mock inference failed" in final_state["error_message"]
    # Internal tracebacks shouldn't be exposed
    assert "Simulated failure" not in final_state["error_message"]


def test_predict_existing_case_runs_mock_inference(
    client: TestClient, tmp_path: Path
) -> None:
    """POST /cases/{id}/predict on a READY case enqueues a job and completes."""
    files = create_ready_case_files(client, tmp_path)
    case_id = client.post("/api/v1/cases").json()["case_id"]
    for modality, payload in files.items():
        upload = client.post(
            f"/api/v1/cases/{case_id}/files/{modality}", files={"file": payload}
        )
        assert upload.status_code == 200
    assert client.post(f"/api/v1/cases/{case_id}/complete").json()["status"] == "READY"

    response = client.post(f"/api/v1/cases/{case_id}/predict")
    assert response.status_code == 202
    body = response.json()
    assert body["case_id"] == case_id
    assert body["status"] == JobStatus.QUEUED.value
    assert body["inference"] == "mock"

    job = wait_for_job(client, body["job_id"])
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["result_id"]


def test_predict_existing_case_not_ready_returns_409(client: TestClient) -> None:
    """A case with no uploads must be refused with an actionable error, not a 500."""
    case_id = client.post("/api/v1/cases").json()["case_id"]

    response = client.post(f"/api/v1/cases/{case_id}/predict")

    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "INVALID_CASE_STATE"
    assert body["case_id"] == case_id
    assert "READY" in body["detail"]
    # The message must be actionable and must not leak internals.
    assert "Traceback" not in body["detail"]


def test_predict_unknown_case_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/cases/does-not-exist/predict")
    assert response.status_code == 404
    assert response.json()["error"] == "CASE_NOT_FOUND"
