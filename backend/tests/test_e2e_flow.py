import time
from fastapi.testclient import TestClient
from app.core.constants import CaseStatus, JobStatus
from tests.conftest import create_ready_case_files, wait_for_job

def test_complete_e2e_flow(client: TestClient, tmp_path, monkeypatch):
    """
    Exercise the complete application contract from case creation to report artifact access.
    Uses MockInferenceService to remain deterministic and CPU-compatible.
    """
    # Force Mock backend for this test
    monkeypatch.setenv("INFERENCE_BACKEND", "mock")
    
    # 1. Create Case
    create_resp = client.post("/api/v1/cases")
    assert create_resp.status_code == 201
    case_id = create_resp.json()["case_id"]
    assert create_resp.json()["status"] == CaseStatus.CREATED
    
    # 2. Upload T1, T1ce, T2, FLAIR
    files = create_ready_case_files(client, tmp_path, shape=(32, 32, 32))
    for modality, file_tuple in files.items():
        resp = client.post(
            f"/api/v1/cases/{case_id}/files/{modality}", 
            files={"file": file_tuple}
        )
        assert resp.status_code == 200
        
    case_resp = client.get(f"/api/v1/cases/{case_id}")
    assert set(case_resp.json()["modalities_present"]) == {"t1", "t1ce", "t2", "flair"}
    assert case_resp.json()["status"] == CaseStatus.VALIDATING
    
    # 3. Complete / Validate Case
    complete_resp = client.post(f"/api/v1/cases/{case_id}/complete")
    assert complete_resp.status_code == 200
    assert complete_resp.json()["status"] == CaseStatus.READY
    
    # 4. Create Prediction Job
    predict_resp = client.post(f"/api/v1/cases/{case_id}/predict")
    assert predict_resp.status_code == 202
    job_id = predict_resp.json()["job_id"]
    assert predict_resp.json()["inference"] == "mock"
    
    # 5. Wait for Completion
    job = wait_for_job(client, job_id, timeout=15.0)
    assert job["status"] == JobStatus.COMPLETED
    assert job["progress"] == 100
    result_id = job["result_id"]
    assert result_id is not None
    
    # 6. Obtain Result
    result_resp = client.get(f"/api/v1/results/{result_id}")
    assert result_resp.status_code == 200
    result_data = result_resp.json()
    assert result_data["status"] == "COMPLETED"
    assert result_data["inference_source"] == "mock"
    assert result_data["segmentation"]["available"] is True
    
    # 7. Obtain Measurements
    meas_resp = client.get(f"/api/v1/results/{result_id}/measurements")
    assert meas_resp.status_code == 200
    meas_data = meas_resp.json()
    assert meas_data["synthetic"] is True
    # The mock draws an ellipsoid. It should have foreground voxels.
    assert meas_data["foreground_voxels"] > 0
    
    # 8. Obtain Report
    report_resp = client.get(f"/api/v1/results/{result_id}/report")
    assert report_resp.status_code == 200
    report_data = report_resp.json()
    assert report_data["result_id"] == result_id
    assert report_data["case_id"] == case_id
    assert "provenance" in report_data
    assert report_data["provenance"]["inference_source"] == "mock"
    
    # 9. Obtain Segmentation Artifact
    artifact_resp = client.get(f"/api/v1/cases/{case_id}/artifacts/segmentation.nii.gz")
    assert artifact_resp.status_code == 200
    assert artifact_resp.headers["content-type"] == "application/gzip"
    assert len(artifact_resp.content) > 0
