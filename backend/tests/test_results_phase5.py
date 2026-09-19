import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_ready_case_files, wait_for_job


def test_measurements_and_metrics_endpoints(client: TestClient, tmp_path):
    # 1. Create a case with ground truth to trigger metrics
    files = create_ready_case_files(client, tmp_path, include_seg=True)
    response = client.post("/api/v1/evaluate", files=files)
    assert response.status_code == 202
    data = response.json()
    job_id = data["job_id"]
    
    # Wait for completion
    job = wait_for_job(client, job_id)
    assert job["status"] == "COMPLETED"
    result_id = job["result_id"]
    
    # 2. Get measurements
    meas_resp = client.get(f"/api/v1/results/{result_id}/measurements")
    assert meas_resp.status_code == 200
    measurements = meas_resp.json()
    assert measurements["synthetic"] is True
    assert "regions" in measurements
    
    # 3. Get report
    report_resp = client.get(f"/api/v1/results/{result_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["measurements"] == measurements
    
    # Check evaluation
    eval = report["evaluation"]
    assert eval["available"] is True
    assert eval["ground_truth_available"] is True
    
    # Mock inference produces synthetic seg. The ground truth (seg) in conftest
    # has different coordinates. So we expect some Dice/HD95.
    assert eval["mean_dice"] is not None
    assert eval["mean_hd95"] is not None


def test_missing_result_returns_404(client: TestClient):
    response = client.get("/api/v1/results/nonexistent/report")
    assert response.status_code == 404
