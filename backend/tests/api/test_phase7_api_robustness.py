def test_phase7_health_response_schema(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "inference_source" in data

def test_phase7_model_info_schema(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "inference_source" in data
    assert "message" in data
    # Phase 7 new fields
    assert "backend" in data
    assert "device" in data
    assert "parameter_count" in data
    assert "compat_fingerprint" in data
    assert "patch_size" in data

def test_phase7_missing_artifact(client):
    case_resp = client.post("/api/v1/cases")
    case_id = case_resp.json()["case_id"]
    
    # Try to download missing artifact
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/segmentation")
    assert response.status_code == 404
    # detail is a dictionary {"error": "...", "message": "..."} OR just a string.
    # AppError returns {"error": "ARTIFACT_NOT_FOUND", "message": "Artifact 'segmentation' is not available."} if using custom error handler.
    assert "Artifact 'segmentation' is not available" in str(response.json())

def test_phase7_invalid_artifact_name(client):
    case_resp = client.post("/api/v1/cases")
    case_id = case_resp.json()["case_id"]
    
    # Try to download unallowed artifact
    response = client.get(f"/api/v1/cases/{case_id}/artifacts/invalid_name")
    assert response.status_code == 400
    assert "Unknown artifact" in str(response.json())
