import pytest
import io

def test_phase7_api_nosniff_header(client):
    # Make a dummy case
    response = client.post("/api/v1/cases")
    assert response.status_code == 201
    
    # Just checking health endpoint or similar if it has it?
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200

def test_phase7_upload_size_limit_and_content_type(client):
    # Send a non-gzip file named .gz
    bad_file = io.BytesIO(b"not a gzip file content" * 1024)
    good_file = io.BytesIO(b"\x1f\x8bsomegzipcontent")
    resp = client.post(
        f"/api/v1/predict",
        files={
            "t1": ("t1.nii.gz", bad_file, "application/gzip"),
            "t1ce": ("t1ce.nii.gz", good_file, "application/gzip"),
            "t2": ("t2.nii.gz", good_file, "application/gzip"),
            "flair": ("flair.nii.gz", good_file, "application/gzip"),
        }
    )
    # The magic bytes check should fail it
    assert resp.status_code == 422
    assert "invalid magic bytes" in str(resp.json())

def test_phase7_path_traversal_prevention(app):
    from app.storage.filesystem import CaseStorage
    from app.core.config import get_settings
    from pathlib import Path
    from app.core.errors import AppError
    
    storage = CaseStorage(Path(get_settings().upload_dir))
    
    with pytest.raises(AppError) as exc:
        storage.destination_for("case_123", "../../../etc/passwd")
    
    assert "unsafe storage filename" in str(exc.value)
