from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_without_loaded_model() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["inference_source"] == "unavailable"
    assert "X-Request-ID" in response.headers


def test_health_echoes_request_id() -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-req-001"})
    assert response.headers["X-Request-ID"] == "test-req-001"


def test_model_info_stub_has_no_invented_metrics() -> None:
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["checkpoint_id"] is None
    assert body["headline_metrics"] is None
    assert body["input_modalities"] == ["T1", "T1ce", "T2", "FLAIR"]
    assert "not available" in body["message"].lower()


def test_unknown_route_returns_structured_error() -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "http_error"
    assert "request_id" in body
    assert "Traceback" not in body["detail"]
