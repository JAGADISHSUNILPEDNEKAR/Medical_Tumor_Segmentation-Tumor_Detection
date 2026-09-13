from fastapi.testclient import TestClient


def test_health_returns_ok_with_mock_inference(client: TestClient) -> None:
    """Phase 3 evolution: inference_source changes from 'unavailable' to 'mock'.

    This is an intentional contract change, not a regression.
    Phase 2 returned inference_source='unavailable' because no inference was available.
    Phase 3 returns inference_source='mock' because MockInferenceService is registered.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["inference_source"] == "mock"
    assert "X-Request-ID" in response.headers


def test_health_echoes_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-req-001"})
    assert response.headers["X-Request-ID"] == "test-req-001"


def test_model_info_stub_has_no_invented_metrics(client: TestClient) -> None:
    """Phase 3: model_loaded remains False, inference_source is 'mock',
    no fake Dice/HD95 metrics are reported."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["inference_source"] == "mock"
    assert body["model_version"] is None
    assert body["checkpoint_id"] is None
    assert body["headline_metrics"] is None
    assert body["input_modalities"] == ["T1", "T1ce", "T2", "FLAIR"]
    assert "not available" in body["message"].lower() or "mock" in body["message"].lower()


def test_unknown_route_returns_structured_error(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "http_error"
    assert "request_id" in body
    assert "Traceback" not in body["detail"]
