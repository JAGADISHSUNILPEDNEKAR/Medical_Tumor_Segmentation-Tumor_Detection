"""Phase 6 — end-to-end integration through the existing application.

Drives the real API: upload → validate → POST predict → 202 → job queue →
RealBraTSInferenceService → segmentation.nii.gz → MeasurementService →
MetricsService → ResultService → results / measurements / report endpoints →
viewer artifact stream.

The point of these tests is that NO Phase 1–5 code had to change: the same
routes, the same schemas, and the same frontend contract carry real inference.

The registered backend uses a SYNTHETIC checkpoint (random weights) at a
reduced width and patch size so the suite runs on a laptop CPU. Correctness of
the plumbing is asserted; segmentation quality is not, and cannot be, since the
weights are untrained.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from tests.conftest import create_ready_case_files, wait_for_job  # noqa: E402
from tests.phase6_helpers import small_config, write_synthetic_checkpoint  # noqa: E402

CASE_SHAPE = (16, 16, 16)


@pytest.fixture
def pytorch_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real application with a real PyTorch backend registered.

    `build_inference_service` is patched only to substitute the reduced test
    geometry; the service class, the loading path, and every downstream stage
    are the production ones.
    """
    config = small_config()
    checkpoint = write_synthetic_checkpoint(tmp_path / "fixture.pth", config)

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "500")
    monkeypatch.setenv("INFERENCE_BACKEND", "pytorch")
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    get_settings.cache_clear()

    from app.inference.pytorch_service import RealBraTSInferenceService

    def build_small(settings):
        service = RealBraTSInferenceService(
            Path(settings.model_path),
            device_preference="cpu",
            model_version=settings.model_version,
            config=config,
        )
        service.load()
        return service

    import app.main as main_module

    monkeypatch.setattr(main_module, "build_inference_service", build_small)

    application = main_module.create_app()
    yield application
    get_settings.cache_clear()


@pytest.fixture
def pytorch_client(pytorch_app) -> TestClient:
    with TestClient(pytorch_app) as client:
        yield client


# ── Health and provenance ────────────────────────────────────────────────────
def test_health_reports_a_loaded_pytorch_model(pytorch_client: TestClient) -> None:
    body = pytorch_client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["inference_source"] == "pytorch"


def test_model_info_reports_real_provenance_without_inventing_metrics(
    pytorch_client: TestClient,
) -> None:
    body = pytorch_client.get("/api/v1/model/info").json()

    assert body["model_loaded"] is True
    assert body["inference_source"] == "pytorch"
    assert body["checkpoint_id"] is not None
    assert body["model_version"] is not None
    assert body["num_classes"] == 4
    assert body["input_modalities"] == ["T1", "T1CE", "T2", "FLAIR"]
    # No fabricated headline accuracy, ever.
    assert body["headline_metrics"] is None
    assert body["details"]["architecture"].startswith("UNet3D")


def test_health_reports_unavailable_when_a_checkpoint_cannot_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'bad.db'}")
    monkeypatch.setenv("INFERENCE_BACKEND", "pytorch")

    # Needs a real file to pass Phase 7 startup validation
    missing = tmp_path / "missing.pth"
    missing.write_bytes(b"corrupt")
    monkeypatch.setenv("MODEL_PATH", str(missing))

    get_settings.cache_clear()

    from app.main import create_app

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False
        assert data["inference_source"] == "unavailable"

        info = client.get("/api/v1/model/info").json()
        assert info["model_loaded"] is False
        assert "could not be loaded" in info["message"]

        response = client.post("/api/v1/cases")
        case_id = response.json()["case_id"]
        predict = client.post(f"/api/v1/cases/{case_id}/predict")
        assert predict.status_code in (503, 409)

    get_settings.cache_clear()


# ── Full pipeline ────────────────────────────────────────────────────────────
def test_predict_runs_real_inference_end_to_end(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    files = create_ready_case_files(pytorch_client, tmp_path, shape=CASE_SHAPE)

    accepted = pytorch_client.post("/api/v1/predict", files=files)
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["status"] == "QUEUED"
    assert body["inference"] == "pytorch"
    assert "trained 3D U-Net" in body["message"]

    job = wait_for_job(pytorch_client, body["job_id"], timeout=120.0)
    assert job["status"] == "COMPLETED", job
    assert job["inference_source"] == "pytorch"
    assert job["model_version"] is not None
    assert job["result_id"]

    result = pytorch_client.get(f"/api/v1/results/{job['result_id']}").json()
    assert result["inference_source"] == "pytorch"
    assert result["segmentation"]["available"] is True
    assert result["provenance"]["synthetic"] is False
    assert result["provenance"]["checkpoint_id"] is not None
    assert result["measurements"]["synthetic"] is False


def test_measurements_endpoint_serves_real_inference_output(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    files = create_ready_case_files(pytorch_client, tmp_path, shape=CASE_SHAPE)
    body = pytorch_client.post("/api/v1/predict", files=files).json()
    job = wait_for_job(pytorch_client, body["job_id"], timeout=120.0)

    response = pytorch_client.get(
        f"/api/v1/results/{job['result_id']}/measurements"
    )
    assert response.status_code == 200
    measurements = response.json()

    assert measurements["synthetic"] is False
    labels = {region["label"] for region in measurements["regions"]}
    assert labels == {1, 2, 4}
    assert measurements["voxel_volume_mm3"] == pytest.approx(1.0)


def test_evaluate_computes_real_metrics_against_ground_truth(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    """Phase 5's metrics pipeline runs unchanged on the real model's output."""
    files = create_ready_case_files(
        pytorch_client, tmp_path, include_seg=True, shape=CASE_SHAPE
    )

    body = pytorch_client.post("/api/v1/evaluate", files=files).json()
    job = wait_for_job(pytorch_client, body["job_id"], timeout=120.0)
    assert job["status"] == "COMPLETED", job

    result = pytorch_client.get(f"/api/v1/results/{job['result_id']}").json()
    evaluation = result["evaluation"]

    assert evaluation["available"] is True
    assert evaluation["ground_truth_available"] is True
    per_class = {entry["label"]: entry for entry in evaluation["per_class"]}
    assert set(per_class) == {1, 2, 4}
    for entry in per_class.values():
        assert entry["dice"] is not None
        assert 0.0 <= entry["dice"]["value"] <= 1.0
        assert entry["hd95"] is not None


def test_report_endpoint_renders_real_provenance(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    files = create_ready_case_files(pytorch_client, tmp_path, shape=CASE_SHAPE)
    body = pytorch_client.post("/api/v1/predict", files=files).json()
    job = wait_for_job(pytorch_client, body["job_id"], timeout=120.0)

    response = pytorch_client.get(f"/api/v1/results/{job['result_id']}/report")
    assert response.status_code == 200
    report = response.json()

    assert report["provenance"]["inference_source"] == "pytorch"
    assert report["provenance"]["synthetic"] is False
    assert report["measurements"]["synthetic"] is False


def test_viewer_can_stream_the_real_segmentation_artifact(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    files = create_ready_case_files(pytorch_client, tmp_path, shape=CASE_SHAPE)
    body = pytorch_client.post("/api/v1/predict", files=files).json()
    wait_for_job(pytorch_client, body["job_id"], timeout=120.0)

    response = pytorch_client.get(
        f"/api/v1/cases/{body['case_id']}/artifacts/segmentation"
    )
    assert response.status_code == 200
    assert len(response.content) > 0

    artifact = tmp_path / "streamed.nii.gz"
    artifact.write_bytes(response.content)
    image = nib.load(str(artifact))
    data = np.asanyarray(image.dataobj)

    assert image.shape == CASE_SHAPE
    assert set(int(v) for v in np.unique(data)).issubset({0, 1, 2, 4})


def test_jobs_run_sequentially_on_the_single_worker(
    pytorch_client: TestClient, tmp_path: Path
) -> None:
    """The FIFO queue is preserved: no parallel GPU contention is introduced."""
    job_ids = []
    for index in range(3):
        files = create_ready_case_files(
            pytorch_client, tmp_path / f"case{index}", shape=CASE_SHAPE
        )
        body = pytorch_client.post("/api/v1/predict", files=files).json()
        assert body["status"] == "QUEUED"
        job_ids.append(body["job_id"])

    completed = [wait_for_job(pytorch_client, jid, timeout=180.0) for jid in job_ids]
    assert all(job["status"] == "COMPLETED" for job in completed), completed

    starts = [job["started_at"] for job in completed]
    ends = [job["completed_at"] for job in completed]
    for previous_end, next_start in zip(ends, starts[1:]):
        assert previous_end <= next_start, "jobs overlapped — the queue is not FIFO"


def test_mock_backend_still_works_unchanged(client: TestClient, tmp_path: Path) -> None:
    """Phase 3's behaviour is untouched when INFERENCE_BACKEND stays mock."""
    files = create_ready_case_files(client, tmp_path)
    body = client.post("/api/v1/predict", files=files).json()
    assert body["inference"] == "mock"

    job = wait_for_job(client, body["job_id"])
    assert job["status"] == "COMPLETED"
    assert job["inference_source"] == "mock"

    result = client.get(f"/api/v1/results/{job['result_id']}").json()
    assert result["inference_source"] == "mock"
    assert result["provenance"]["synthetic"] is True

    health = client.get("/api/v1/health").json()
    assert health["model_loaded"] is False
    assert health["inference_source"] == "mock"
