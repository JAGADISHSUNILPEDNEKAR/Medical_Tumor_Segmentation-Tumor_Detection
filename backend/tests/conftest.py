from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "500")
    get_settings.cache_clear()
    from app.main import create_app

    application = create_app()
    yield application
    get_settings.cache_clear()


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def write_nifti(
    path: Path,
    *,
    shape: tuple[int, int, int] = (8, 8, 8),
    affine: np.ndarray | None = None,
    data: np.ndarray | None = None,
    zooms: tuple[float, float, float] | None = None,
    segmentation: bool = False,
    extra_label: int | None = None,
) -> Path:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    if data is None:
        if segmentation:
            volume = np.zeros(shape, dtype=np.int16)
            volume[1:4, 1:4, 1:4] = 1
            volume[2:5, 2:5, 2:5] = 2
            volume[3:6, 3:6, 3:6] = 4
            if extra_label is not None:
                volume[0, 0, 0] = extra_label
            data = volume
        else:
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image = nib.Nifti1Image(data, affine)
    if zooms is not None:
        image.header.set_zooms(zooms)
    nib.save(image, str(path))
    return path
