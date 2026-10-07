import pytest
from app.core.config import Settings
from pydantic import ValidationError

def test_phase7_config_validates_pytorch_model_path():
    # If mock, model_path isn't required
    settings = Settings(inference_backend="mock")
    assert settings.inference_backend == "mock"
    
    # If pytorch, model_path is required
    with pytest.raises(ValidationError) as exc_info:
        Settings(inference_backend="pytorch", model_path="")
    assert "requires MODEL_PATH" in str(exc_info.value)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings(inference_backend="pytorch", model_path="/does/not/exist.pth")
    assert "does not exist" in str(exc_info.value)

def test_phase7_config_concurrency_constraints():
    # Workers must be >= max_concurrent_jobs
    with pytest.raises(ValidationError) as exc_info:
        Settings(inference_workers=1, max_concurrent_jobs=2)
    assert "cannot exceed INFERENCE_WORKERS" in str(exc_info.value)
