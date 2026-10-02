from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Environment-driven configuration. Never hardcode host paths or secrets."""

    model_config = SettingsConfigDict(
        env_file=(_REPO_DIR / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    from pydantic import model_validator
    
    @model_validator(mode="after")
    def _validate_startup_config(self) -> "Settings":
        if self.inference_backend.lower() == "pytorch":
            if not self.model_path:
                raise ValueError("INFERENCE_BACKEND=pytorch requires MODEL_PATH to be set")
            p = Path(self.model_path)
            if not p.is_absolute():
                # Resolve relative to repo root since that's where we usually run from
                p = (_REPO_DIR / p).resolve()
            if not p.exists():
                raise ValueError(f"MODEL_PATH {self.model_path} does not exist")
                
        # Validate concurrent jobs vs workers constraint
        if self.max_concurrent_jobs > self.inference_workers:
            raise ValueError(f"MAX_CONCURRENT_JOBS ({self.max_concurrent_jobs}) cannot exceed INFERENCE_WORKERS ({self.inference_workers})")
        
        return self

    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    model_path: str | None = None
    model_version: str | None = None

    # ── Inference backend (Phase 6) ──────────────────────────────────────────
    # "mock"    -> MockInferenceService (synthetic geometry, no checkpoint)
    # "pytorch" -> RealBraTSInferenceService (requires MODEL_PATH)
    inference_backend: str = "mock"
    # "auto" resolves to CUDA when torch reports a device, otherwise CPU.
    inference_device: str = "auto"
    # Refuse a checkpoint whose recorded model/patch/preprocessing fingerprint
    # differs from the one this service reconstructs. Turning this off does NOT
    # relax state_dict loading, which is always strict.
    inference_strict_fingerprint: bool = True
    upload_dir: str = "./data/uploads"
    results_dir: str = "./data/results"
    temp_dir: str = "./data/tmp"
    max_upload_size_mb: int = 500
    database_url: str = "sqlite:///./data/app.db"
    
    # Phase 7 concurrency and reliability
    inference_workers: int = 1
    max_concurrent_jobs: int = 1
    job_timeout_seconds: int = 600

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_registered(self) -> bool:
        return bool(self.model_path and self.model_path.strip())

    @property
    def is_pytorch(self) -> bool:
        return self.inference_backend.lower() == "pytorch"


@lru_cache
def get_settings() -> Settings:
    return Settings()
