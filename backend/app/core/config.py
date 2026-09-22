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
    max_upload_size_mb: int = 500
    database_url: str = "sqlite:///./data/app.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_registered(self) -> bool:
        return bool(self.model_path and self.model_path.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
