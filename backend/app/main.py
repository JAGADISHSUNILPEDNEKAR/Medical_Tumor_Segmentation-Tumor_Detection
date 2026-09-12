from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.db.session import configure_database


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
    _ensure_sqlite_parent(settings.database_url)
    configure_database(settings.database_url)

    app = FastAPI(
        title="Medical Tumor Segmentation API",
        description=(
            "Research / decision-support prototype. Not a diagnostic device. "
            "Not clinically validated. Phase 2 validates and stores BraTS NIfTI cases. "
            "Inference is not started."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


def _ensure_sqlite_parent(database_url: str) -> None:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return
    raw = database_url.removeprefix(prefix)
    if raw.startswith("/") and not raw.startswith("///"):
        path = Path(raw)
    else:
        path = Path(raw)
    if path.parent.as_posix() not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)


app = create_app()
