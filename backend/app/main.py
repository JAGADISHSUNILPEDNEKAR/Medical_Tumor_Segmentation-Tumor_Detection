from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.db.session import configure_database
from app.inference.factory import build_inference_service
from app.services.job_queue import JobQueue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: start/stop the FIFO job queue worker."""
    settings = get_settings()
    upload_root = Path(settings.upload_dir).resolve()

    # Built once per process: the checkpoint is loaded at startup and every
    # job reuses the same in-memory model. A real backend that cannot load
    # yields an UnavailableInferenceService rather than aborting startup, so
    # /health can report the problem instead of the API failing to come up.
    inference_service = build_inference_service(settings)
    queue = JobQueue(inference_service, upload_root)

    app.state.job_queue = queue
    app.state.inference_service = inference_service

    await queue.start()
    yield
    await queue.stop()


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
            "Not clinically validated. Inference runs through a configurable "
            "backend: INFERENCE_BACKEND=mock produces a synthetic segmentation, "
            "INFERENCE_BACKEND=pytorch runs the trained 3D U-Net checkpoint."
        ),
        version="0.6.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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
