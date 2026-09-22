from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


# Explicit allowlist: `extra=` fields are only emitted if named here, so a
# caller can never accidentally log voxel data, file contents, or secrets.
# Server-side paths (model_path) are deliberately included — an operator
# diagnosing a failed checkpoint load needs them, and logs are not client-facing.
_LOGGED_EXTRA_FIELDS: tuple[str, ...] = (
    # Phase 1-5
    "case_id",
    "modality",
    "duration_ms",
    "validation_result",
    "job_id",
    "job_type",
    "inference_source",
    "error",
    "status_code",
    "output_shape",
    # Phase 6 - inference backend and checkpoint diagnostics
    "backend",
    "reason",
    "model_path",
    "model_version",
    "checkpoint",
    "checkpoint_id",
    "checkpoint_fingerprint",
    "service_fingerprint",
    "format",
    "epoch",
    "fingerprint_verified",
    "device",
    "parameters",
    "model_load_seconds",
    "inference_seconds",
    "num_patches",
    "transition",
    "queue_size",
)


class JsonFormatter(logging.Formatter):
    """Structured JSON logs without secrets or image payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key in _LOGGED_EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
