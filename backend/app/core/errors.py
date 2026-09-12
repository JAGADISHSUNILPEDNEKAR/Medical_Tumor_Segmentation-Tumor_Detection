from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_ctx
from app.schemas.errors import APIError

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class AppError(Exception):
    """Safe, client-facing application error."""

    def __init__(
        self,
        error: str,
        detail: str,
        status_code: int = 400,
        case_id: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.status_code = status_code
        self.case_id = case_id


def _request_id(request: Request) -> str:
    existing = request.headers.get(REQUEST_ID_HEADER)
    return existing.strip() if existing and existing.strip() else str(uuid.uuid4())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or _request_id(request)
        logger.warning(
            "application_error",
    extra={"error": exc.error, "status_code": exc.status_code},
        )
        body = APIError(
            error=exc.error,
            detail=exc.detail,
            request_id=request_id,
            case_id=exc.case_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(exclude_none=True),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or _request_id(request)
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        body = APIError(error="http_error", detail=detail, request_id=request_id)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or _request_id(request)
        logger.info("validation_error")
        body = APIError(
            error="validation_error",
            detail="Request validation failed.",
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or _request_id(request)
        logger.exception("unhandled_error")
        body = APIError(
            error="internal_error",
            detail="An unexpected error occurred. No clinical result was produced.",
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=body.model_dump())
