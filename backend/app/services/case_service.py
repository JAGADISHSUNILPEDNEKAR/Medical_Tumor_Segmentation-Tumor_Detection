from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.constants import (
    ALL_MODALITIES,
    ALLOWED_NIFTI_EXTENSIONS,
    REQUIRED_MODALITIES,
    CaseStatus,
    ErrorCode,
)
from app.core.errors import AppError
from app.db.models import CaseFileRecord, CaseRecord
from app.schemas.cases import (
    CaseResponse,
    CaseValidationReport,
    ModalityFileInfo,
    SpatialChecks,
)
from app.storage.filesystem import CaseStorage
from app.validation.nifti import VolumeInfo, assert_spatially_compatible, inspect_nifti

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024
_MUTABLE_STATUSES = {CaseStatus.CREATED, CaseStatus.UPLOADING}


def nifti_extension(filename: str) -> str:
    name = _client_basename(filename).lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise AppError(
        ErrorCode.INVALID_EXTENSION,
        "Only NIfTI files with extension .nii or .nii.gz are accepted.",
        status_code=422,
    )


def _client_basename(filename: str) -> str:
    return filename.replace("\\", "/").split("/")[-1]


class CaseService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.storage = CaseStorage(Path(settings.upload_dir))
        self.max_case_bytes = settings.max_upload_size_mb * 1024 * 1024

    def create_case(self) -> CaseRecord:
        now = datetime.now(timezone.utc)
        case = CaseRecord(
            case_id=str(uuid4()),
            status=CaseStatus.CREATED.value,
            created_at=now,
            updated_at=now,
        )
        self.session.add(case)
        self.session.flush()
        self.storage.prepare_case(case.case_id)
        self._write_metadata(case)
        logger.info(
            "case_created",
            extra={"case_id": case.case_id, "validation_result": CaseStatus.CREATED.value},
        )
        return case

    def get_artifact_file(self, case_id: str, artifact: str) -> Path:
        """Return the stored NIfTI path for an allowlisted viewer artifact."""
        self.get_case(case_id)
        return self.storage.resolve_artifact(case_id, artifact)

    def get_case(self, case_id: str) -> CaseRecord:
        case = self.session.get(CaseRecord, case_id)
        if case is None:
            raise AppError(
                ErrorCode.CASE_NOT_FOUND,
                "No case exists for the given identifier.",
                status_code=404,
                case_id=case_id,
            )
        return case

    async def add_file(self, case_id: str, modality: str, upload: UploadFile) -> CaseRecord:
        modality = modality.lower()
        if modality not in ALL_MODALITIES:
            raise AppError(
                ErrorCode.INVALID_MODALITY,
                f"Unknown modality '{modality}'. Expected one of {', '.join(ALL_MODALITIES)}.",
                status_code=400,
                case_id=case_id,
            )

        case = self.get_case(case_id)
        if CaseStatus(case.status) not in _MUTABLE_STATUSES:
            raise AppError(
                ErrorCode.CASE_NOT_MUTABLE,
                f"Case is {case.status} and cannot accept additional files.",
                status_code=400,
                case_id=case_id,
            )

        if any(existing.modality == modality for existing in case.files):
            raise AppError(
                ErrorCode.DUPLICATE_MODALITY,
                f"Modality '{modality.upper()}' was already uploaded for this case.",
                status_code=400,
                case_id=case_id,
            )

        original = _client_basename(upload.filename or "upload.nii.gz")
        try:
            extension = nifti_extension(original)
        except AppError as exc:
            exc.case_id = case_id
            raise

        remaining = self.max_case_bytes - case.total_bytes
        if remaining <= 0:
            raise AppError(
                ErrorCode.CASE_TOO_LARGE,
                f"Case exceeds the configured limit of {self.settings.max_upload_size_mb} MB.",
                status_code=413,
                case_id=case_id,
            )

        stored_name = f"{modality}{extension}"
        destination = self.storage.destination_for(case_id, stored_name)
        destination.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        size_bytes = await _stream_to_disk(upload, destination, remaining, case_id)
        try:
            info = inspect_nifti(destination, is_segmentation=(modality == "seg"), case_id=case_id)
        except AppError:
            destination.unlink(missing_ok=True)
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "modality_uploaded",
            extra={
                "case_id": case_id,
                "modality": modality,
                "duration_ms": duration_ms,
                "validation_result": "accepted",
            },
        )

        record = CaseFileRecord(
            case_id=case_id,
            modality=modality,
            stored_relpath=self.storage.relative_to_root(destination),
            original_filename=original,
            size_bytes=size_bytes,
            shape_json=json.dumps(list(info.shape)),
            affine_json=json.dumps(info.affine),
            spacing_json=json.dumps(list(info.spacing)),
            warnings_json=json.dumps(info.warnings),
        )
        self.session.add(record)
        case.files.append(record)
        case.status = CaseStatus.UPLOADING.value
        case.total_bytes += size_bytes
        if modality == "seg":
            case.has_ground_truth = True
        self.session.flush()
        self._write_metadata(case)
        return case

    def complete_case(self, case_id: str) -> CaseRecord:
        case = self.get_case(case_id)
        if CaseStatus(case.status) not in {CaseStatus.CREATED, CaseStatus.UPLOADING}:
            if CaseStatus(case.status) == CaseStatus.READY:
                return case
            raise AppError(
                ErrorCode.CASE_NOT_MUTABLE,
                f"Case is {case.status} and cannot be validated.",
                status_code=400,
                case_id=case_id,
            )

        present = {file.modality for file in case.files}
        missing = [name for name in REQUIRED_MODALITIES if name not in present]
        if missing:
            named = ", ".join(item.upper() for item in missing)
            raise AppError(
                ErrorCode.MISSING_MODALITY,
                f"Required modality '{missing[0].upper()}' is missing."
                if len(missing) == 1
                else f"Required modalities are missing: {named}.",
                status_code=400,
                case_id=case_id,
            )

        case.status = CaseStatus.VALIDATING.value
        self.session.flush()
        started = time.perf_counter()

        try:
            volumes: dict[str, VolumeInfo] = {}
            for file in case.files:
                path = Path(self.settings.upload_dir).resolve() / file.stored_relpath
                info = inspect_nifti(path, is_segmentation=(file.modality == "seg"), case_id=case_id)
                volumes[file.modality] = info
            notes = assert_spatially_compatible(volumes, case_id=case_id)
            report = self._build_report(case, volumes, notes, ready=True)
            case.validation_json = report.model_dump_json()
            case.status = CaseStatus.READY.value
            case.error_code = None
            case.error_detail = None
            result = "READY"
        except AppError as exc:
            case.status = CaseStatus.FAILED.value
            case.error_code = exc.error
            case.error_detail = exc.detail
            report = self._build_report(case, {}, [], ready=False)
            case.validation_json = report.model_dump_json()
            self.session.flush()
            self._write_metadata(case)
            logger.info(
                "case_validation_failed",
                extra={
                    "case_id": case_id,
                    "validation_result": exc.error,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise

        self.session.flush()
        self._write_metadata(case)
        logger.info(
            "case_validated",
            extra={
                "case_id": case_id,
                "validation_result": result,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return case

    def to_response(self, case: CaseRecord, *, message: str | None = None) -> CaseResponse:
        files = [
            ModalityFileInfo(
                modality=item.modality,
                original_filename=item.original_filename,
                size_bytes=item.size_bytes,
                shape=json.loads(item.shape_json) if item.shape_json else None,
                spacing=json.loads(item.spacing_json) if item.spacing_json else None,
                warnings=json.loads(item.warnings_json) if item.warnings_json else [],
            )
            for item in sorted(case.files, key=lambda row: row.modality)
        ]
        present = [item.modality for item in files]
        validation = (
            CaseValidationReport.model_validate_json(case.validation_json)
            if case.validation_json
            else self._build_report(case, {}, [], ready=case.status == CaseStatus.READY.value)
        )
        payload = CaseResponse(
            case_id=case.case_id,
            status=case.status,
            created_at=case.created_at or datetime.now(timezone.utc),
            modalities_present=present,
            has_ground_truth=case.has_ground_truth,
            total_bytes=case.total_bytes,
            files=files,
            validation=validation,
            error=case.error_code,
            detail=case.error_detail,
            inference="not_started",
        )
        return payload

    def _build_report(
        self,
        case: CaseRecord,
        volumes: dict[str, VolumeInfo],
        notes: list[str],
        *,
        ready: bool,
    ) -> CaseValidationReport:
        present = {file.modality for file in case.files}
        warnings: list[str] = []
        for info in volumes.values():
            warnings.extend(info.warnings)
        spatial_ok = ready and not case.error_code
        return CaseValidationReport(
            modalities={name: name in present for name in ALL_MODALITIES},
            spatial=SpatialChecks(
                shape_consistent=spatial_ok if volumes else None,
                affine_consistent=spatial_ok if volumes else None,
                spacing_consistent=spatial_ok if volumes else None,
                notes=notes,
            ),
            warnings=warnings,
            ready_for_prediction=ready and all(name in present for name in REQUIRED_MODALITIES),
            ready_for_evaluation=ready and "seg" in present,
        )

    def _write_metadata(self, case: CaseRecord) -> None:
        response = self.to_response(case)
        self.storage.write_metadata(case.case_id, response.model_dump(mode="json"))


async def _stream_to_disk(
    upload: UploadFile, destination: Path, max_bytes: int, case_id: str
) -> int:
    written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise AppError(
                        ErrorCode.CASE_TOO_LARGE,
                        "Upload exceeds the remaining case size limit.",
                        status_code=413,
                        case_id=case_id,
                    )
                handle.write(chunk)
    except AppError:
        destination.unlink(missing_ok=True)
        raise
    except OSError:
        destination.unlink(missing_ok=True)
        raise AppError(
            ErrorCode.STORAGE_ERROR,
            "The file could not be stored.",
            status_code=500,
            case_id=case_id,
        ) from None
    if written == 0:
        destination.unlink(missing_ok=True)
        raise AppError(
            ErrorCode.INVALID_NIFTI,
            "Uploaded file is empty.",
            status_code=422,
            case_id=case_id,
        )
    return written
