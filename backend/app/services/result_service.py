from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.db.models import ResultRecord
from app.inference.base import InferenceResult
from app.schemas.results import (
    EvaluationInfo,
    MeasurementsInfo,
    ResultResponse,
    SegmentationInfo,
)

logger = logging.getLogger(__name__)


class ResultService:
    """Result persistence and retrieval."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_result(
        self,
        job_id: str,
        case_id: str,
        inference_result: InferenceResult,
    ) -> ResultRecord:
        """Persist a result from a completed inference job."""
        now = datetime.now(timezone.utc)
        result = ResultRecord(
            result_id=str(uuid4()),
            job_id=job_id,
            case_id=case_id,
            segmentation_path=inference_result.segmentation_path,
            measurements_json=json.dumps(inference_result.measurements, default=str),
            metadata_json=json.dumps(inference_result.metadata, default=str),
            created_at=now,
        )
        self.session.add(result)
        self.session.flush()
        logger.info(
            "result_created",
            extra={
                "job_id": job_id,
                "case_id": case_id,
                "result_id": result.result_id,
            },
        )
        return result

    def get_result(self, result_id: str) -> ResultRecord:
        """Look up a result by UUID. Raises RESULT_NOT_FOUND on miss."""
        result = self.session.get(ResultRecord, result_id)
        if result is None:
            raise AppError(
                ErrorCode.RESULT_NOT_FOUND,
                "No result exists for the given identifier.",
                status_code=404,
            )
        return result

    def to_response(self, result: ResultRecord) -> ResultResponse:
        """Convert a ResultRecord to an API response.

        Does not expose raw filesystem paths.
        """
        measurements_data = (
            json.loads(result.measurements_json)
            if result.measurements_json
            else {}
        )
        metadata_data = (
            json.loads(result.metadata_json)
            if result.metadata_json
            else {}
        )

        has_segmentation = bool(result.segmentation_path)
        is_synthetic = measurements_data.get("synthetic", True)
        has_ground_truth = metadata_data.get("has_ground_truth", False)

        return ResultResponse(
            result_id=result.result_id,
            job_id=result.job_id,
            case_id=result.case_id,
            status="COMPLETED",
            inference_source=metadata_data.get("inference_source", "mock"),
            model_version=metadata_data.get("model_version"),
            segmentation=SegmentationInfo(available=has_segmentation),
            measurements=MeasurementsInfo(
                synthetic=is_synthetic,
                description=measurements_data.get("description"),
                foreground_voxels=measurements_data.get("foreground_voxels", 0),
                foreground_volume_mm3=measurements_data.get("foreground_volume_mm3", 0.0),
                foreground_volume_cm3=measurements_data.get("foreground_volume_cm3", 0.0),
                regions=measurements_data.get("regions"),
            ),
            evaluation=EvaluationInfo(
                available=False,
                dice=None,
                hd95_mm=None,
            ),
            created_at=result.created_at,
        )
