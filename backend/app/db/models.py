from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import CaseStatus, JobStatus, JobType
from app.db.base import Base


class CaseRecord(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=CaseStatus.CREATED.value)
    has_ground_truth: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    files: Mapped[list[CaseFileRecord]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[JobRecord]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )


class CaseFileRecord(Base):
    __tablename__ = "case_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.case_id"), nullable=False, index=True)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    stored_relpath: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    shape_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    affine_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    spacing_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    case: Mapped[CaseRecord] = relationship(back_populates="files")


class JobRecord(Base):
    """Persistent job model with explicit state machine.

    States: QUEUED → RUNNING → COMPLETED | FAILED
    A queued job must never jump directly to COMPLETED without execution.
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.case_id"), nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=JobStatus.QUEUED.value, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inference_source: Mapped[str] = mapped_column(String(20), nullable=False, default="mock")
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    case: Mapped[CaseRecord] = relationship(back_populates="jobs")
    result: Mapped[ResultRecord | None] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class ResultRecord(Base):
    """Persistent result model. 1:1 with JobRecord.

    Stores artifact paths (relative, never exposed to API clients),
    measurements JSON, and metadata JSON.
    """

    __tablename__ = "results"

    result_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.job_id"), nullable=False, unique=True
    )
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.case_id"), nullable=False, index=True)
    segmentation_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    measurements_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[JobRecord] = relationship(back_populates="result")
