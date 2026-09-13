from collections.abc import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.services.case_service import CaseService
from app.services.job_queue import JobQueue
from app.services.job_service import JobService
from app.services.result_service import ResultService


def get_case_service(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Generator[CaseService, None, None]:
    yield CaseService(session, settings)


def get_job_service(
    session: Session = Depends(get_session),
) -> Generator[JobService, None, None]:
    yield JobService(session)


def get_result_service(
    session: Session = Depends(get_session),
) -> Generator[ResultService, None, None]:
    yield ResultService(session)


def get_job_queue(request: Request) -> JobQueue:
    """Retrieve the process-local job queue from app state."""
    return request.app.state.job_queue
