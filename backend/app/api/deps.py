from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.services.case_service import CaseService


def get_case_service(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Generator[CaseService, None, None]:
    yield CaseService(session, settings)
