from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import AppError
from app.db.base import Base
from app.db import models as _models  # noqa: F401 — register metadata

_engine: Engine | None = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, class_=Session)


def configure_database(database_url: str) -> Engine:
    global _engine
    if _engine is not None:
        _engine.dispose()

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    _engine = create_engine(database_url, connect_args=connect_args, future=True)
    SessionLocal.configure(bind=_engine)
    Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except AppError:
        session.commit()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
