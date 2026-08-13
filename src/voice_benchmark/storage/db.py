"""Database engine and session management.

Uses whatever DATABASE_URL is set in settings -- SQLite by default (zero
setup), swap to Postgres for anything beyond local dev (see .env.example).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from voice_benchmark.config.settings import get_settings
from voice_benchmark.storage.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # SQLite needs this to be usable across the threads uvicorn/streamlit
        # may use later; harmless for Postgres, which ignores it.
        connect_args = (
            {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        )
        _engine = create_engine(settings.database_url, connect_args=connect_args)
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call repeatedly --
    idempotent, does not touch existing tables/data."""
    Base.metadata.create_all(bind=get_engine())


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on any error,
    always closes. Use as: `with session_scope() as session: ...`"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
