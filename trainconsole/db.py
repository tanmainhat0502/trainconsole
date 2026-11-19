from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str):
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _on_connect(dbapi_connection, _record):  # type: ignore[no-untyped-def]
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False, class_=Session
        )

    def create_all(self) -> None:
        from . import models  # noqa: F401

        _ = models
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that predate a model change (SQLite has no migrations)."""

        from sqlalchemy import inspect, text

        inspector = inspect(self.engine)
        if "tc_infer_runs" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("tc_infer_runs")}
        if "job_ids" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE tc_infer_runs ADD COLUMN job_ids JSON")
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def ensure_sqlite_parent(url: str) -> None:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
