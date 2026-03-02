from __future__ import annotations

import datetime as dt
from typing import Optional, List

from sqlalchemy import String, Text, DateTime, Boolean, Integer, ForeignKey, JSON, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .settings import settings

class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    vertical: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    queue: Mapped[str] = mapped_column(String(16), default="cpu")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    cache_key: Mapped[Optional[str]] = mapped_column(String(96), index=True)
    use_cache: Mapped[bool] = mapped_column(Boolean, default=True)

    config: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    events: Mapped[List["JobEvent"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    artifacts: Mapped[List["JobArtifact"]] = relationship(back_populates="job", cascade="all, delete-orphan")

class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow(), index=True)
    level: Mapped[str] = mapped_column(String(12), default="INFO")
    message: Mapped[str] = mapped_column(Text)
    job: Mapped["Job"] = relationship(back_populates="events")

class JobArtifact(Base):
    __tablename__ = "job_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    s3_key: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    job: Mapped["Job"] = relationship(back_populates="artifacts")

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db() -> None:
    Base.metadata.create_all(engine)
