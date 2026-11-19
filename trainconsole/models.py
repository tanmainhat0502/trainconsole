from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Batch(Base):
    __tablename__ = "tc_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    pool_root: Mapped[str] = mapped_column(Text)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    cvat_task_id: Mapped[int] = mapped_column(Integer, index=True)
    budget: Mapped[int] = mapped_column(Integer)
    job_size: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="created")
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UploadedImage(Base):
    __tablename__ = "tc_uploaded_images"
    __table_args__ = (UniqueConstraint("path", name="uq_tc_uploaded_path"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("tc_batches.id"), nullable=True, index=True)
    cvat_task_id: Mapped[int] = mapped_column(Integer, index=True)
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserCredential(Base):
    __tablename__ = "tc_user_credentials"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    password: Mapped[str] = mapped_column(String(200))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ModelVersion(Base):
    __tablename__ = "tc_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    path: Mapped[str] = mapped_column(Text)
    names: Mapped[list[str]] = mapped_column(JSON, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    family: Mapped[str] = mapped_column(String(32), default="yolo26", index=True)
    state: Mapped[str] = mapped_column(String(24), default="production", index=True)
    train_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cvat_project_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainRun(Base):
    __tablename__ = "tc_train_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    task_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    state: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(240), default="queued")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dataset_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    weights_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class InferRun(Base):
    __tablename__ = "tc_infer_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    task_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    job_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    gpu: Mapped[str | None] = mapped_column(String(64), nullable=True)
    batch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("tc_models.id"), nullable=True)
    train_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(240), default="queued")
    counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
