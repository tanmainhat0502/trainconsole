"""Background worker: export CVAT -> train YOLO26x -> infer new CVAT jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import traceback

from sqlalchemy import select

from .config import settings
from .db import Database, ensure_sqlite_parent
from .models import InferRun, ModelVersion, TrainRun
from .services.inference import run_inference
from .services.training import (
    TrainingCancelled,
    build_yolo_dataset,
    completed_jobs,
    kill_process_tree,
    latest_model,
    train_yolo,
)


def _now():
    return datetime.now(timezone.utc)


def _set_train(database: Database, run_id: int, **values) -> None:
    with database.session() as db:
        run = db.get(TrainRun, run_id)
        if run is not None:
            for key, value in values.items():
                setattr(run, key, value)
            db.flush()


def _set_infer(database: Database, run_id: int, **values) -> None:
    with database.session() as db:
        run = db.get(InferRun, run_id)
        if run is not None:
            for key, value in values.items():
                setattr(run, key, value)
            db.flush()


def _is_cancelled(database: Database, run_id: int) -> bool:
    with database.session() as db:
        run = db.get(TrainRun, run_id)
        return run is not None and run.state in {"cancelling", "cancelled"}


def _set_train_pid(database: Database, run_id: int, pid: int) -> None:
    with database.session() as db:
        run = db.get(TrainRun, run_id)
        if run is not None:
            config = dict(run.config or {})
            config["pid"] = pid
            run.config = config
            db.flush()


def _queue_inference(database: Database, *, project_id: int, task_ids: list[int], model_id: int) -> int:
    with database.session() as db:
        run = InferRun(
            name=f"auto-infer-model-{model_id}",
            cvat_project_id=project_id,
            task_ids=task_ids,
            model_id=model_id,
            state="queued",
            stage="waiting for new jobs",
        )
        db.add(run)
        db.flush()
        return run.id


def _reclaim_stale_runs(database: Database) -> None:
    """Requeue runs left ``running`` by a worker that died mid-job.

    A restarted worker cannot adopt the old training launcher, so any process
    tree saved in a ``running`` run is killed first. Exports already downloaded
    are kept, which makes the restart cheap.
    """

    with database.session() as db:
        for run in db.scalars(select(TrainRun).where(TrainRun.state == "running")).all():
            pid = (run.config or {}).get("pid")
            # A restarted worker cannot wait() on the old launcher. Kill the
            # whole tree unconditionally: the launcher may already be dead
            # while DDP ranks were reparented to init and are still using GPUs.
            kill_process_tree(
                int(pid) if pid else None,
                run_marker=f"train_runs/{run.id}/",
            )
            run.state = "queued"
            run.stage = "requeued after worker restart"
            run.progress = 0
        for run in db.scalars(select(InferRun).where(InferRun.state == "running")).all():
            run.state = "queued"
            run.stage = "requeued after worker restart"
            run.progress = 0
        db.flush()


def process_train(database: Database) -> int | None:
    with database.session() as db:
        run = db.scalar(
            select(TrainRun).where(TrainRun.state == "queued").order_by(TrainRun.id).limit(1)
        )
        if run is None:
            return None
        run.state = "running"
        run.started_at = _now()
        run.stage = "collecting completed CVAT jobs"
        run.progress = 2
        run_id = run.id
        project_id = run.cvat_project_id
        task_ids = list(run.task_ids or [])
        config = dict(run.config or {})

    try:
        jobs = completed_jobs(project_id, task_ids)
        if not jobs:
            raise ValueError("no completed CVAT jobs found")
        _set_train(database, run_id, progress=5, stage=f"exporting {len(jobs)} completed jobs")
        run_dir = settings.artifact_root / "train_runs" / str(run_id)
        dataset_dir = run_dir / "dataset"

        def dataset_progress(progress: int, stage: str) -> None:
            _set_train(database, run_id, progress=progress, stage=stage)

        dataset = build_yolo_dataset(
            jobs,
            dataset_dir,
            val_fraction=settings.val_fraction,
            exclude_classes=settings.exclude_from_detection,
            progress=dataset_progress,
            should_cancel=lambda: _is_cancelled(database, run_id),
        )
        _set_train(database, run_id, dataset_dir=str(dataset_dir), progress=40, stage="starting YOLO26x")
        with database.session() as db:
            # First run starts from the base checkpoint; later runs tune the
            # newest trained weights so each round builds on the previous one.
            previous = latest_model(db, project_id)
            base = previous.path if previous else settings.model
            if previous and list(previous.names or []) != list(dataset["names"]):
                # A model trained before seatbelt was excluded has a different
                # class order. Start from the base YOLO checkpoint instead of
                # silently remapping heads.
                base = settings.model
        if _is_cancelled(database, run_id):
            _set_train(
                database,
                run_id,
                state="cancelled",
                stage="cancelled by admin",
                completed_at=_now(),
            )
            return run_id
        # The batch the user enters is PER GPU: multiply by the GPU count so
        # Ultralytics DDP (which splits the total batch) gives each GPU the
        # requested size.
        train_gpu = str(config.get("gpu", settings.gpu))
        world = len([g for g in train_gpu.split(",") if g.strip()]) or 1
        total_batch = max(1, int(config.get("batch", settings.yolo_batch))) * world
        result = train_yolo(
            dataset_dir,
            run_dir,
            epochs=int(config.get("epochs", settings.epochs)),
            batch=total_batch,
            imgsz=int(config.get("imgsz", settings.imgsz)),
            gpu=train_gpu,
            workers=int(config.get("workers", settings.workers)),
            base_model=base,
            progress=lambda progress, stage: _set_train(database, run_id, progress=progress, stage=stage),
            should_cancel=lambda: _is_cancelled(database, run_id),
            report_pid=lambda pid: _set_train_pid(database, run_id, pid),
        )
        with database.session() as db:
            old = db.scalars(
                select(ModelVersion).where(ModelVersion.state == "production")
            )
            for item in old:
                item.state = "retired"
            model = ModelVersion(
                name=f"yolo26x-v{run_id}",
                path=result["weights"],
                names=dataset["names"],
                metrics={"classes": dataset, **result["metrics"]},
                state="production",
                train_run_id=run_id,
                cvat_project_id=project_id,
            )
            db.add(model)
            run = db.get(TrainRun, run_id)
            assert run is not None
            run.state = "completed"
            run.progress = 100
            run.stage = "completed"
            run.dataset_dir = str(dataset_dir)
            run.weights_path = result["weights"]
            run.metrics = {"classes": dataset, **result["metrics"]}
            run.completed_at = _now()
            db.flush()
            model_id = model.id
        infer_id = _queue_inference(
            database, project_id=project_id, task_ids=task_ids, model_id=model_id
        )
        _set_train(database, run_id, stage=f"completed; inference #{infer_id} queued")
        return run_id
    except TrainingCancelled:
        _set_train(
            database,
            run_id,
            state="cancelled",
            stage="cancelled by admin",
            completed_at=_now(),
        )
        return run_id
    except Exception as exc:
        with database.session() as db:
            run = db.get(TrainRun, run_id)
            if run is not None and run.state == "cancelled":
                # cancelled from the API: the process was killed directly
                return run_id
        _set_train(
            database,
            run_id,
            state="failed",
            progress=100,
            stage=f"failed: {type(exc).__name__}",
            error="".join(traceback.format_exception_only(type(exc), exc))[-4000:],
            completed_at=_now(),
        )
        return run_id


def process_infer(database: Database) -> int | None:
    with database.session() as db:
        run = db.scalar(
            select(InferRun).where(InferRun.state == "queued").order_by(InferRun.id).limit(1)
        )
        if run is None:
            return None
        run.state = "running"
        run.started_at = _now()
        run.stage = "looking for new CVAT jobs"
        run.progress = 2
        run_id = run.id
        project_id = run.cvat_project_id
        task_ids = list(run.task_ids or [])
        job_ids = list(run.job_ids or [])
        model = latest_model(db, project_id)
        if model is None and run.model_id:
            model = db.get(ModelVersion, run.model_id)
        model_path = model.path if model else None
        model_names = list(model.names or []) if model else []
        model_id = model.id if model else None

    if not model_path or not Path(model_path).exists():
        _set_infer(
            database,
            run_id,
            state="failed",
            progress=100,
            stage="no production model",
            error="Train a model first or promote a model, then run inference.",
            completed_at=_now(),
        )
        return run_id

    try:
        counts = run_inference(
            model_path=model_path,
            names=model_names,
            project_id=project_id,
            task_ids=task_ids,
            job_ids=job_ids,
            confidence=settings.confidence,
            imgsz=settings.imgsz,
            gpu=settings.primary_gpu,
            progress=lambda progress, stage: _set_infer(
                database, run_id, progress=progress, stage=stage
            ),
        )
        _set_infer(
            database,
            run_id,
            state="completed",
            progress=100,
            stage="completed",
            counts={**counts, "model_id": model_id},
            completed_at=_now(),
        )
        return run_id
    except Exception as exc:
        _set_infer(
            database,
            run_id,
            state="failed",
            progress=100,
            stage=f"failed: {type(exc).__name__}",
            error="".join(traceback.format_exception_only(type(exc), exc))[-4000:],
            completed_at=_now(),
        )
        return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainconsole-worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--queue", choices=("all", "train", "infer"), default="all")
    parser.add_argument("--poll", type=float, default=3.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.ensure_dirs()
    ensure_sqlite_parent(settings.database_url)
    database = Database(settings.database_url)
    database.create_all()
    _reclaim_stale_runs(database)
    print("worker ready", flush=True)
    while True:
        worked = False
        if args.queue in {"all", "train"}:
            run_id = process_train(database)
            if run_id is not None:
                print(f"train run {run_id} processed")
                worked = True
        if args.queue in {"all", "infer"}:
            run_id = process_infer(database)
            if run_id is not None:
                print(f"infer run {run_id} processed")
                worked = True
        if args.once:
            return 0
        if not worked:
            time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
