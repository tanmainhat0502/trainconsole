"""Background worker: export CVAT -> train YOLO26x -> infer new CVAT jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from sqlalchemy import select

from .config import settings
from .db import Database, ensure_sqlite_parent
from .models import InferRun, ModelVersion, TrainRun
from .services.inference import inference_job_ids, run_inference
from .services.training import (
    TrainingCancelled,
    build_rfdetr_dataset,
    build_yolo_dataset,
    completed_jobs,
    kill_process_tree,
    latest_model,
    prune_old_weights,
    train_rfdetr,
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


def _model_keep_paths(database: Database) -> set[Path]:
    """Checkpoints that must survive cleanup: production + newest per project."""

    with database.session() as db:
        rows = list(db.scalars(select(ModelVersion).order_by(ModelVersion.id)))
    newest: dict[tuple[int | None, str], ModelVersion] = {}
    for row in rows:
        newest[(row.cvat_project_id, row.family or "yolo26")] = row
    return {
        Path(row.path).resolve()
        for row in rows
        if row.path
        and (
            row.state == "production"
            or newest.get((row.cvat_project_id, row.family or "yolo26")) is row
        )
    }


def _queue_inference(
    database: Database,
    *,
    project_id: int,
    task_ids: list[int],
    model_id: int,
    gpu: str,
    batch: int,
) -> int:
    with database.session() as db:
        run = InferRun(
            name=f"auto-infer-model-{model_id}",
            cvat_project_id=project_id,
            task_ids=task_ids,
            gpu=gpu,
            batch=batch,
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
        family = str(config.get("family") or "yolo26")

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
        rfdetr_dataset = None
        if family == "rfdetr":
            rfdetr_dataset = build_rfdetr_dataset(dataset, dataset_dir)
        label = "RF-DETR-2XL" if family == "rfdetr" else "YOLO26x"
        _set_train(
            database,
            run_id,
            dataset_dir=str(dataset_dir),
            progress=40,
            stage=f"starting {label}",
        )
        with database.session() as db:
            # First run starts from the base checkpoint; later runs tune the
            # newest trained weights of the same family.
            previous = latest_model(db, project_id, family)
            if family == "yolo26":
                base = previous.path if previous else settings.model
                if previous and list(previous.names or []) != list(dataset["names"]):
                    # A model trained before seatbelt was excluded has a
                    # different class order; start from the base checkpoint.
                    base = settings.model
            else:
                base = previous.path if previous else None
                if previous and list(previous.names or []) != list(dataset["names"]):
                    base = None
        if _is_cancelled(database, run_id):
            _set_train(
                database,
                run_id,
                state="cancelled",
                stage="cancelled by admin",
                completed_at=_now(),
            )
            return run_id
        train_gpu = str(config.get("gpu", settings.gpu))
        if family == "yolo26":
            # The batch the user enters is PER GPU: multiply by the GPU count
            # so Ultralytics DDP gives each GPU the requested size.
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
                progress=lambda progress, stage: _set_train(
                    database, run_id, progress=progress, stage=stage
                ),
                should_cancel=lambda: _is_cancelled(database, run_id),
                report_pid=lambda pid: _set_train_pid(database, run_id, pid),
            )
        else:
            assert rfdetr_dataset is not None
            rfdetr_batch: str | int = max(
                1, int(config.get("batch", settings.yolo_batch))
            )
            result = train_rfdetr(
                rfdetr_dataset,
                run_dir,
                epochs=int(config.get("epochs", settings.epochs)),
                batch=rfdetr_batch,
                resolution=int(config.get("imgsz") or settings.rfdetr_resolution),
                gpu=train_gpu,
                base_model=base,
                progress=lambda progress, stage: _set_train(
                    database, run_id, progress=progress, stage=stage
                ),
                should_cancel=lambda: _is_cancelled(database, run_id),
                report_pid=lambda pid: _set_train_pid(database, run_id, pid),
            )
        with database.session() as db:
            old = db.scalars(
                select(ModelVersion).where(
                    ModelVersion.state == "production",
                    ModelVersion.cvat_project_id == project_id,
                    ModelVersion.family == family,
                )
            )
            for item in old:
                item.state = "retired"
            model_name = "rfdetr-2xl" if family == "rfdetr" else "yolo26x"
            model = ModelVersion(
                name=f"{model_name}-v{run_id}",
                path=result["weights"],
                names=dataset["names"],
                metrics={"classes": dataset, **result["metrics"]},
                family=family,
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
        if settings.prune_old_weights:
            try:
                keep_paths = _model_keep_paths(database)
                removed, freed = prune_old_weights(settings.artifact_root, keep_paths)
                if removed:
                    print(
                        f"pruned {removed} old checkpoint(s), "
                        f"freed {freed / 2**20:.0f} MiB",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"checkpoint cleanup failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        per_gpu_batch = max(1, int(config.get("batch", settings.yolo_batch)))
        if family == "rfdetr":
            try:
                training_cfg = json.loads(
                    (run_dir / "train" / "training_config.json").read_text(
                        encoding="utf-8"
                    )
                ).get("train_config", {})
                per_gpu_batch = max(
                    1, int(training_cfg.get("batch_size") or per_gpu_batch)
                )
            except Exception:
                pass
        infer_id = _queue_inference(
            database,
            project_id=project_id,
            task_ids=task_ids,
            model_id=model_id,
            gpu=train_gpu,
            batch=per_gpu_batch,
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
        model = db.get(ModelVersion, run.model_id) if run.model_id else None
        if model is None:
            model = latest_model(db, project_id)
        model_path = model.path if model else None
        model_names = list(model.names or []) if model else []
        model_id = model.id if model else None
        model_family = (
            str(getattr(model, "family", "yolo26") or "yolo26") if model else "yolo26"
        )
        infer_gpu = str(run.gpu or settings.primary_gpu)
        infer_batch = max(1, int(run.batch or 1))

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
        gpus = [g.strip() for g in infer_gpu.split(",") if g.strip()] or [
            settings.primary_gpu
        ]
        if len(gpus) <= 1:
            counts = run_inference(
                model_path=model_path,
                names=model_names,
                project_id=project_id,
                task_ids=task_ids,
                job_ids=job_ids,
                family=model_family,
                batch=infer_batch,
                confidence=settings.confidence,
                imgsz=settings.imgsz,
                gpu=gpus[0],
                progress=lambda progress, stage: _set_infer(
                    database, run_id, progress=progress, stage=stage
                ),
            )
        else:
            job_list = inference_job_ids(project_id, task_ids, job_ids)
            if not job_list:
                counts = {
                    "jobs": 0,
                    "processed": 0,
                    "skipped": 0,
                    "images": 0,
                    "boxes": 0,
                    "model": model_path,
                }
            else:
                _set_infer(
                    database,
                    run_id,
                    progress=10,
                    stage=f"inference on {len(gpus)} GPUs",
                )
                shard_dir = settings.artifact_root / "infer_runs" / str(run_id)
                shard_dir.mkdir(parents=True, exist_ok=True)
                script = Path(__file__).resolve().parent / "infer_cli.py"
                chunks = [job_list[index:: len(gpus)] for index in range(len(gpus))]
                procs: list[tuple[subprocess.Popen, Path, str, Path]] = []
                for gpu, chunk in zip(gpus, chunks):
                    if not chunk:
                        continue
                    out_path = shard_dir / f"gpu_{gpu}.json"
                    log_path = shard_dir / f"gpu_{gpu}.log"
                    command = [
                        sys.executable,
                        str(script),
                        "--model", model_path,
                        "--family", model_family,
                        "--project-id", str(project_id),
                        "--job-ids", ",".join(str(value) for value in chunk),
                        "--gpu", gpu,
                        "--batch", str(infer_batch),
                        "--out", str(out_path),
                    ]
                    log = open(log_path, "w", encoding="utf-8")
                    proc = subprocess.Popen(
                        command,
                        cwd=str(Path.cwd()),
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    procs.append((proc, out_path, gpu, log_path))
                failures = []
                for proc, _out_path, gpu, log_path in procs:
                    code = proc.wait()
                    if code != 0:
                        tail = log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-20:]
                        failures.append(f"GPU {gpu} exited {code}: " + "\n".join(tail))
                if failures:
                    raise RuntimeError("\n".join(failures))
                counts = {
                    "jobs": len(job_list),
                    "processed": 0,
                    "skipped": 0,
                    "images": 0,
                    "boxes": 0,
                    "model": model_path,
                }
                for _proc, out_path, _gpu, _log in procs:
                    payload = json.loads(out_path.read_text(encoding="utf-8"))
                    for key in ("processed", "skipped", "images", "boxes"):
                        counts[key] += int(payload.get(key) or 0)
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
