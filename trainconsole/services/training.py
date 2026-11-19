from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import settings
from ..cvat import export_job_yolo, list_jobs

Progress = Callable[[int, str], None]
logger = logging.getLogger(__name__)


class TrainingCancelled(RuntimeError):
    """Raised when an admin cancels a running training job."""


def _collect_tree(root: int) -> set[int]:
    """Return ``root`` plus every live descendant read from /proc."""

    to_visit = [root]
    seen: set[int] = set()
    while to_visit:
        current = to_visit.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for task_dir in Path(f"/proc/{current}/task").iterdir():
                children_file = task_dir / "children"
                if children_file.exists():
                    for child in children_file.read_text().split():
                        if child:
                            to_visit.append(int(child))
        except (OSError, ValueError):
            pass
    return seen


def _scan_run_processes(marker: str) -> set[int]:
    """Find our YOLO/DDP processes by the run marker in their cmdline/script.

    DDP ranks run ``.../Ultralytics/DDP/_temp_*.py``; the run directory only
    appears inside that script. When the launcher dies first, the ranks are
    reparented to init, so walking children from the old pid finds nothing —
    this scan is the fallback that catches them.
    """

    found: set[int] = set()
    if not marker:
        return found
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        current = int(entry.name)
        if current == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if "trainconsole/yolo_cli.py" not in cmdline and "Ultralytics/DDP/_temp_" not in cmdline:
            continue
        if marker in cmdline:
            found.add(current)
            continue
        if "Ultralytics/DDP/_temp_" in cmdline:
            script = cmdline.split()[-1]
            try:
                text = Path(script).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker in text:
                found.add(current)
    return found


def kill_process_tree(pid: int | None, *, run_marker: str | None = None) -> None:
    """SIGKILL a training process and every related process.

    Ultralytics DDP runs each rank in its own process group (``start_new_session``
    inside torchrun), so killing the launcher's group is not enough. We walk
    /proc for descendants and, as a fallback, scan for processes belonging to
    ``run_marker`` (e.g. ``train_runs/23/``) in case the launcher already died.
    """

    targets: set[int] = set()
    try:
        root = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        root = None
    if root is not None and Path(f"/proc/{root}").exists():
        targets |= _collect_tree(root)
    if run_marker:
        targets |= _scan_run_processes(str(run_marker))
    for current in targets:
        try:
            os.killpg(os.getpgid(current), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(current, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def completed_jobs(project_id: int, task_ids: list[int] | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(project_id=project_id, task_ids=task_ids)
        if job["state"] == "completed"
    ]


def _extract_job(job: dict[str, Any], archive: Path, dataset: Path, split: str) -> str | None:
    names_seen: str | None = None
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.endswith("obj.names")]
        if names:
            names_seen = zf.read(names[0]).decode("utf-8", errors="replace").strip()
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            suffix = Path(member).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                target_dir = dataset / "images" / split
            elif suffix == ".txt" and "obj_train_data" in member:
                target_dir = dataset / "labels" / split
            else:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"job{job['id']}_{Path(member).name}"
            target.write_bytes(zf.read(member))
    return names_seen


def _job_signature(job: dict[str, Any]) -> str | None:
    value = job.get("updated_date")
    return str(value) if value else None


def _job_cache_valid(job: dict[str, Any], cache_dir: Path) -> bool:
    if not settings.export_cache:
        return False
    signature = _job_signature(job)
    if not signature:
        return False
    job_id = int(job["id"])
    cache_path = cache_dir / f"job_{job_id}.zip"
    meta_path = cache_dir / f"job_{job_id}.json"
    if not cache_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("updated_date") == signature
        and meta.get("state") == "completed"
        and meta.get("format") == "YOLO 1.1"
    )


def cached_job_export(job: dict[str, Any], cache_dir: Path) -> tuple[Path, bool]:
    """Export one job to the persistent cache, reusing it when unchanged.

    CVAT updates ``job.updated_date`` whenever annotations are changed, so a
    matching timestamp means the cached zip is still valid. Returns
    ``(archive_path, was_cached)``.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    job_id = int(job["id"])
    cache_path = cache_dir / f"job_{job_id}.zip"
    meta_path = cache_dir / f"job_{job_id}.json"
    if _job_cache_valid(job, cache_dir):
        logger.info("reuse cached export for job %s", job_id)
        return cache_path, True

    temp_path = cache_dir / f".job_{job_id}.{os.getpid()}.tmp"
    try:
        export_job_yolo(job_id, temp_path)
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)

    meta = {
        "job_id": job_id,
        "task_id": job.get("task_id"),
        "project_id": job.get("project_id"),
        "state": "completed",
        "updated_date": _job_signature(job),
        "format": "YOLO 1.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_tmp = cache_dir / f".job_{job_id}.meta.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    logger.info("cached export for job %s (%s bytes)", job_id, cache_path.stat().st_size)
    return cache_path, False


def build_yolo_dataset(
    jobs: list[dict[str, Any]],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    exclude_classes: tuple[str, ...] = (),
    progress: Progress | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if not jobs:
        raise ValueError("no completed CVAT jobs selected for training")
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = out_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    cache_dir = settings.artifact_root / "job_cache"
    train_jobs = list(jobs)
    names: list[str] | None = None

    def extract(job_list: list[dict[str, Any]], split: str) -> None:
        nonlocal names
        for index, job in enumerate(job_list):
            cached_archive, was_cached = cached_job_export(job, cache_dir)
            archive = exports / f"job_{job['id']}.zip"
            if archive.exists():
                archive.unlink()
            try:
                os.link(cached_archive, archive)
            except OSError:
                shutil.copy2(cached_archive, archive)
            job_names = _extract_job(job, archive, out_dir, split)
            if job_names:
                parsed = [line.strip() for line in job_names.splitlines() if line.strip()]
                if names is None:
                    names = parsed
                elif names != parsed:
                    raise ValueError(
                        f"job {job['id']} has a different class order: {parsed} != {names}"
                    )
            if progress:
                action = "cached" if was_cached else "export"
                progress(
                    10 + int(30 * (index + 1) / max(1, len(job_list))),
                    f"{action} train: job {job['id']}",
                )
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled("cancelled by admin")

    extract(train_jobs, "train")
    train_images = sorted((out_dir / "images" / "train").glob("*"))
    if not train_images:
        raise ValueError("YOLO export produced no training images")
    val_count = min(
        len(train_images),
        max(1, round(len(train_images) * float(val_fraction))),
    )
    # Seen validation: sample from the training set, but keep those images in
    # train too. This measures memorization/overfit, not generalization.
    val_images = sorted(
        train_images,
        key=lambda path: hashlib.sha256(f"seen-val:{path.name}".encode()).hexdigest(),
    )[:val_count]
    for source in val_images:
        target = out_dir / "images" / "val" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        label_source = out_dir / "labels" / "train" / f"{source.stem}.txt"
        label_target = out_dir / "labels" / "val" / f"{source.stem}.txt"
        label_target.parent.mkdir(parents=True, exist_ok=True)
        if label_source.exists():
            shutil.copy2(label_source, label_target)
        else:
            label_target.write_text("", encoding="utf-8")
    (out_dir / "eval_note.txt").write_text(
        "train = all completed images; val = 20% of the same train images (SEEN). "
        "mAP measures memorization, not generalization.",
        encoding="utf-8",
    )

    if not names:
        raise ValueError("YOLO export did not contain obj.names")
    removed = {name.lower() for name in exclude_classes}
    kept = [(index, name) for index, name in enumerate(names) if name.lower() not in removed]
    if not kept:
        raise ValueError("all classes were excluded from detection training")
    remap = {old: new for new, (old, _) in enumerate(kept)}
    filtered_names = [name for _, name in kept]
    for split in ("train", "val"):
        label_dir = out_dir / "labels" / split
        for path in label_dir.glob("*.txt"):
            lines: list[str] = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    old_class = int(float(parts[0]))
                except ValueError:
                    continue
                new_class = remap.get(old_class)
                if new_class is None:
                    continue
                lines.append(" ".join([str(new_class), *parts[1:]]))
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(filtered_names)},
    }
    data_file = out_dir / "data.yaml"
    data_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    counts = {
        "train_jobs": [job["id"] for job in train_jobs],
        "val_jobs": [job["id"] for job in train_jobs],
        "train_images": len(list((out_dir / "images" / "train").glob("*"))),
        "val_images": len(list((out_dir / "images" / "val").glob("*"))),
        "names": filtered_names,
        "excluded_from_detection": list(exclude_classes),
        "original_names": names,
        "eval_mode": "train20-seen",
        "val_is_seen": True,
        "val_fraction": val_fraction,
    }
    (out_dir / "dataset.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return {"data_file": str(data_file), **counts}


def latest_model_path(session, project_id: int | None = None) -> str | None:
    """Return the newest trained checkpoint for the project (or globally)."""

    from sqlalchemy import select

    from ..models import ModelVersion

    query = select(ModelVersion)
    if project_id is not None:
        query = query.where(ModelVersion.cvat_project_id == project_id)
    model = session.scalar(query.order_by(ModelVersion.id.desc()).limit(1))
    return model.path if model else None


def latest_model(session, project_id: int | None = None):
    """Return the newest ModelVersion row for the project (or globally)."""

    from sqlalchemy import select

    from ..models import ModelVersion

    query = select(ModelVersion)
    if project_id is not None:
        query = query.where(ModelVersion.cvat_project_id == project_id)
    return session.scalar(query.order_by(ModelVersion.id.desc()).limit(1))


def _epoch_from_log(
    path: Path, default_total: int, start_offset: int = 0
) -> tuple[int, int] | None:
    """Read the newest ``epoch/total`` line written in the current run.

    Needed for DDP: Ultralytics runs the actual training in fresh subprocesses,
    so callbacks registered by ``yolo_cli.py`` never fire there and
    ``progress.json`` stays at epoch 0. ``start_offset`` skips lines left in the
    append-only log by earlier attempts of the same run.
    """

    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            begin = max(start_offset, size - 65536)
            fh.seek(begin)
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    matches = re.findall(r"(\d+)/(\d+)\s+[\d.]+G", data)
    if not matches:
        return None
    epoch, total = matches[-1]
    return int(epoch), int(total or default_total)


def train_yolo(
    dataset_dir: Path,
    run_dir: Path,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    gpu: str,
    workers: int,
    base_model: str | None,
    progress: Progress | None = None,
    should_cancel: Callable[[], bool] | None = None,
    report_pid: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_marker = f"train_runs/{run_dir.name}/"
    result_file = run_dir / "result.json"
    status_file = run_dir / "progress.json"
    log_path = run_dir / "train.log"
    status_file.unlink(missing_ok=True)
    script = Path(__file__).resolve().parents[1] / "yolo_cli.py"
    command = [
        settings.train_python,
        str(script),
        "--data", str(dataset_dir / "data.yaml"),
        "--run-dir", str(run_dir),
        "--epochs", str(max(1, int(epochs))),
        "--batch", str(max(1, int(batch))),
        "--imgsz", str(max(64, int(imgsz))),
        "--gpu", str(gpu or settings.gpu),
        "--workers", str(max(0, int(workers))),
        "--base", str(base_model or settings.model),
        "--result", str(result_file),
        "--status", str(status_file),
    ]
    if progress:
        progress(45, f"YOLO26x via {settings.train_env} · GPU {gpu or settings.gpu}")
    last_status: tuple[int, int] | None = None
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        run_log_offset = log.tell()
        # new session so cancelling can kill the whole DDP process group
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if report_pid is not None:
            report_pid(process.pid)
        while process.poll() is None:
            if should_cancel is not None and should_cancel():
                kill_process_tree(process.pid, run_marker=run_marker)
                process.wait()
                raise TrainingCancelled("cancelled by admin")
            if status_file.exists():
                try:
                    payload = json.loads(status_file.read_text(encoding="utf-8"))
                    epoch = int(payload.get("epoch") or 0)
                    total = int(payload.get("epochs") or epochs)
                except Exception:
                    payload = None
                else:
                    if total > 0 and (epoch, total) != last_status:
                        last_status = (epoch, total)
                        if progress:
                            progress(
                                45 + int(50 * epoch / total),
                                f"epoch {epoch}/{total}",
                            )
            parsed = _epoch_from_log(log_path, epochs, run_log_offset)
            if parsed is not None:
                epoch, total = parsed
                if total > 0 and (epoch, total) != last_status:
                    last_status = (epoch, total)
                    if progress:
                        progress(
                            45 + int(50 * epoch / total),
                            f"epoch {epoch}/{total}",
                        )
            time.sleep(1)
    returncode = process.returncode
    # The API can SIGKILL the tree directly; in that case poll() returns before
    # the cancellation check inside the loop runs, so re-check here to avoid
    # overwriting a user-visible "cancelled" state with "failed". The extra
    # kill catches DDP ranks that survived when only the launcher was killed.
    if should_cancel is not None and should_cancel():
        kill_process_tree(process.pid, run_marker=run_marker)
        raise TrainingCancelled("cancelled by admin")
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"YOLO runner exited with {returncode}:\n{tail}")
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    if progress:
        progress(95, "YOLO26 checkpoint ready")
    return {
        "weights": payload["weights"],
        "metrics": payload.get("metrics") or {},
        "base_model": payload.get("base_model") or settings.model,
        "gpu": payload.get("gpu") or settings.gpu,
        "env": settings.train_env,
    }
