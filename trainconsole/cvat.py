"""Thin CVAT integration layer.

CVAT is the source of truth for labeling. The console only creates tasks/jobs,
reads progress, exports completed jobs, and uploads pre-annotations.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Iterator

import requests

from .config import settings

logger = logging.getLogger(__name__)


class CvatError(RuntimeError):
    pass


@contextmanager
def client() -> Iterator[Any]:
    if not settings.cvat_username or not settings.cvat_password:
        raise CvatError(
            "CVAT credentials are missing. Set TC_CVAT_USERNAME / TC_CVAT_PASSWORD "
            "or point TC_CVAT_ENV_FILE to the CVAT .env file."
        )
    from cvat_sdk import make_client

    with make_client(
        settings.cvat_url,
        credentials=(settings.cvat_username, settings.cvat_password),
    ) as c:
        yield c


def state(job: Any) -> str:
    value = getattr(job, "state", "new")
    value = getattr(value, "value", value)
    return str(value).lower().replace("_", " ")


def assignee(job: Any) -> str | None:
    user = getattr(job, "assignee", None)
    return getattr(user, "username", None) if user else None


def job_row(job: Any, task: Any | None = None, project_id: int | None = None) -> dict[str, Any]:
    start = int(getattr(job, "start_frame", 0))
    stop = int(getattr(job, "stop_frame", start))
    updated = getattr(job, "updated_date", None)
    return {
        "id": int(job.id),
        "task_id": int(job.task_id),
        "task_name": getattr(task, "name", "") if task is not None else "",
        "project_id": project_id,
        "state": state(job),
        "assignee": assignee(job),
        "assignee_id": getattr(getattr(job, "assignee", None), "id", None),
        "start_frame": start,
        "stop_frame": stop,
        "frames": max(0, stop - start + 1),
        "stage": str(getattr(getattr(job, "stage", ""), "value", getattr(job, "stage", ""))),
        "updated_date": (
            updated.isoformat()
            if hasattr(updated, "isoformat")
            else (str(updated) if updated else None)
        ),
        "url": f"{settings.cvat_ui_url}/tasks/{job.task_id}/jobs/{job.id}",
    }


def list_projects() -> list[dict[str, Any]]:
    with client() as c:
        return [
            {"id": int(p.id), "name": p.name, "tasks": int(getattr(p, "tasks_count", 0) or 0)}
            for p in c.projects.list()
        ]


def list_users() -> list[dict[str, Any]]:
    with client() as c:
        return [
            {
                "id": int(u.id),
                "username": u.username,
                "email": getattr(u, "email", ""),
                "first_name": getattr(u, "first_name", ""),
                "last_name": getattr(u, "last_name", ""),
                "is_superuser": bool(getattr(u, "is_superuser", False)),
            }
            for u in c.users.list()
        ]


def get_user(user_id: int) -> dict[str, Any]:
    with client() as c:
        user = c.users.retrieve(user_id)
        return {
            "id": int(user.id),
            "username": user.username,
            "email": getattr(user, "email", ""),
            "first_name": getattr(user, "first_name", ""),
            "last_name": getattr(user, "last_name", ""),
            "is_superuser": bool(getattr(user, "is_superuser", False)),
        }


def set_user_password(username: str, password: str) -> None:
    """Change a CVAT user password through the CVAT management command."""

    command = [
        "docker",
        "exec",
        "-i",
        settings.cvat_container,
        "python",
        "manage.py",
        "changepassword",
        username,
    ]
    try:
        result = subprocess.run(
            command,
            input=f"{password}\n{password}\n",
            text=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001 - surface a readable error
        raise CvatError(f"Could not run the CVAT password change: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        raise CvatError("CVAT password change failed: " + " ".join(tail))
    try:
        check = requests.post(
            f"{settings.cvat_url}/api/auth/login",
            json={"username": username, "password": password},
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001 - surface a readable error
        raise CvatError(f"Could not verify the new password: {exc}") from exc
    if check.status_code != 200:
        raise CvatError("CVAT did not accept the new password")


def assign_job(job_id: int, user_id: int | None) -> dict[str, Any]:
    """Assign a job to a CVAT user, or unassign it when ``user_id`` is None."""

    with client() as c:
        try:
            job = c.jobs.retrieve(job_id)
            job.update({"assignee": user_id})
        except Exception as exc:  # noqa: BLE001 - surface a readable CVAT error
            raise CvatError(f"CVAT could not assign the job: {exc}") from exc
        job = c.jobs.retrieve(job_id)
        task = c.tasks.retrieve(job.task_id)
        return job_row(job, task, getattr(task, "project_id", None))


def delete_user(user_id: int) -> None:
    """Delete a CVAT user after releasing the jobs assigned to them."""

    with client() as c:
        try:
            user = c.users.retrieve(user_id)
        except Exception as exc:  # noqa: BLE001 - surface a readable CVAT error
            raise CvatError(f"User #{user_id} not found: {exc}") from exc
        if getattr(user, "is_superuser", False):
            raise CvatError("The CVAT admin account cannot be deleted.")
        for job in c.jobs.list():
            owner = getattr(job, "assignee", None)
            if owner and int(getattr(owner, "id", 0) or 0) == int(user_id):
                job.update({"assignee": None})
        try:
            c.api_client.users_api.destroy(int(user_id))
        except Exception as exc:  # noqa: BLE001 - surface a readable CVAT error
            raise CvatError(f"CVAT could not delete the user: {exc}") from exc


def list_jobs(
    project_id: int | None = None,
    task_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with client() as c:
        tasks = list(c.tasks.list(project_id=project_id)) if project_id else list(c.tasks.list())
        for task in tasks:
            if task_ids and int(task.id) not in set(task_ids):
                continue
            for job in c.jobs.list(task_id=task.id):
                rows.append(job_row(job, task, project_id or getattr(task, "project_id", None)))
    rows.sort(key=lambda item: (item["project_id"] or 0, item["task_id"], item["start_frame"]))
    return rows


def jobs_by_ids(job_ids: list[int]) -> list[dict[str, Any]]:
    """Fetch specific CVAT jobs (used by targeted inference)."""

    rows: list[dict[str, Any]] = []
    with client() as c:
        for job_id in job_ids:
            job = c.jobs.retrieve(int(job_id))
            task = c.tasks.retrieve(job.task_id)
            project_id = getattr(task, "project_id", None)
            rows.append(job_row(job, task, int(project_id) if project_id else None))
    return rows


def find_or_create_project(project_name: str, labels: list[str]) -> int:
    with client() as c:
        for project in c.projects.list():
            if project.name == project_name:
                return int(project.id)
        project = c.projects.create(
            {
                "name": project_name,
                "labels": [{"name": name} for name in labels],
            }
        )
        return int(project.id)


def create_label_batch(
    *,
    name: str,
    image_paths: list[Path],
    job_size: int,
    assignee_ids: list[int] | None = None,
    project_name: str | None = None,
    labels: list[str] | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    if not image_paths:
        raise CvatError("no images supplied")
    with client() as c:
        project = None
        if project_id is not None:
            try:
                project = c.projects.retrieve(int(project_id))
            except Exception:
                project = None
        target_name = project_name or settings.project_name
        if project is None:
            for item in c.projects.list():
                if item.name == target_name:
                    project = item
                    break
        if project is None:
            project = c.projects.create(
                {
                    "name": target_name,
                    "labels": [{"name": label} for label in (labels or settings.labels)],
                }
            )
        task = c.tasks.create_from_data(
            spec={
                "name": name,
                "project_id": int(project.id),
                "segment_size": max(1, int(job_size)),
            },
            resources=[str(path) for path in image_paths],
            data_params={"image_quality": 80, "sorting_method": "predefined"},
            status_check_period=2,
        )
        jobs = sorted(c.jobs.list(task_id=task.id), key=lambda job: int(job.start_frame))
        assigned: list[dict[str, Any]] = []
        ids = [int(value) for value in (assignee_ids or []) if value]
        for index, job in enumerate(jobs):
            if ids:
                try:
                    job.update({"assignee": ids[index % len(ids)]})
                except Exception:
                    # Assignment is best-effort; the task still gets created.
                    pass
            assigned.append(job_row(job, task, int(project.id)))
        return {
            "project_id": int(project.id),
            "task_id": int(task.id),
            "jobs": assigned,
            "project_name": project.name,
            "task_name": task.name,
        }


@contextmanager
def job_context(job_id: int) -> Iterator[tuple[Any, Any, Any]]:
    """Yield ``(client, job, task)`` for one job."""

    with client() as c:
        job = c.jobs.retrieve(job_id)
        task = c.tasks.retrieve(job.task_id)
        yield c, job, task


def get_job_annotations(job_id: int) -> list[Any]:
    with job_context(job_id) as (_, job, _task):
        data = job.get_annotations()
        return list(getattr(data, "shapes", []) or [])


def export_job_yolo(job_id: int, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("exporting CVAT job %s -> %s", job_id, destination)
    with job_context(job_id) as (_, job, _task):
        job.export_dataset(
            "YOLO 1.1",
            str(destination),
            include_images=True,
            status_check_period=2,
        )
    size = destination.stat().st_size if destination.exists() else 0
    logger.info("exported CVAT job %s (%s bytes)", job_id, size)
    return destination


def task_labels(task_id: int) -> dict[str, int]:
    with client() as c:
        task = c.tasks.retrieve(task_id)
        return {label.name: int(label.id) for label in task.get_labels()}


def set_job_shapes(job_id: int, shapes: list[dict[str, Any]]) -> None:
    from cvat_sdk import models

    with job_context(job_id) as (_, job, _task):
        job.set_annotations(
            models.LabeledDataRequest(
                shapes=[models.LabeledShapeRequest(**shape) for shape in shapes]
            )
        )


def import_coco_labels(
    task_id: int,
    source_coco: str | Path,
    selected_paths: list[str],
    label_aliases: dict[str, str] | None = None,
) -> int:
    """Import COCO ground truth into a newly created image task.

    Frame numbers are assigned by the exact upload order; the caller creates the
    task with ``sorting_method='predefined'``. COCO bbox is xywh, CVAT rectangle
    is absolute xyxy. No resize/normalization is applied.
    """

    from cvat_sdk import models

    source = Path(source_coco)
    if not source.exists():
        return 0
    payload = json.loads(source.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    images_by_path = {
        str(image["file_name"]).replace("\\", "/").lstrip("./"): image
        for image in payload.get("images", [])
    }
    shapes: list[dict[str, Any]] = []
    with client() as c:
        task = c.tasks.retrieve(task_id)
        label_by_name = {label.name.lower(): int(label.id) for label in task.get_labels()}
        aliases = {str(key): str(value) for key, value in (label_aliases or {}).items()}
        for frame, relative in enumerate(selected_paths):
            clean = str(relative).replace("\\", "/").lstrip("./")
            image = images_by_path.get(clean)
            if image is None:
                continue
            width = float(image.get("width") or 0)
            height = float(image.get("height") or 0)
            for annotation in by_image.get(int(image["id"]), []):
                raw_name = categories.get(int(annotation["category_id"]))
                if raw_name is None:
                    continue
                label_name = aliases.get(raw_name, raw_name)
                label_id = label_by_name.get(label_name.lower())
                if label_id is None:
                    continue
                x, y, w, h = (float(value) for value in annotation["bbox"])
                x1 = max(0.0, min(x, width)) if width else x
                y1 = max(0.0, min(y, height)) if height else y
                x2 = max(x1, min(x + w, width)) if width else x + w
                y2 = max(y1, min(y + h, height)) if height else y + h
                if x2 - x1 < 0.5 or y2 - y1 < 0.5:
                    continue
                shapes.append(
                    {
                        "frame": frame,
                        "label_id": label_id,
                        "type": "rectangle",
                        "points": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                        "occluded": False,
                    }
                )
        if shapes:
            task.set_annotations(
                models.LabeledDataRequest(
                    shapes=[models.LabeledShapeRequest(**shape) for shape in shapes]
                )
            )
    return len(shapes)


def frame_bytes(task_id: int, frame_id: int) -> bytes:
    with client() as c:
        task = c.tasks.retrieve(task_id)
        stream = task.get_frame(frame_id)
        try:
            return stream.read() if hasattr(stream, "read") else bytes(stream)
        finally:
            if hasattr(stream, "close"):
                stream.close()
