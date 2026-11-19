from __future__ import annotations

import json
import os
import random
import secrets
import shutil
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, select
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from ..config import settings
from ..cvat import (
    CvatError,
    assign_job,
    create_label_batch,
    delete_user,
    get_user,
    import_coco_labels,
    list_jobs,
    list_projects,
    list_users,
    set_user_password,
)
from ..db import Database, ensure_sqlite_parent
from ..models import Batch, InferRun, ModelVersion, TrainRun, UploadedImage, UserCredential
from ..projects import project_config
from ..services.label_export import export_coco
from ..services.pool import load_preview, selection_preview
from ..services.training import kill_process_tree


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def asset_url(path: str) -> str:
    """Return a /static URL with a mtime version so browsers never reuse stale JS/CSS."""

    name = path.lstrip("/")
    try:
        stamp = (BASE_DIR / "static" / name).stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"/static/{name}?v={stamp}"


templates.env.globals["asset"] = asset_url


def create_app(*, database_url: str | None = None) -> FastAPI:
    settings.ensure_dirs()
    url = database_url or settings.database_url
    ensure_sqlite_parent(url)
    database = Database(url)
    database.create_all()

    app = FastAPI(title="Train Console", version="0.1.0")
    app.state.database = database
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="tc_session",
        max_age=12 * 3600,
        same_site="lax",
        https_only=False,
    )
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.middleware("http")
    async def static_cache_control(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            # Force revalidation so updated JS/CSS is picked up without a hard refresh.
            response.headers["Cache-Control"] = "no-cache"
        return response

    def user(request: Request) -> str | None:
        return request.session.get("admin")

    def require(request: Request) -> str:
        name = user(request)
        if not name:
            raise HTTPException(status_code=401, detail="admin login required")
        return name

    def db_session():
        with database.session() as session:
            yield session

    def iso_utc(value):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def config_for_project_id(project_id: int | None):
        if project_id is None:
            return project_config(None)
        try:
            for item in list_projects():
                if int(item["id"]) == int(project_id):
                    return project_config(item["name"])
        except Exception:
            pass
        return project_config(None)

    def train_payload(run: TrainRun) -> dict:
        return {
            "id": run.id,
            "name": run.name,
            "project_id": run.cvat_project_id,
            "task_ids": run.task_ids,
            "state": run.state,
            "progress": run.progress,
            "stage": run.stage,
            "config": run.config,
            "dataset_dir": run.dataset_dir,
            "weights_path": run.weights_path,
            "metrics": run.metrics,
            "error": run.error,
            "created_at": iso_utc(run.created_at),
            "started_at": iso_utc(run.started_at),
            "completed_at": iso_utc(run.completed_at),
        }

    def infer_payload(run: InferRun) -> dict:
        return {
            "id": run.id,
            "name": run.name,
            "project_id": run.cvat_project_id,
            "task_ids": run.task_ids,
            "job_ids": run.job_ids or [],
            "gpu": run.gpu,
            "batch": run.batch,
            "model_id": run.model_id,
            "state": run.state,
            "progress": run.progress,
            "stage": run.stage,
            "counts": run.counts,
            "error": run.error,
            "created_at": iso_utc(run.created_at),
            "started_at": iso_utc(run.started_at),
            "completed_at": iso_utc(run.completed_at),
        }

    def model_payload(model: ModelVersion) -> dict:
        return {
            "id": model.id,
            "name": model.name,
            "path": model.path,
            "names": model.names,
            "metrics": model.metrics,
            "family": model.family,
            "state": model.state,
            "project_id": model.cvat_project_id,
            "created_at": iso_utc(model.created_at),
        }

    # ------------------------------------------------------------------ pages
    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if user(request):
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(
            request, "login.html", {"error": None, "url": settings.cvat_url}
        )

    @app.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...)):
        if username == settings.admin_username and password == settings.admin_password:
            request.session["admin"] = username
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Wrong username or password.", "url": settings.cvat_url},
            status_code=401,
        )

    @app.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    def page(request: Request, template: str, page_name: str, **context):
        name = user(request)
        if not name:
            return RedirectResponse("/login", status_code=303)
        return templates.TemplateResponse(
            request,
            template,
            {
                "user": name,
                "page": page_name,
                "default_budget": settings.batch_size,
                "default_job_size": settings.job_size,
                "default_epochs": settings.epochs,
                "default_batch": settings.yolo_batch,
                "default_gpu": settings.gpu,
                **context,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return RedirectResponse("/management", status_code=303)

    @app.get("/management", response_class=HTMLResponse)
    def management_page(request: Request):
        return page(
            request,
            "jobs.html",
            "management",
            default_pool=str(settings.pool_root),
            default_labels=", ".join(settings.labels),
        )

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        return RedirectResponse("/management", status_code=303)

    @app.get("/overview", response_class=HTMLResponse)
    def overview_page(request: Request):
        return RedirectResponse("/management", status_code=303)

    @app.get("/batch", response_class=HTMLResponse)
    def batch_page(request: Request):
        return RedirectResponse("/management?batch=1", status_code=303)

    @app.get("/train", response_class=HTMLResponse)
    def train_page(request: Request):
        return RedirectResponse("/training", status_code=303)

    @app.get("/inference", response_class=HTMLResponse)
    def inference_page(request: Request):
        return RedirectResponse("/training", status_code=303)

    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request):
        return RedirectResponse("/training", status_code=303)

    @app.get("/users", response_class=HTMLResponse)
    def users_page(request: Request):
        return RedirectResponse("/management?users=1", status_code=303)

    @app.get("/training", response_class=HTMLResponse)
    def training_page(request: Request):
        return page(request, "training.html", "training", default_model=settings.model)

    # -------------------------------------------------------------------- api
    @app.get("/api/status")
    def api_status(request: Request):
        require(request)
        try:
            projects = list_projects()
            users = list_users()
            return {
                "ok": True,
                "url": settings.cvat_url,
                "ui_url": settings.cvat_ui_url,
                "projects": projects,
                "users": users,
                "model": settings.model,
            }
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "url": settings.cvat_url, "error": str(exc)},
                status_code=503,
            )

    @app.get("/api/overview")
    def api_overview(request: Request, project_id: int | None = None, session=Depends(db_session)):
        require(request)
        try:
            jobs = list_jobs(project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        by_state: dict[str, int] = {}
        by_user: dict[str, dict[str, int]] = {}
        for job in jobs:
            by_state[job["state"]] = by_state.get(job["state"], 0) + 1
            name = job["assignee"] or "unassigned"
            entry = by_user.setdefault(
                name, {"jobs": 0, "completed": 0, "frames": 0, "completed_frames": 0}
            )
            entry["jobs"] += 1
            entry["frames"] += int(job.get("frames") or 0)
            if job["state"] == "completed":
                entry["completed"] += 1
                entry["completed_frames"] += int(job.get("frames") or 0)
        runs = list(
            session.scalars(select(TrainRun).order_by(TrainRun.id.desc()).limit(5))
        )
        models = list(
            session.scalars(
                select(ModelVersion)
                .where(ModelVersion.state == "production")
                .order_by(ModelVersion.id.desc())
                .limit(1)
            )
        )
        return {
            "jobs": len(jobs),
            "by_state": by_state,
            "by_user": by_user,
            "train_runs": [train_payload(run) for run in runs],
            "production_model": model_payload(models[0]) if models else None,
        }

    @app.get("/api/projects")
    def api_projects(request: Request):
        require(request)
        try:
            projects = list_projects()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        for item in projects:
            config = project_config(item["name"])
            item["pool_root"] = str(config.pool_root) if config.pool_root else ""
            item["labels"] = list(config.labels)
            item["has_coco"] = bool(config.coco and config.coco.is_file())
            item["take_all"] = bool(config.take_all)
        return projects

    @app.get("/api/users")
    def api_users(request: Request, session=Depends(db_session)):
        require(request)
        try:
            users = list_users()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        credentials = {
            item.username: item.password
            for item in session.scalars(select(UserCredential))
        }
        for user in users:
            user["password"] = credentials.get(user["username"])
        return users

    @app.post("/api/users")
    async def api_create_user(request: Request, session=Depends(db_session)):
        require(request)
        payload = await request.json()
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        email = str(payload.get("email") or "").strip()
        if len(username) < 3 or len(password) < 8:
            raise HTTPException(
                status_code=400,
                detail="username must be >= 3 characters, password >= 8 characters",
            )
        body = {
            "username": username,
            "password1": password,
            "password2": password,
            "email": email or f"{username}@example.local",
            # CVAT rejects blank first/last names, so fall back to the username.
            "first_name": str(payload.get("first_name") or "").strip() or username,
            "last_name": str(payload.get("last_name") or "").strip() or username,
        }
        try:
            response = requests.post(
                f"{settings.cvat_url}/api/auth/register",
                json=body,
                timeout=20,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"CVAT did not respond: {exc}") from exc
        if response.status_code not in (200, 201):
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            if isinstance(detail, dict):
                detail = "; ".join(
                    f"{key}: {value}" for key, value in detail.items() if value
                )
            raise HTTPException(status_code=400, detail=str(detail))
        credential = session.scalar(
            select(UserCredential).where(UserCredential.username == username)
        )
        if credential is None:
            session.add(UserCredential(username=username, password=password))
        else:
            credential.password = password
        return {
            "ok": True,
            "username": username,
            "password": password,
            "username_hint": "Labelers sign in to CVAT with this username and password.",
        }

    @app.delete("/api/users/{user_id}")
    def api_delete_user(user_id: int, request: Request, session=Depends(db_session)):
        require(request)
        try:
            target = get_user(user_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"user not found: {exc}") from exc
        try:
            delete_user(user_id)
        except CvatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        session.execute(
            delete(UserCredential).where(UserCredential.username == target["username"])
        )
        return {"ok": True, "id": user_id}

    @app.patch("/api/users/{user_id}/password")
    async def api_user_password(
        user_id: int, request: Request, session=Depends(db_session)
    ):
        require(request)
        payload = await request.json()
        password = str(payload.get("password") or "").strip() or secrets.token_urlsafe(9)
        if len(password) < 8:
            raise HTTPException(
                status_code=400, detail="password must be >= 8 characters"
            )
        try:
            target = get_user(user_id)
            set_user_password(target["username"], password)
        except CvatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        credential = session.scalar(
            select(UserCredential).where(UserCredential.username == target["username"])
        )
        if credential is None:
            session.add(UserCredential(username=target["username"], password=password))
        else:
            credential.password = password
        return {
            "id": user_id,
            "username": target["username"],
            "password": password,
        }

    @app.patch("/api/jobs/{job_id}/assignee")
    async def api_assign_job(job_id: int, request: Request):
        require(request)
        payload = await request.json()
        raw = payload.get("user_id")
        try:
            user_id = int(raw) if raw not in (None, "", 0, "0") else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid user_id") from exc
        try:
            return assign_job(job_id, user_id)
        except CvatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/preview")
    async def api_preview(request: Request, session=Depends(db_session)):
        require(request)
        payload = await request.json()
        project_id = payload.get("project_id")
        config = config_for_project_id(int(project_id)) if project_id else project_config(None)
        pool_root = payload.get("pool_root")
        if not pool_root:
            pool_root = str(config.pool_root) if config.pool_root else str(settings.pool_root)
        try:
            return selection_preview(
                database,
                pool_root=Path(pool_root),
                budget=int(payload.get("budget") or settings.batch_size),
                seed=int(payload.get("seed") or 0),
                coco_path=config.coco,
                take_all=bool(payload.get("take_all")),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/batches")
    async def api_create_batch(request: Request, session=Depends(db_session)):
        require(request)
        payload = await request.json()
        preview = load_preview(str(payload["preview_id"]))
        root = Path(preview["pool_root"]).resolve()
        paths = []
        for item in preview["items"]:
            path = Path(item["absolute_path"]).resolve()
            if not path.is_relative_to(root):
                raise HTTPException(status_code=400, detail="preview path escaped pool root")
            paths.append(path)
        assignees = [int(value) for value in (payload.get("assignee_ids") or []) if value]
        project_id = payload.get("project_id")
        config = config_for_project_id(int(project_id)) if project_id else project_config(None)
        try:
            created = create_label_batch(
                name=payload.get("name") or f"Batch {datetime.now():%Y-%m-%d %H:%M}",
                image_paths=paths,
                job_size=int(payload.get("job_size") or settings.job_size),
                assignee_ids=assignees,
                project_name=config.name,
                labels=list(config.labels),
                project_id=int(project_id) if project_id else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        imported = 0
        if payload.get("import_labels", True) and preview.get("source_coco"):
            try:
                imported = import_coco_labels(
                    created["task_id"],
                    preview["source_coco"],
                    [item["path"] for item in preview["items"]],
                    label_aliases=config.label_aliases,
                )
            except Exception as exc:
                # Task creation already succeeded; report the import failure but
                # keep the task usable.
                imported = 0
                created["import_error"] = str(exc)
        batch = Batch(
            name=payload.get("name") or "Batch",
            pool_root=preview["pool_root"],
            cvat_project_id=created["project_id"],
            cvat_task_id=created["task_id"],
            budget=int(preview["budget"]),
            job_size=int(payload.get("job_size") or settings.job_size),
            seed=int(preview.get("seed") or 0),
            status="created",
            report=preview.get("report") or {},
            items=preview.get("items") or [],
        )
        session.add(batch)
        session.flush()
        for index, item in enumerate(preview["items"]):
            session.add(
                UploadedImage(
                    batch_id=batch.id,
                    cvat_task_id=created["task_id"],
                    path=item["absolute_path"],
                    frame_index=index,
                )
            )
        session.flush()
        return {"batch_id": batch.id, **created, "imported_annotations": imported}

    @app.get("/api/jobs")
    def api_jobs(
        request: Request,
        project_id: int | None = None,
        state: str | None = None,
        user_name: str | None = None,
    ):
        require(request)
        try:
            rows = list_jobs(project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if state:
            rows = [row for row in rows if row["state"] == state]
        if user_name:
            rows = [row for row in rows if row["assignee"] == user_name]
        return rows

    @app.get("/api/jobs/summary")
    def api_jobs_summary(request: Request, project_id: int):
        require(request)
        try:
            jobs = list_jobs(project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        states: dict[str, int] = {}
        frames_total = 0
        frames_completed = 0
        for job in jobs:
            states[job["state"]] = states.get(job["state"], 0) + 1
            frames = int(job.get("frames") or 0)
            frames_total += frames
            if job["state"] == "completed":
                frames_completed += frames
        config = config_for_project_id(project_id)
        pool_root = config.pool_root
        pool_total = 0
        if pool_root is not None and pool_root.exists():
            pool_total = sum(
                1
                for path in pool_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        return {
            "jobs_total": len(jobs),
            "jobs_by_state": states,
            "pool_total": pool_total,
            "on_cvat": frames_total,
            "labeled": frames_completed,
            "unlabeled": max(0, frames_total - frames_completed),
            "not_on_cvat": max(0, pool_total - frames_total),
        }

    @app.get("/api/jobs/user-progress")
    def api_jobs_user_progress(request: Request, project_id: int):
        require(request)
        try:
            jobs = list_jobs(project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        stats: dict[str, dict] = {}
        for job in jobs:
            name = job["assignee"] or "unassigned"
            entry = stats.setdefault(
                name,
                {
                    "assignee": name,
                    "jobs": 0,
                    "completed_jobs": 0,
                    "frames": 0,
                    "completed_frames": 0,
                },
            )
            frames = int(job.get("frames") or 0)
            entry["jobs"] += 1
            entry["frames"] += frames
            if job["state"] == "completed":
                entry["completed_jobs"] += 1
                entry["completed_frames"] += frames
        rows = list(stats.values())
        named = sorted(
            (row for row in rows if row["assignee"] != "unassigned"),
            key=lambda row: row["assignee"].lower(),
        )
        return named + [row for row in rows if row["assignee"] == "unassigned"]

    @app.get("/api/labels/export")
    def api_export_labels(request: Request, project_id: int, scope: str = "completed"):
        require(request)
        if scope not in {"completed", "all"}:
            raise HTTPException(status_code=400, detail="scope must be 'completed' or 'all'")
        try:
            payload = export_coco(project_id, scope)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"CVAT export failed: {exc}") from exc
        filename = f"labels_{project_id}_{scope}.json"
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/inference/candidates")
    def api_inference_candidates(request: Request, project_id: int):
        require(request)
        try:
            rows = list_jobs(project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        new_jobs = [row for row in rows if row["state"] == "new"]
        return {
            "count": len(new_jobs),
            "jobs": new_jobs,
            "excluded": [row for row in rows if row["state"] != "new"],
        }

    @app.get("/api/compare/latest")
    def api_compare_latest(request: Request, project_id: int):
        require(request)
        path = (
            settings.artifact_root
            / "compare"
            / f"project_{int(project_id)}.json"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no previous compare")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"compare cache invalid: {exc}") from exc

    @app.post("/api/compare")
    async def api_compare(request: Request, session=Depends(db_session)):
        """Run YOLO26 + RF-DETR on random pool images; read-only, no CVAT writes."""

        require(request)
        payload = await request.json()
        project_id = int(payload["project_id"])
        count = max(1, min(int(payload.get("count") or 50), 200))
        gpu_raw = str(payload.get("gpu") or settings.gpu)
        gpus = [
            part.strip()
            for part in gpu_raw.replace(" ", ",").split(",")
            if part.strip()
        ]
        if not gpus:
            gpus = [settings.primary_gpu]
        for gpu in gpus:
            if not gpu.isdigit():
                raise HTTPException(status_code=400, detail=f"invalid GPU id: {gpu}")
        batch = max(1, int(payload.get("batch") or settings.yolo_batch))

        active_train = session.scalar(
            select(TrainRun)
            .where(TrainRun.state.in_(("queued", "running", "cancelling")))
            .limit(1)
        )
        active_infer = session.scalar(
            select(InferRun)
            .where(InferRun.state.in_(("queued", "running")))
            .limit(1)
        )
        if active_train is not None or active_infer is not None:
            raise HTTPException(
                status_code=409,
                detail="wait for the current training/inference run to finish",
            )

        yolo = session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.cvat_project_id == project_id,
                ModelVersion.family == "yolo26",
            )
            .order_by(ModelVersion.id.desc())
            .limit(1)
        )
        detr = session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.cvat_project_id == project_id,
                ModelVersion.family == "rfdetr",
            )
            .order_by(ModelVersion.id.desc())
            .limit(1)
        )
        if yolo is None or detr is None:
            raise HTTPException(
                status_code=400,
                detail="train both YOLO26 and RF-DETR-2XL before comparing",
            )

        try:
            with cvat_client() as c:
                project = c.projects.retrieve(project_id)
                project_name = project.name
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"CVAT project lookup failed: {exc}") from exc
        config = project_config(project_name)
        pool = config.pool_root or settings.pool_root
        coco = config.coco
        if pool is None or not Path(pool).is_dir():
            raise HTTPException(status_code=400, detail="project pool root not found")
        if coco is None or not Path(coco).is_file():
            raise HTTPException(status_code=400, detail="project COCO ground truth not found")

        compare_root = settings.artifact_root / "compare"
        compare_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="compare-", dir=compare_root))
        result_path = temp_dir / "result.json"
        script = Path(__file__).resolve().parents[1] / "compare_cli.py"
        command = [
            settings.rfdetr_python,
            str(script),
            "--pool", str(pool),
            "--coco", str(coco),
            "--yolo", yolo.path,
            "--detr", detr.path,
            "--count", str(count),
            "--conf", str(settings.confidence),
            "--imgsz", str(settings.imgsz),
            "--gpus", ",".join(gpus),
            "--batch", str(batch),
            "--seed", str(random.randint(1, 10**9)),
            "--nms-iou", str(settings.nms_iou),
            "--nms-containment", str(settings.nms_containment),
            "--out", str(result_path),
        ]
        env = {**os.environ}

        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )

        try:
            completed = await run_in_threadpool(_run)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()[-3000:]
                raise HTTPException(
                    status_code=500, detail=f"compare failed: {detail}"
                )
            data = json.loads(result_path.read_text(encoding="utf-8"))
            data["model_yolo"] = yolo.name
            data["model_detr"] = detr.name
            data["created_at"] = datetime.now(timezone.utc).isoformat()
            cache_path = compare_root / f"project_{project_id}.json"
            temp_cache = cache_path.with_suffix(".tmp")
            temp_cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(temp_cache, cache_path)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="compare timed out") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return data

    @app.post("/api/train-runs")
    async def api_train(request: Request, session=Depends(db_session)):
        require(request)
        payload = await request.json()
        gpu_raw = str(payload.get("gpu") or settings.gpu)
        gpus = ",".join(
            part.strip() for part in gpu_raw.replace(" ", ",").split(",") if part.strip()
        )
        if not gpus:
            gpus = settings.gpu
        for token in gpus.split(","):
            if not token.isdigit():
                raise HTTPException(
                    status_code=400, detail=f"invalid GPU id: {token}"
                )
        family = str(payload.get("family") or "yolo26")
        if family not in {"yolo26", "rfdetr"}:
            raise HTTPException(status_code=400, detail=f"invalid model family: {family}")
        label = "RF-DETR-2XL" if family == "rfdetr" else "YOLO26x"
        run = TrainRun(
            name=payload.get("name") or f"{label} {datetime.now():%Y-%m-%d %H:%M}",
            cvat_project_id=int(payload["project_id"]),
            task_ids=[int(value) for value in (payload.get("task_ids") or [])],
            state="queued",
            stage="waiting for the worker",
            config={
                "family": family,
                "epochs": int(payload.get("epochs") or settings.epochs),
                "batch": int(payload.get("batch") or settings.yolo_batch),
                "imgsz": int(payload.get("imgsz") or settings.imgsz),
                "gpu": gpus,
                "workers": int(payload.get("workers") or settings.workers),
            },
        )
        session.add(run)
        session.flush()
        return train_payload(run)

    @app.post("/api/gpu/test")
    async def api_gpu_test(request: Request, session=Depends(db_session)):
        """Suggest YOLO batch and show the last RF-DETR auto-batch."""

        require(request)
        payload = await request.json()
        gpu_raw = str(payload.get("gpu") or "")
        gpus = [part.strip() for part in gpu_raw.replace(" ", ",").split(",") if part.strip()]
        if not gpus:
            raise HTTPException(status_code=400, detail="GPU is required")
        for token in gpus:
            if not token.isdigit():
                raise HTTPException(status_code=400, detail=f"invalid GPU id: {token}")
        gpus = [int(token) for token in gpus]
        imgsz = int(payload.get("imgsz") or settings.imgsz)
        project_id = payload.get("project_id")

        # The YOLO probe must never receive an RF-DETR checkpoint.
        weights = settings.model
        if project_id:
            newest = session.scalar(
                select(ModelVersion)
                .where(
                    ModelVersion.cvat_project_id == int(project_id),
                    ModelVersion.family == "yolo26",
                )
                .order_by(ModelVersion.id.desc())
                .limit(1)
            )
            if newest is not None and Path(newest.path).is_file():
                weights = newest.path

        cache_file = settings.artifact_root / "gpu_probe_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache: dict = {}
        if cache_file.is_file():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
        key = f"v2:{weights}:{imgsz}"
        probe = cache.get(key)
        was_cached = probe is not None
        yolo_error: str | None = None
        if not probe:
            spec = json.dumps({"weights": weights, "imgsz": imgsz, "gpus": gpus})
            script = Path(__file__).resolve().parents[1] / "gpu_probe.py"
            try:
                out = subprocess.run(
                    [settings.train_python, str(script), spec],
                    cwd=str(Path.cwd()),
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
            except subprocess.TimeoutExpired:
                yolo_error = "GPU test timed out"
            except FileNotFoundError as exc:
                yolo_error = f"training env python not found: {exc}"
            else:
                if out.returncode != 0:
                    detail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
                    yolo_error = "GPU test failed: " + " ".join(detail)
                else:
                    try:
                        probe = json.loads(out.stdout.strip().splitlines()[-1])
                    except Exception as exc:
                        yolo_error = f"GPU test output invalid: {exc}"
                    else:
                        if probe.get("error"):
                            yolo_error = str(probe["error"])
                            probe = None
                        else:
                            cache[key] = probe
                            cache_file.write_text(json.dumps(cache, indent=2))

        try:
            free_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=15,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"nvidia-smi failed: {exc}") from exc
        free: dict[int, int] = {}
        for line in free_out.strip().splitlines():
            idx, mib = [part.strip() for part in line.split(",")]
            free[int(idx)] = int(mib) * 1024 * 1024
        missing = [str(g) for g in gpus if g not in free]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"GPU {','.join(missing)} not found on this machine"
            )

        yolo_result: dict = {"suggested": 0, "per_gpu": {}, "error": yolo_error}
        if probe:
            per_image = float(probe.get("per_image_bytes") or 0)
            overhead = float(probe.get("overhead_bytes") or 0)
            per_gpu = {}
            for g in gpus:
                budget = free[g] * 0.9 - overhead
                per_gpu[str(g)] = (
                    max(0, int(budget / per_image) // 2 * 2)
                    if per_image > 0 and budget > 0
                    else 0
                )
            yolo_result = {
                "suggested": min(per_gpu.values()) if per_gpu else 0,
                "per_gpu": per_gpu,
                "error": None,
            }

        rfdetr_result: dict = {"batch": None, "grad_accum": None, "source": None}
        if project_id:
            runs = list(
                session.scalars(
                    select(TrainRun)
                    .where(
                        TrainRun.cvat_project_id == int(project_id),
                        TrainRun.state == "completed",
                    )
                    .order_by(TrainRun.id.desc())
                    .limit(30)
                )
            )
            for run in runs:
                if str((run.config or {}).get("family") or "yolo26") != "rfdetr":
                    continue
                cfg_path = (
                    settings.artifact_root
                    / "train_runs"
                    / str(run.id)
                    / "train"
                    / "training_config.json"
                )
                if not cfg_path.is_file():
                    continue
                try:
                    train_cfg = json.loads(cfg_path.read_text(encoding="utf-8")).get(
                        "train_config", {}
                    )
                except Exception:
                    continue
                rfdetr_result = {
                    "batch": train_cfg.get("batch_size"),
                    "grad_accum": train_cfg.get("grad_accum_steps"),
                    "source": f"run {run.id}",
                }
                break

        print(
            "gpu test "
            f"gpus={gpus} imgsz={imgsz} "
            f"yolo_batch={yolo_result.get('suggested')} "
            f"rfdetr_batch={rfdetr_result.get('batch')}",
            flush=True,
        )
        free_gb = {str(g): round(free[g] / 1024**3, 1) for g in gpus}
        return {
            "suggested": yolo_result.get("suggested") or 0,
            "per_gpu": yolo_result.get("per_gpu") or {},
            "free_gb": free_gb,
            "imgsz": imgsz,
            "gpus": [str(g) for g in gpus],
            "cached": was_cached,
            "yolo": yolo_result,
            "rfdetr": rfdetr_result,
        }

    @app.get("/api/train-runs")
    def api_train_runs(request: Request, session=Depends(db_session)):
        require(request)
        rows = list(session.scalars(select(TrainRun).order_by(TrainRun.id.desc()).limit(30)))
        return [train_payload(row) for row in rows]

    @app.post("/api/train-runs/{run_id}/cancel")
    def api_cancel_train(run_id: int, request: Request, session=Depends(db_session)):
        require(request)
        run = session.get(TrainRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="training run not found")
        if run.state == "queued":
            run.state = "cancelled"
            run.stage = "cancelled by admin"
            run.completed_at = datetime.now(timezone.utc)
        elif run.state in {"running", "cancelling"}:
            # kill the training tree directly so cancel is instant, even if
            # the worker restarted or the launcher pid is already dead
            pid = (run.config or {}).get("pid")
            kill_process_tree(
                int(pid) if pid else None,
                run_marker=f"train_runs/{run.id}/",
            )
            run.state = "cancelled"
            run.stage = "cancelled by admin"
            run.completed_at = datetime.now(timezone.utc)
        else:
            raise HTTPException(
                status_code=400, detail=f"cannot cancel a {run.state} run"
            )
        session.flush()
        return train_payload(run)

    @app.post("/api/infer-runs")
    async def api_infer(request: Request, session=Depends(db_session)):
        require(request)
        payload = await request.json()
        project_id = int(payload["project_id"])
        job_ids: list[int] = []
        for value in payload.get("job_ids") or []:
            try:
                job_id = int(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"invalid job id: {value}")
            if job_id <= 0:
                raise HTTPException(status_code=400, detail=f"invalid job id: {value}")
            if job_id not in job_ids:
                job_ids.append(job_id)
        gpu_raw = str(payload.get("gpu") or settings.gpu)
        gpus = ",".join(
            part.strip()
            for part in gpu_raw.replace(" ", ",").split(",")
            if part.strip()
        )
        if not gpus:
            gpus = settings.gpu
        for token in gpus.split(","):
            if not token.isdigit():
                raise HTTPException(status_code=400, detail=f"invalid GPU id: {token}")
        batch = max(1, int(payload.get("batch") or settings.yolo_batch))
        # Inference uses the newest weights of the requested family.
        family = payload.get("family")
        if family is not None and family not in {"yolo26", "rfdetr"}:
            raise HTTPException(status_code=400, detail=f"invalid model family: {family}")
        query = select(ModelVersion).where(
            ModelVersion.cvat_project_id == project_id
        )
        if family:
            query = query.where(ModelVersion.family == family)
        latest = session.scalar(query.order_by(ModelVersion.id.desc()).limit(1))
        if latest is None:
            raise HTTPException(
                status_code=400,
                detail=f"no trained model for {family or 'this project'}",
            )
        run = InferRun(
            name=payload.get("name") or f"Infer {datetime.now():%Y-%m-%d %H:%M}",
            cvat_project_id=project_id,
            task_ids=[int(value) for value in (payload.get("task_ids") or [])],
            job_ids=job_ids,
            gpu=gpus,
            batch=batch,
            model_id=latest.id if latest else None,
            state="queued",
            stage="waiting for the worker",
        )
        session.add(run)
        session.flush()
        return infer_payload(run)

    @app.get("/api/infer-runs")
    def api_infer_runs(request: Request, session=Depends(db_session)):
        require(request)
        rows = list(session.scalars(select(InferRun).order_by(InferRun.id.desc()).limit(30)))
        return [infer_payload(row) for row in rows]

    @app.get("/api/models")
    def api_models(
        request: Request,
        project_id: int | None = None,
        session=Depends(db_session),
    ):
        require(request)
        query = select(ModelVersion).order_by(ModelVersion.id.desc())
        if project_id is not None:
            query = query.where(ModelVersion.cvat_project_id == project_id)
        rows = list(session.scalars(query))
        return [model_payload(row) for row in rows]

    @app.post("/api/models/{model_id}/promote")
    def api_promote(model_id: int, request: Request, session=Depends(db_session)):
        require(request)
        model = session.get(ModelVersion, model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="model not found")
        for item in session.scalars(select(ModelVersion).where(ModelVersion.state == "production")):
            item.state = "retired"
        model.state = "production"
        session.flush()
        return model_payload(model)

    @app.get("/health")
    def health():
        return {"ok": True, "service": "trainconsole"}

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("trainconsole.app.main:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    run()
