from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _load_dotenv() -> None:
    """Load ./.env into os.environ (existing variables win)."""

    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _cvat_credentials() -> tuple[str, str]:
    username = os.getenv("TC_CVAT_USERNAME", "") or os.getenv("CVAT_ADMIN_USERNAME", "")
    password = os.getenv("TC_CVAT_PASSWORD", "") or os.getenv("CVAT_ADMIN_PASSWORD", "")
    return username, password


_CVAT_USER, _CVAT_PASS = _cvat_credentials()


@dataclass(frozen=True)
class Settings:
    cvat_url: str = field(default_factory=lambda: _env("TC_CVAT_URL", "http://localhost:51102"))
    cvat_ui_url: str = field(
        default_factory=lambda: _env("TC_CVAT_UI_URL", "http://localhost:8081")
    )
    cvat_username: str = field(default_factory=lambda: _CVAT_USER)
    cvat_password: str = field(default_factory=lambda: _CVAT_PASS)
    cvat_container: str = field(default_factory=lambda: _env("TC_CVAT_CONTAINER", "cvat_server"))
    admin_username: str = field(default_factory=lambda: _env("TC_ADMIN_USERNAME", "admin"))
    admin_password: str = field(default_factory=lambda: _env("TC_ADMIN_PASSWORD", "admin"))
    secret_key: str = field(default_factory=lambda: _env("TC_SECRET_KEY", "trainconsole-dev-secret"))
    database_url: str = field(
        default_factory=lambda: _env("TC_DATABASE_URL", "sqlite:///./var/trainconsole.db")
    )
    data_root: Path = field(default_factory=lambda: Path(_env("TC_DATA_ROOT", "./data")))
    artifact_root: Path = field(
        default_factory=lambda: Path(_env("TC_ARTIFACT_ROOT", "./var/trainconsole"))
    )
    pool_root: Path = field(default_factory=lambda: Path(_env("TC_POOL_ROOT", "./data/pool")))
    labels: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in _env(
                "TC_LABELS", "head,shoulder,hand,seatbelt,cell phone,cigarette"
            ).split(",")
            if item.strip()
        )
    )
    exclude_from_detection: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip().lower()
            for item in _env("TC_EXCLUDE_FROM_DETECTION", "seatbelt").split(",")
            if item.strip()
        )
    )
    project_name: str = field(default_factory=lambda: _env("TC_CVAT_PROJECT", "Train Console"))
    batch_size: int = field(default_factory=lambda: int(_env("TC_BATCH_SIZE", "500")))
    job_size: int = field(default_factory=lambda: int(_env("TC_JOB_SIZE", "50")))
    model: str = field(default_factory=lambda: _env("TC_MODEL", "yolo26x.pt"))
    epochs: int = field(default_factory=lambda: int(_env("TC_EPOCHS", "20")))
    yolo_batch: int = field(default_factory=lambda: int(_env("TC_YOLO_BATCH", "8")))
    imgsz: int = field(default_factory=lambda: int(_env("TC_IMGSZ", "640")))
    nms_iou: float = field(default_factory=lambda: float(_env("TC_NMS_IOU", "0.5")))
    nms_containment: float = field(
        default_factory=lambda: float(_env("TC_NMS_CONTAINMENT", "0.85"))
    )
    max_detections: int = field(default_factory=lambda: int(_env("TC_MAX_DET", "100")))
    gpu: str = field(default_factory=lambda: _env("TC_GPU", "0"))
    train_env: str = field(default_factory=lambda: _env("TC_TRAIN_ENV", "tanmai"))
    train_python: str = field(
        default_factory=lambda: _env("TC_TRAIN_PYTHON", "python3")
    )
    rfdetr_python: str = field(
        default_factory=lambda: _env("TC_RFDETR_PYTHON", "python3")
    )
    rfdetr_model: str = field(
        default_factory=lambda: _env("TC_RFDETR_MODEL", "RFDETR2XLarge")
    )
    rfdetr_resolution: int = field(
        default_factory=lambda: int(_env("TC_RFDETR_RESOLUTION", "880"))
    )
    workers: int = field(default_factory=lambda: int(_env("TC_YOLO_WORKERS", "2")))
    export_cache: bool = field(
        default_factory=lambda: _env("TC_EXPORT_CACHE", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    prune_old_weights: bool = field(
        default_factory=lambda: _env("TC_PRUNE_OLD_WEIGHTS", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    confidence: float = field(default_factory=lambda: float(_env("TC_CONF", "0.25")))
    val_fraction: float = field(default_factory=lambda: float(_env("TC_VAL_FRACTION", "0.2")))

    def ensure_dirs(self) -> None:
        for path in (self.artifact_root, self.artifact_root / "previews"):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def primary_gpu(self) -> str:
        """First GPU of a possibly multi-GPU string like '0,1,3'."""

        first = self.gpu.split(",")[0].strip()
        return first or "0"


settings = Settings()
