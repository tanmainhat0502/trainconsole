from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    pool_root: Path | None
    labels: tuple[str, ...]
    coco: Path | None = None
    label_aliases: dict[str, str] = field(default_factory=dict)
    take_all: bool = True


_EXTRA_PROJECTS: dict[str, ProjectConfig] = {
    "thu_xi_ga_project": ProjectConfig(
        name="thu_xi_ga_project",
        pool_root=Path("/tmp/craft_task07_review/ecig_valid_relabel/images"),
        labels=("ecigar",),
        coco=Path("/tmp/craft_task07_review/ecig_valid_relabel/valid_ecig_gt.json"),
        # The COCO ground truth uses "e_cigarette"; the CVAT project label is "ecigar".
        label_aliases={"e_cigarette": "ecigar"},
        take_all=True,
    ),
}


def project_config(name: str | None) -> ProjectConfig:
    if name and name in _EXTRA_PROJECTS:
        return _EXTRA_PROJECTS[name]
    is_default = name is None or name == settings.project_name
    coco = None
    if is_default:
        for candidate in ("labels_instance_split.json", "labels.json"):
            path = settings.pool_root / candidate
            if path.is_file():
                coco = path
                break
    return ProjectConfig(
        name=name or settings.project_name,
        pool_root=settings.pool_root if is_default else None,
        labels=settings.labels,
        coco=coco,
    )
