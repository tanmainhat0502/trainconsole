from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, settings


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    pool_root: Path | None
    labels: tuple[str, ...]
    coco: Path | None = None
    label_aliases: dict[str, str] = field(default_factory=dict)
    take_all: bool = False


# Optional per-project overrides.
#
# Add an entry when a CVAT project needs its own pool folder, label list and/or
# COCO ground truth file. The key must match the CVAT project name exactly.
#
# Example:
#
# _EXTRA_PROJECTS["my_project"] = ProjectConfig(
#     name="my_project",
#     pool_root=ROOT / "data" / "my_project" / "images",
#     labels=("ecigar",),
#     coco=ROOT / "data" / "my_project" / "labels.json",
#     # GT category names that differ from the CVAT label names
#     label_aliases={"e_cigarette": "ecigar"},
#     # default the batch modal to "take all images"
#     take_all=True,
# )
_EXTRA_PROJECTS: dict[str, ProjectConfig] = {}


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
