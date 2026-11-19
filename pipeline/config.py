"""YAML configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class ProjectConfig:
    name: str
    display_name: str
    image_root: Path
    annotations: Path
    labels: list[str]
    manifest_csv: Path | None = None
    grouping: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        raw = load_yaml(path)
        # Project files use paths relative to the repository root, not to configs/.
        image_root = resolve_path(raw["image_root"])
        annotations = resolve_path(raw["annotations"])
        manifest_csv = raw.get("manifest_csv")
        return cls(
            name=raw["project"],
            display_name=raw.get("display_name", raw["project"]),
            image_root=image_root,
            annotations=annotations,
            labels=list(raw.get("labels") or []),
            manifest_csv=resolve_path(manifest_csv) if manifest_csv else None,
            grouping=dict(raw.get("grouping") or {}),
            sampling=dict(raw.get("sampling") or {}),
            dataset=dict(raw.get("dataset") or {}),
            raw=raw,
        )


@dataclass
class SplitPolicyConfig:
    id: str
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 0
    stratify_by: list[str] = field(default_factory=lambda: ["source"])
    min_val_boxes_per_class: int = 10
    max_val_fraction: float = 0.3
    lock: bool = True
    allow_val_growth: bool = False
    grouping_version: str = "v1"

    @classmethod
    def load(cls, path: str | Path) -> "SplitPolicyConfig":
        raw = load_yaml(path)
        return cls(
            id=raw.get("id") or Path(path).stem,
            val_fraction=float(raw.get("val_fraction", 0.2)),
            test_fraction=float(raw.get("test_fraction", 0.0)),
            seed=int(raw.get("seed", 0)),
            stratify_by=list(raw.get("stratify_by") or ["source"]),
            min_val_boxes_per_class=int(raw.get("min_val_boxes_per_class", 10)),
            max_val_fraction=float(raw.get("max_val_fraction", 0.3)),
            lock=bool(raw.get("lock", True)),
            allow_val_growth=bool(raw.get("allow_val_growth", False)),
            grouping_version=str(raw.get("grouping_version", "v1")),
        )


@dataclass
class SamplingConfig:
    id: str = "default"
    audit_fraction: float = 0.1
    per_group_cap: int = 3
    min_frame_gap: int = 10
    near_duplicate_cosine: float = 0.995
    source_power: float = 0.5
    hard_fraction: float = 0.0
    min_boxes_per_class: int = 3
    embedding_model: str = "histogram-v1"

    @classmethod
    def load(cls, path: str | Path) -> "SamplingConfig":
        raw = load_yaml(path)
        return cls(
            id=raw.get("id") or Path(path).stem,
            audit_fraction=float(raw.get("audit_fraction", 0.1)),
            per_group_cap=int(raw.get("per_group_cap", 3)),
            min_frame_gap=int(raw.get("min_frame_gap", 10)),
            near_duplicate_cosine=float(raw.get("near_duplicate_cosine", 0.995)),
            source_power=float(raw.get("source_power", 0.5)),
            hard_fraction=float(raw.get("hard_fraction", 0.0)),
            min_boxes_per_class=int(raw.get("min_boxes_per_class", 3)),
            embedding_model=str(raw.get("embedding_model", "histogram-v1")),
        )

    def merged(self, project_sampling: dict[str, Any] | None = None) -> "SamplingConfig":
        if not project_sampling:
            return self
        values = self.__dict__ | dict(project_sampling)
        return SamplingConfig(**values)
