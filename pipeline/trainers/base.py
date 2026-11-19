"""Trainer adapter contract.

A trainer is *not* imported into the labeltool process. It is a subprocess
recipe, so every project can use its own environment (DEIM, Ultralytics,
MMDetection, a private repo) without dependency conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml


@dataclass
class TrainerProfile:
    id: str
    kind: str = "subprocess"
    repo: str = "repo"
    python: str = "python"
    workdir: str = "repo"
    dataset_format: str = "coco"
    config_template: str | None = None
    command: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TrainerProfile":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            id=raw.get("id") or Path(path).stem,
            kind=raw.get("kind", "subprocess"),
            repo=raw.get("repo", "repo"),
            python=raw.get("python", "python"),
            workdir=raw.get("workdir", "repo"),
            dataset_format=raw.get("dataset_format", "coco"),
            config_template=raw.get("config_template"),
            command=list(raw.get("command") or []),
            metrics=dict(raw.get("metrics") or {}),
            weights=dict(raw.get("weights") or {}),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        )


@dataclass
class PreparedTrainer:
    command: list[str]
    workdir: Path
    config_path: Path | None
    env: dict[str, str]
    profile: TrainerProfile
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainerAdapter(Protocol):
    def prepare(
        self,
        *,
        profile: TrainerProfile,
        snapshot_dir: Path,
        run_dir: Path,
        options: dict[str, Any],
    ) -> PreparedTrainer: ...
