"""Core data types shared by the pipeline.

The pipeline is deliberately framework-agnostic. It reads annotation sources into
one canonical representation, applies sampling/splitting policies, then exports
whatever format a trainer adapter asks for.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from typing import Any


class Role(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Box:
    """A bounding box in absolute pixel coordinates (COCO xywh)."""

    label: str
    x: float
    y: float
    w: float
    h: float
    score: float | None = None

    def as_coco(self) -> list[float]:
        return [round(self.x, 3), round(self.y, 3), round(self.w, 3), round(self.h, 3)]


@dataclass
class Sample:
    """One image plus its current (effective) annotations.

    ``key`` is the content hash by default. It is the stable identity used by
    split locks, prediction batches and training manifests, so moving or
    renaming a file never changes its role.
    """

    key: str
    path: str
    width: int
    height: int
    group: str
    source: str
    session: str = ""
    clip: str = ""
    frame_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    boxes: list[Box] = field(default_factory=list)
    role: Role | None = None

    @property
    def class_counts(self) -> Counter[str]:
        return Counter(box.label for box in self.boxes)

    @property
    def classes(self) -> set[str]:
        return set(self.class_counts)

    def annotation_hash(self) -> str:
        payload = {
            "key": self.key,
            "boxes": [
                {"label": b.label, "box": b.as_coco(), "score": b.score}
                for b in sorted(self.boxes, key=lambda b: (b.label, b.x, b.y, b.w, b.h))
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest_row(self, role: Role) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "role": role.value,
            "group": self.group,
            "source": self.source,
            "session": self.session,
            "clip": self.clip,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "class_counts": dict(sorted(self.class_counts.items())),
            "annotation_hash": self.annotation_hash(),
            "meta": self.meta,
        }


@dataclass
class Group:
    key: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def source(self) -> str:
        return self.samples[0].source if self.samples else ""

    @property
    def class_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for sample in self.samples:
            counts.update(sample.class_counts)
        return counts

    @property
    def size(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class Selection:
    sample_key: str
    score: float
    reason: str
    stratum: str
    cluster: int | None = None


@dataclass
class SplitResult:
    roles: dict[str, Role]  # sample key -> role
    groups: dict[str, Role]  # group key -> role
    locks: dict[str, Role]  # group key -> frozen role
    report: dict[str, Any]
