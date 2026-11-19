"""COCO import/export.

The internal representation uses class *names*, so category id remapping can
never silently change semantics between datasets.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..config import ProjectConfig
from ..types import Box, Role, Sample
from .grouping import GroupingPolicy
from .images import file_sha256, image_size


def _load_manifest_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    import csv

    with open(path, newline="", encoding="utf-8") as stream:
        return {
            (row.get("subset_relative_path") or row.get("path") or ""): dict(row)
            for row in csv.DictReader(stream)
        }


def load_coco(project: ProjectConfig, *, identity: str = "sha256") -> list[Sample]:
    """Read the project's COCO file into canonical samples.

    ``identity='sha256'`` costs one read of every image but gives stable,
    content-addressed IDs (recommended). ``identity='path'`` avoids hashing for
    very large imports, at the cost of weaker duplicate detection.
    """

    with open(project.annotations, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    policy = GroupingPolicy.from_config(project.grouping, project.manifest_csv)
    samples: list[Sample] = []
    for image in payload.get("images", []):
        relative = str(image["file_name"]).replace("\\", "/")
        absolute = project.image_root / relative
        if not absolute.exists():
            raise FileNotFoundError(f"image referenced by {project.annotations} is missing: {absolute}")
        key = file_sha256(absolute) if identity == "sha256" else f"path:{relative}"
        group, session, clip = policy.group_for(
            source=str(image.get("source") or "unknown"),
            relative_path=relative,
            extra=image,
        )
        try:
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError):
            width, height = image_size(absolute)
        boxes: list[Box] = []
        for annotation in annotations_by_image.get(int(image["id"]), []):
            category = categories.get(int(annotation["category_id"]))
            if category is None:
                raise ValueError(
                    f"annotation {annotation.get('id')} uses unknown category "
                    f"{annotation.get('category_id')}"
                )
            x, y, w, h = (float(value) for value in annotation["bbox"])
            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            w = max(0.0, min(w, float(width) - x))
            h = max(0.0, min(h, float(height) - y))
            if w < 0.5 or h < 0.5:
                continue
            boxes.append(Box(label=category, x=x, y=y, w=w, h=h))
        samples.append(
            Sample(
                key=key,
                path=relative,
                width=width,
                height=height,
                group=group,
                source=str(image.get("source") or "unknown"),
                session=session,
                clip=clip,
                frame_index=_frame_index(relative),
                meta={
                    "coco_id": int(image["id"]),
                    "user": image.get("user"),
                    "camera": image.get("camera"),
                },
                boxes=boxes,
            )
        )
    return samples


def _frame_index(path: str) -> int | None:
    import re

    match = re.search(r"[_-]frame[_-]?(\d+)$", Path(path).stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def write_coco(
    samples: Iterable[Sample],
    path: str | Path,
    labels: list[str],
    *,
    image_root: str | Path | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a COCO detection file. Category ids follow the project label order."""

    label_to_id = {name: index for index, name in enumerate(labels)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    sample_list = list(samples)
    for image_id, sample in enumerate(sample_list, start=1):
        unknown = sample.classes - set(labels)
        if unknown:
            raise ValueError(f"sample {sample.key} has labels outside the project schema: {sorted(unknown)}")
        images.append(
            {
                "id": image_id,
                "file_name": sample.path,
                "width": sample.width,
                "height": sample.height,
                "source": sample.source,
                "group": sample.group,
            }
        )
        for box in sample.boxes:
            x, y, w, h = box.as_coco()
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[box.label],
                    "bbox": [x, y, w, h],
                    "area": round(w * h, 3),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    info: dict[str, Any] = {
        "description": "auto_label_and_train snapshot",
        "image_root": str(image_root) if image_root else None,
        "counts": {
            "images": len(images),
            "annotations": len(annotations),
            "classes": len(labels),
        },
    }
    if extra_info:
        info.update(extra_info)
    payload = {
        "info": info,
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(labels)],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return payload


def write_role_files(
    samples: list[Sample],
    roles: dict[str, Role],
    out_dir: str | Path,
    labels: list[str],
    *,
    image_root: str | Path | None = None,
    snapshot_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Write one COCO file per role under ``out_dir/<role>/annotations.json``."""

    out = Path(out_dir)
    by_role: dict[Role, list[Sample]] = defaultdict(list)
    for sample in samples:
        role = roles.get(sample.key, sample.role)
        if role is None:
            raise ValueError(f"sample {sample.key} has no split role")
        by_role[role].append(sample)

    written: dict[str, dict[str, Any]] = {}
    for role in (Role.TRAIN, Role.VAL, Role.TEST):
        subset = by_role.get(role, [])
        if not subset and role is Role.TEST:
            continue
        written[role.value] = write_coco(
            subset,
            out / role.value / "annotations.json",
            labels,
            image_root=image_root,
            extra_info={"role": role.value, "snapshot_id": snapshot_id},
        )
    return written
