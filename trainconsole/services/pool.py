from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

from PIL import Image
from sqlalchemy import delete, select

from pipeline.config import SamplingConfig
from pipeline.embeddings import load_or_build_embeddings
from pipeline.sampling.diversity import select_batch
from pipeline.types import Sample

from ..config import settings
from ..db import Database
from ..models import Batch, UploadedImage


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_pool(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        source = parts[0] if parts else "unknown"
        group = f"{source}:{parts[1]}" if len(parts) > 2 else f"{source}:{relative.parent.as_posix()}"
        with Image.open(path) as image:
            width, height = image.size
        digest = hashlib.sha256(str(relative).encode()).hexdigest()
        samples.append(
            Sample(
                key=digest,
                path=relative.as_posix(),
                width=width,
                height=height,
                group=group,
                source=source,
            )
        )
    return samples


def uploaded_paths(database: Database) -> set[str]:
    """Return uploaded image paths, dropping rows for CVAT tasks that no longer exist."""

    with database.session() as db:
        task_ids = {int(value) for value in db.scalars(select(UploadedImage.cvat_task_id))}
    if task_ids:
        try:
            from ..cvat import client as cvat_client

            with cvat_client() as c:
                existing = {int(task.id) for task in c.tasks.list()}
            stale = task_ids - existing
            if stale:
                with database.session() as db:
                    db.execute(
                        delete(UploadedImage).where(UploadedImage.cvat_task_id.in_(stale))
                    )
                    db.execute(delete(Batch).where(Batch.cvat_task_id.in_(stale)))
        except Exception:
            # A CVAT hiccup should not block a preview; fall back to the DB state.
            pass
    with database.session() as db:
        return {str(path) for path in db.scalars(select(UploadedImage.path))}


def _counts(samples: list[Sample], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        value = str(getattr(sample, attribute))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _balanced_order(samples: list[Sample]) -> list[Sample]:
    """Round-robin images across sources so every job gets a diverse mix."""

    buckets: dict[str, deque[Sample]] = {}
    for sample in samples:
        buckets.setdefault(sample.source, deque()).append(sample)
    for queue in buckets.values():
        ordered = sorted(queue, key=lambda sample: sample.path)
        queue.clear()
        queue.extend(ordered)
    keys = sorted(buckets)
    result: list[Sample] = []
    while True:
        progressed = False
        for key in keys:
            if buckets[key]:
                result.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            return result


def _detect_source_coco(root: Path) -> str | None:
    for name in ("labels_instance_split.json", "labels.json"):
        candidate = root / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def selection_preview(
    database: Database,
    *,
    pool_root: Path,
    budget: int,
    seed: int,
    coco_path: Path | None = None,
    take_all: bool = False,
) -> dict:
    root = pool_root.resolve()
    if not root.exists():
        raise ValueError(f"pool root does not exist: {root}")
    samples = scan_pool(root)
    excluded_paths = uploaded_paths(database)
    samples = [sample for sample in samples if str(root / sample.path) not in excluded_paths]
    if not samples:
        raise ValueError("no images left in the pool (all uploaded or empty)")

    use_all = take_all or budget >= len(samples)
    if use_all:
        ordered = _balanced_order(samples)
        report = {
            "budget": budget,
            "selected": len(ordered),
            "per_source": _counts(ordered, "source"),
            "per_reason": {"all": len(ordered)},
            "groups_used": len({sample.group for sample in ordered}),
            "max_per_group": max(
                (_counts(ordered, "group").values()), default=0
            ),
            "strategy": "all-balanced",
        }
        items = [
            {
                "sample_key": sample.key,
                "path": sample.path,
                "absolute_path": str(root / sample.path),
                "source": sample.source,
                "group": sample.group,
                "reason": "all",
                "score": 1.0,
            }
            for sample in ordered
        ]
    else:
        cache_key = hashlib.sha256(
            "|".join(sample.key for sample in samples).encode()
        ).hexdigest()[:16]
        cache = settings.artifact_root / f"embeddings_{cache_key}.npz"
        matrix, _ = load_or_build_embeddings(samples, root, cache, model="histogram-v1")
        config = SamplingConfig(
            id="trainconsole",
            audit_fraction=0.1,
            per_group_cap=12,
            min_frame_gap=10,
            near_duplicate_cosine=0.995,
            source_power=0.5,
            hard_fraction=0.0,
            min_boxes_per_class=0,
            embedding_model="histogram-v1",
        )
        selections, report = select_batch(samples, matrix, budget, config, seed=seed)
        by_key = {sample.key: sample for sample in samples}
        items = [
            {
                "sample_key": item.sample_key,
                "path": by_key[item.sample_key].path,
                "absolute_path": str(root / by_key[item.sample_key].path),
                "source": by_key[item.sample_key].source,
                "group": by_key[item.sample_key].group,
                "reason": item.reason,
                "score": item.score,
            }
            for item in selections
        ]
    preview_id = hashlib.sha256(
        json.dumps(
            {
                "pool": str(root),
                "budget": budget,
                "seed": seed,
                "take_all": take_all,
                "keys": [i["sample_key"] for i in items],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    payload = {
        "preview_id": preview_id,
        "pool_root": str(root),
        "budget": budget,
        "seed": seed,
        "source_coco": (
            str(coco_path) if coco_path is not None and coco_path.is_file() else _detect_source_coco(root)
        ),
        "report": report,
        "items": items,
    }
    path = settings.artifact_root / "previews" / f"{preview_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_preview(preview_id: str) -> dict:
    path = settings.artifact_root / "previews" / f"{preview_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"preview not found: {preview_id}")
    return json.loads(path.read_text(encoding="utf-8"))
