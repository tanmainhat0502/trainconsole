"""Tiny, dependency-light "model" for testing the loop.

It is not a detector. It builds a nearest-neighbour index from the training
snapshot: embeddings + boxes. Inference copies the boxes of the closest indexed
image (preferring the same leakage group).

Why this exists: it lets us exercise snapshot → train → checkpoint → inference →
atomic activation in seconds, without a CUDA/torch environment. The DEIM adapter
can replace it later; the state machine and artifacts stay the same.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..embeddings import DEFAULT_MODEL, histogram_features


def train_index(snapshot_dir: str | Path, model_dir: str | Path) -> dict[str, Any]:
    snapshot_path = Path(snapshot_dir).resolve()
    out = Path(model_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((snapshot_path / "snapshot.json").read_text(encoding="utf-8"))
    image_root = Path(snapshot["image_root"])
    coco = json.loads((snapshot_path / "train" / "annotations.json").read_text(encoding="utf-8"))
    labels = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    keys: list[str] = []
    vectors: list[np.ndarray] = []
    boxes: dict[str, list[dict[str, Any]]] = {}
    groups: dict[str, str] = {}
    sources: dict[str, str] = {}
    for image in coco.get("images", []):
        relative = str(image["file_name"])
        path = image_root / relative
        keys.append(relative)
        vectors.append(histogram_features(path))
        groups[relative] = str(image.get("group") or "")
        sources[relative] = str(image.get("source") or "")
        boxes[relative] = [
            {
                "label_id": int(annotation["category_id"]),
                "label": labels.get(int(annotation["category_id"]), str(annotation["category_id"])),
                "x": float(annotation["bbox"][0]),
                "y": float(annotation["bbox"][1]),
                "w": float(annotation["bbox"][2]),
                "h": float(annotation["bbox"][3]),
            }
            for annotation in annotations.get(int(image["id"]), [])
        ]

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(out / "index.npz", keys=np.asarray(keys), vectors=matrix)
    (out / "boxes.json").write_text(json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
    (out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    info = {
        "kind": "nearest-neighbor-index",
        "embedding_model": DEFAULT_MODEL,
        "images": len(keys),
        "boxes": sum(len(value) for value in boxes.values()),
        "labels": list(labels.values()),
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_hash": snapshot.get("snapshot_hash"),
    }
    (out / "model.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256((out / "index.npz").read_bytes()).hexdigest()
    return {**info, "path": str(out), "sha256": digest}


def load_index(model_dir: str | Path) -> dict[str, Any] | None:
    path = Path(model_dir)
    index_path = path / "index.npz"
    if not index_path.exists():
        return None
    data = np.load(index_path, allow_pickle=False)
    keys = [str(item) for item in data["keys"].tolist()]
    return {
        "keys": keys,
        "vectors": data["vectors"].astype(np.float32),
        "boxes": json.loads((path / "boxes.json").read_text(encoding="utf-8")),
        "groups": json.loads((path / "groups.json").read_text(encoding="utf-8")),
        "info": json.loads((path / "model.json").read_text(encoding="utf-8")),
    }
