"""Cheap reusable image embeddings.

The default adapter is intentionally dependency-light and CPU-only so the
pipeline can run before a GPU model stack is installed. It captures colour and
coarse spatial layout, which is enough to reject consecutive/near-identical
frames during batch selection.

Production deployments should swap in DINOv2/SigLIP embeddings: the sampler only
needs a normalized float matrix and a stable model name (the cache key includes
the model name).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .types import Sample


DEFAULT_MODEL = "histogram-v1"


def histogram_features(path: str | Path, size: int = 64) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    features: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(pixels[:, :, channel], bins=8, range=(0.0, 1.0))
        histogram = histogram.astype(np.float32)
        histogram /= max(histogram.sum(), 1.0)
        features.append(histogram)

    gray = pixels.mean(axis=2)
    grid = 4
    cell = size // grid
    for row in range(grid):
        for col in range(grid):
            patch = gray[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
            features.append(np.asarray([patch.mean(), patch.std()], dtype=np.float32))

    features.append(pixels.mean(axis=(0, 1)))
    features.append(pixels.std(axis=(0, 1)))
    vector = np.concatenate(features).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def load_or_build_embeddings(
    samples: list[Sample],
    image_root: str | Path,
    cache_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(matrix, keys)`` aligned to the input sample order.

    The cache is invalidated when the model or the set of sample keys changes.
    It is stored as ``.npz`` so it can be copied to another machine without a DB.
    """

    root = Path(image_root)
    cache = Path(cache_path)
    keys = [sample.key for sample in samples]
    if cache.exists():
        try:
            data = np.load(cache, allow_pickle=False)
            if str(data["model"]) == model and list(data["keys"].tolist()) == keys:
                return data["vectors"].astype(np.float32), keys
        except (KeyError, ValueError, OSError):
            pass

    vectors = [histogram_features(root / sample.path) for sample in samples]
    matrix = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, 0), dtype=np.float32)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, model=np.asarray(model), keys=np.asarray(keys), vectors=matrix)
    return matrix, keys


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity for already-normalized rows."""

    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.clip(matrix @ matrix.T, -1.0, 1.0)
