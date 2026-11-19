"""Image helpers: content hashes, dimensions, stable stems, video grouping."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image


_FRAME_RE = re.compile(r"(?:^|[_-])frame[_-]?(\d+)$", re.IGNORECASE)
_VIDEO_FRAME_RE = re.compile(r"[_-]frame[_-]?\d+$", re.IGNORECASE)
_SESSION_RE = re.compile(r"(20\d{6}[_-]\d{6})(?:_[A-Z]{2})?$")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def frame_index(stem: str) -> int | None:
    match = _FRAME_RE.search(stem)
    return int(match.group(1)) if match else None


def video_stem(stem: str) -> str:
    """Collapse ``clip_frame_000123`` and ``clip-frame-123`` to ``clip``."""
    return _VIDEO_FRAME_RE.sub("", stem)


def session_token(stem: str) -> str:
    match = _SESSION_RE.search(stem)
    return match.group(1) if match else ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
