"""Leakage-group assignment rules.

Why this module exists
----------------------
A split is only as good as its leakage unit. For DMS/traffic data, consecutive
frames of the same clip, IR/RGB pairs of the same person, and multiple cameras of
the same session are *not* independent samples. Therefore the pipeline splits
**groups**, never individual images.

The policy is data-driven because every source names things differently. It can
use either the optional ``manifest.csv`` (dataset, group, sequence) or filename
templates. Rules are matched with ``fnmatch`` against the source name.
"""

from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .images import frame_index, session_token, video_stem


_PLACEHOLDERS = re.compile(r"\{(\w+)\}")


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text in {"_", "-", "unknown", "None", "null"} else text


@dataclass
class GroupingPolicy:
    default: str = "{source}:{user}:{camera}"
    fallback: str = "{source}:{stem}"
    rules: dict[str, str] = field(default_factory=dict)
    manifest_rows: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, grouping: dict[str, Any] | None, manifest_csv: Path | None = None
    ) -> "GroupingPolicy":
        grouping = grouping or {}
        rows: dict[str, dict[str, str]] = {}
        if manifest_csv and manifest_csv.exists():
            with open(manifest_csv, newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    rel = row.get("subset_relative_path") or row.get("path")
                    if rel:
                        rows[rel] = dict(row)
        return cls(
            default=str(grouping.get("default", "{source}:{user}:{camera}")),
            fallback=str(grouping.get("fallback", "{source}:{stem}")),
            rules={str(k): str(v) for k, v in (grouping.get("rules") or {}).items()},
            manifest_rows=rows,
        )

    def _template_for(self, source: str, dataset: str, row: dict[str, str] | None) -> str:
        for pattern, template in self.rules.items():
            if fnmatch.fnmatch(source, pattern) or fnmatch.fnmatch(dataset, pattern):
                return template
        if row:
            return "{source}:{group}"
        return self.default

    @staticmethod
    def _safe_format(template: str, fields: dict[str, str]) -> str | None:
        """Format a template; return None when any referenced field is empty."""

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            value = _clean(fields.get(name))
            if not value:
                raise KeyError(name)
            return value

        try:
            rendered = _PLACEHOLDERS.sub(replace, template)
        except KeyError:
            return None
        return rendered.strip(": ")

    def group_for(
        self, *, source: str, relative_path: str, extra: dict[str, Any] | None = None
    ) -> tuple[str, str, str]:
        """Return ``(group_key, session, clip)`` for one image."""

        rel = relative_path.replace("\\", "/")
        row = self.manifest_rows.get(rel)
        path = Path(rel)
        stem = path.stem
        dataset = source.split("/", 1)[0]
        fields: dict[str, str] = {
            "source": _clean(source),
            "dataset": _clean(dataset),
            "user": _clean((extra or {}).get("user")),
            "camera": _clean((extra or {}).get("camera")),
            "stem": _clean(stem),
            "video_stem": _clean(video_stem(stem)),
            "session": _clean(session_token(stem)),
            "group": _clean(row.get("group") if row else ""),
            "sequence": _clean(row.get("sequence") if row else ""),
            "folder": _clean(path.parent.as_posix()),
        }
        template = self._template_for(source, dataset, row)
        group = self._safe_format(template, fields)
        if not group:
            group = self._safe_format(self.fallback, fields) or f"{source}:{stem}"
        session = fields["session"] or fields["group"] or fields["user"] or fields["video_stem"]
        clip = fields["sequence"] or fields["video_stem"] or fields["folder"] or stem
        return group, session, clip


def annotate_frames(samples) -> None:
    """Best-effort frame index / clip parsing for samples without manifest data."""
    for sample in samples:
        if sample.frame_index is None:
            sample.frame_index = frame_index(Path(sample.path).stem)
