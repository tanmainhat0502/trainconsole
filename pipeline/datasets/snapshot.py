"""Build an immutable dataset snapshot from the current label state.

A snapshot contains *references* to images, never copies. That keeps 1M-image
datasets cheap while still giving every training run a frozen, hash-verifiable
view of images, annotations and split roles.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from ..config import ProjectConfig, SplitPolicyConfig
from ..io.coco import write_role_files
from ..types import Role, Sample
from ..splits.policy import assign_splits, load_locks, render_report, save_locks


def _canonical_hash(samples: list[Sample], roles: dict[str, Role], policy: SplitPolicyConfig) -> str:
    rows = [
        sample.manifest_row(roles[sample.key])
        for sample in sorted(samples, key=lambda item: item.key)
    ]
    payload = {
        "policy": {
            "id": policy.id,
            "seed": policy.seed,
            "val_fraction": policy.val_fraction,
            "test_fraction": policy.test_fraction,
            "stratify_by": policy.stratify_by,
            "grouping_version": policy.grouping_version,
        },
        "samples": rows,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_snapshot(
    samples: list[Sample],
    project: ProjectConfig,
    policy: SplitPolicyConfig,
    out_dir: str | Path,
    *,
    locks_path: str | Path | None = None,
    overwrite: bool = False,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"snapshot directory already exists: {out} (use --overwrite or a new version)"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    lock_file = Path(locks_path) if locks_path else out.parent / "group_locks.json"
    locks = load_locks(lock_file, policy_id=policy.id)
    result = assign_splits(samples, policy, locks=locks)
    roles = result.roles

    snapshot_hash = _canonical_hash(samples, roles, policy)
    snapshot_id = f"{policy.id}-{snapshot_hash[:12]}"

    role_payloads = write_role_files(
        samples,
        roles,
        out,
        project.labels,
        image_root=project.image_root,
        snapshot_id=snapshot_id,
    )

    manifest_rows = [
        sample.manifest_row(roles[sample.key])
        for sample in sorted(samples, key=lambda item: item.key)
    ]
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(out / "manifest.csv", "w", encoding="utf-8") as stream:
        stream.write("key,path,role,group,source,session,clip,frame_index,boxes,annotation_hash\n")
        for row in manifest_rows:
            boxes = sum(row["class_counts"].values())
            stream.write(
                f"{row['key']},{row['path']},{row['role']},{row['group']},"
                f"{row['source']},{row['session']},{row['clip']},{row['frame_index'] or ''},"
                f"{boxes},{row['annotation_hash']}\n"
            )

    snapshot = {
        "snapshot_id": snapshot_id,
        "snapshot_hash": snapshot_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": project.name,
        "image_root": str(project.image_root),
        "labels": project.labels,
        "policy": result.report["policy"],
        "totals": result.report["totals"],
        "by_role": result.report["by_role"],
        "checks": result.report["checks"],
        "warnings": result.report["warnings"],
        "files": {role: f"{role}/annotations.json" for role in role_payloads},
        "locked_groups": {key: role.value for key, role in sorted(result.locks.items())},
        "extra": extra_meta or {},
    }
    _write_json(out / "snapshot.json", snapshot)
    _write_json(out / "split_report.json", result.report)
    (out / "split_report.md").write_text(render_report(result.report), encoding="utf-8")
    _write_json(
        out / "dataset.yaml",
        {
            "snapshot_id": snapshot_id,
            "image_root": str(project.image_root),
            "labels": project.labels,
            "num_classes": len(project.labels),
            "train": {"images": str(project.image_root), "annotations": str(out / "train" / "annotations.json")},
            "val": {"images": str(project.image_root), "annotations": str(out / "val" / "annotations.json")},
            **(
                {"test": {"images": str(project.image_root), "annotations": str(out / "test" / "annotations.json")}}
                if (out / "test" / "annotations.json").exists()
                else {}
            ),
        },
    )
    if policy.lock:
        save_locks(lock_file, result.locks, policy_id=policy.id)

    return {
        "snapshot_id": snapshot_id,
        "snapshot_hash": snapshot_hash,
        "path": str(out),
        "locks_path": str(lock_file),
        "report": result.report,
        "files": snapshot["files"],
    }


def build_snapshot_from_coco(
    project: ProjectConfig,
    policy: SplitPolicyConfig,
    out_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    from ..io.coco import load_coco

    samples = load_coco(project)
    return build_snapshot(samples, project, policy, out_dir, **kwargs)
