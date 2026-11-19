"""Command line entry points for data preparation.

These commands work before the web app is running and are also the reference
implementation used by unit tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .config import ProjectConfig, SamplingConfig, SplitPolicyConfig, resolve_path
from .datasets.snapshot import build_snapshot_from_coco
from .embeddings import load_or_build_embeddings
from .io.coco import load_coco
from .sampling.diversity import select_batch


def _write_json(path: str | Path, payload: Any) -> None:
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print(message: str) -> None:
    print(message, file=sys.stderr)


def cmd_snapshot(args: argparse.Namespace) -> int:
    project = ProjectConfig.load(args.project)
    policy = SplitPolicyConfig.load(args.policy)
    locks = args.locks or (resolve_path(args.out).parent / "group_locks.json")
    result = build_snapshot_from_coco(
        project,
        policy,
        args.out,
        locks_path=locks,
        overwrite=args.overwrite,
    )
    report = result["report"]
    _print(f"snapshot: {result['snapshot_id']}")
    _print(f"path:     {result['path']}")
    for role, values in report["by_role"].items():
        _print(
            f"  {role:5s} images={values['images']:6d} groups={values['groups']:5d} "
            f"boxes={values['boxes']:7d}"
        )
    checks = report["checks"]
    for key in (
        "train_val_sample_overlap",
        "train_test_sample_overlap",
        "val_test_sample_overlap",
    ):
        if checks[key] != 0:
            raise SystemExit(f"LEAKAGE CHECK FAILED: {key}={checks[key]}")
    _print(f"leakage checks: OK · locks={result['locks_path']}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    project = ProjectConfig.load(args.project)
    config = SamplingConfig.load(args.config).merged(project.sampling)
    samples = load_coco(project)
    cache = args.embeddings or resolve_path(
        f"data/derived/{project.name}/embeddings_{config.embedding_model}.npz"
    )
    _print(f"loading embeddings for {len(samples)} images (cache: {cache})")
    matrix, _ = load_or_build_embeddings(
        samples, project.image_root, cache, model=config.embedding_model
    )
    excluded: set[str] = set()
    if args.exclude_labeled:
        excluded = {sample.key for sample in samples if sample.boxes}
    selections, report = select_batch(
        samples,
        matrix,
        args.budget,
        config,
        seed=args.seed,
        excluded_keys=excluded,
    )
    by_key = {sample.key: sample for sample in samples}
    selection_id = hashlib.sha256(
        json.dumps(
            {
                "project": project.name,
                "config": config.__dict__,
                "budget": args.budget,
                "seed": args.seed,
                "keys": [item.sample_key for item in selections],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    payload = {
        "selection_id": f"{project.name}-{selection_id}",
        "project": project.name,
        "budget": args.budget,
        "seed": args.seed,
        "config": config.__dict__,
        "report": report,
        "items": [
            {
                "sample_key": item.sample_key,
                "path": by_key[item.sample_key].path,
                "source": by_key[item.sample_key].source,
                "group": by_key[item.sample_key].group,
                "classes": sorted(by_key[item.sample_key].classes),
                "reason": item.reason,
                "score": item.score,
                "cluster": item.cluster,
            }
            for item in selections
        ],
    }
    out = args.out or f"data/derived/{project.name}/selection_{selection_id}.json"
    _write_json(out, payload)
    _print(f"selection: {payload['selection_id']}")
    _print(f"selected:  {report['selected']}/{args.budget}")
    _print(f"path:      {resolve_path(out)}")
    _print(f"sources:   {report['per_source']}")
    _print(f"reasons:   {report['per_reason']}")
    _print(f"classes:   {report['per_class_images']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report_path = resolve_path(args.snapshot) / "split_report.md"
    if not report_path.exists():
        raise SystemExit(f"no split report at {report_path}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    project = ProjectConfig.load(args.project)
    samples = load_coco(project)
    groups = Counter(sample.group for sample in samples)
    print(f"project:   {project.name}")
    print(f"images:    {len(samples)}")
    print(f"boxes:     {sum(len(sample.boxes) for sample in samples)}")
    print(f"labels:    {project.labels}")
    print(f"groups:    {len(groups)}")
    print(f"sources:   {dict(sorted(Counter(s.source for s in samples).items()))}")
    print(f"group cap: min={min(groups.values()) if groups else 0} max={max(groups.values()) if groups else 0}")
    empty = sum(1 for sample in samples if not sample.boxes)
    print(f"empty GT:  {empty}")
    return 0


def cmd_import_coco(args: argparse.Namespace) -> int:
    """Seed the labeltool database from a COCO file.

    Imported annotations are marked as *manual/reviewed* because they are ground
    truth, not model predictions. Re-importing is idempotent by content hash.
    """

    from labeltool.app.services.seed import seed_labeltool

    project = ProjectConfig.load(args.project)
    result = seed_labeltool(
        project,
        database_url=args.database_url,
        labels=project.labels,
        labeled_fraction=args.labeled_fraction,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_user_create(args: argparse.Namespace) -> int:
    from labeltool.app.services.users import create_user

    print(
        json.dumps(
            create_user(
                username=args.username,
                password=args.password,
                role=args.role,
                display_name=args.display_name or args.username,
                database_url=args.database_url,
            ),
            indent=2,
        )
    )
    return 0


def cmd_user_list(args: argparse.Namespace) -> int:
    from labeltool.app.services.users import list_users

    print(json.dumps(list_users(database_url=args.database_url), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alt", description="auto_label_and_train pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="summarize a project manifest")
    info.add_argument("--project", required=True)
    info.set_defaults(func=cmd_info)

    snapshot = sub.add_parser("snapshot", help="build a frozen split snapshot")
    snapshot.add_argument("--project", required=True)
    snapshot.add_argument("--policy", required=True)
    snapshot.add_argument("--out", required=True)
    snapshot.add_argument("--locks", default=None)
    snapshot.add_argument("--overwrite", action="store_true")
    snapshot.set_defaults(func=cmd_snapshot)

    sample = sub.add_parser("sample", help="select a representative labeling batch")
    sample.add_argument("--project", required=True)
    sample.add_argument("--config", required=True)
    sample.add_argument("--budget", type=int, default=500)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--embeddings", default=None)
    sample.add_argument("--out", default=None)
    sample.add_argument(
        "--exclude-labeled",
        action="store_true",
        help="only select images without ground-truth boxes",
    )
    sample.set_defaults(func=cmd_sample)

    report = sub.add_parser("report", help="print a snapshot split report")
    report.add_argument("--snapshot", required=True)
    report.set_defaults(func=cmd_report)

    seed = sub.add_parser("import-coco", help="import COCO labels into the labeltool DB")
    seed.add_argument("--project", required=True)
    seed.add_argument("--database-url", default=None)
    seed.add_argument(
        "--labeled-fraction",
        type=float,
        default=1.0,
        help="fraction of imported images that get ground-truth labels (rest stay unlabeled)",
    )
    seed.set_defaults(func=cmd_import_coco)

    user = sub.add_parser("user-create", help="create or update a labeltool user")
    user.add_argument("--username", required=True)
    user.add_argument("--password", required=True)
    user.add_argument(
        "--role",
        default="annotator",
        choices=["admin", "manager", "annotator", "reviewer"],
    )
    user.add_argument("--display-name", default="")
    user.add_argument("--database-url", default=None)
    user.set_defaults(func=cmd_user_create)

    users = sub.add_parser("user-list", help="list labeltool users")
    users.add_argument("--database-url", default=None)
    users.set_defaults(func=cmd_user_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
