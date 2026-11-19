"""Run CVAT inference for one GPU/job shard (used by multi-GPU inference)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainconsole.config import settings  # noqa: E402
from trainconsole.services.inference import run_inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="infer-cli")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="yolo26")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    job_ids = [int(value) for value in args.job_ids.split(",") if value.strip()]
    counts = run_inference(
        model_path=args.model,
        names=[],
        project_id=args.project_id,
        task_ids=[],
        job_ids=job_ids,
        family=args.family,
        batch=max(1, int(args.batch)),
        confidence=settings.confidence,
        imgsz=settings.imgsz,
        gpu=str(args.gpu),
    )
    Path(args.out).write_text(json.dumps(counts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
