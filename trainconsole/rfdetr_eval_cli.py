"""Evaluate an RF-DETR checkpoint in a clean single-GPU process."""

from __future__ import annotations

import argparse
import json

from rfdetr import RFDETR


def main() -> int:
    parser = argparse.ArgumentParser(prog="rfdetr-eval-cli")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = RFDETR.from_checkpoint(
        args.checkpoint, device="cuda:0", trust_checkpoint=True
    )
    metrics = model.evaluate(
        dataset_dir=args.data,
        split="val",
        batch_size=1,
        device="cuda:0",
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
