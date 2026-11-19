"""Standalone YOLO26x runner for the tanmai conda env (Python 3.8).

The Train Console worker runs in its own env and invokes this script through
`conda run -n tanmai`. Keeping training out-of-process avoids dependency
conflicts and pins every run to one GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base", default="yolo26x.pt")
    parser.add_argument("--result", required=True)
    parser.add_argument("--status", default="")
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status) if args.status else None
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"epoch": 0, "epochs": args.epochs}), encoding="utf-8")

    model = YOLO(args.base)

    def on_epoch_end(trainer) -> None:
        if status_path is None:
            return
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", args.epochs) or args.epochs)
        status_path.write_text(
            json.dumps({"epoch": epoch, "epochs": total}), encoding="utf-8"
        )

    model.add_callback("on_train_epoch_end", on_epoch_end)
    results = model.train(
        data=args.data,
        epochs=max(1, args.epochs),
        batch=max(1, args.batch),
        imgsz=max(64, args.imgsz),
        device=str(args.gpu),
        workers=max(0, args.workers),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=True,
        plots=False,
        val=True,
    )
    best = run_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("YOLO finished without a checkpoint: %s" % best)

    metrics = {}
    box = getattr(results, "box", None)
    if box is not None:
        for key, attr in (
            ("map50", "map50"),
            ("map50_95", "map"),
            ("precision", "mp"),
            ("recall", "mr"),
        ):
            value = getattr(box, attr, None)
            try:
                if value is not None:
                    metrics[key] = round(float(value), 6)
            except (TypeError, ValueError):
                pass
    if not metrics:
        # DDP runs validation inside the child processes, so the parent's
        # ``results`` carries no metrics. Read the last row of results.csv.
        csv_path = run_dir / "train" / "results.csv"
        try:
            lines = [
                line
                for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            header = [cell.strip() for cell in lines[0].split(",")]
            last = [cell.strip() for cell in lines[-1].split(",")]
            row = dict(zip(header, last))
            for key, column in (
                ("map50", "metrics/mAP50(B)"),
                ("map50_95", "metrics/mAP50-95(B)"),
                ("precision", "metrics/precision(B)"),
                ("recall", "metrics/recall(B)"),
            ):
                if column in row:
                    metrics[key] = round(float(row[column]), 6)
        except Exception:
            pass
    payload = {
        "weights": str(best),
        "base_model": args.base,
        "metrics": {
            **metrics,
            "eval_mode": "train20-seen",
            "seen_map50": metrics.get("map50"),
            "seen_map50_95": metrics.get("map50_95"),
        },
        "gpu": str(args.gpu),
    }
    Path(args.result).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
