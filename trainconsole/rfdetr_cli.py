"""Train RF-DETR-2XL in its own Python environment.

The console calls this script from ``train_rfdetr``. It builds nothing itself;
the worker already produced a YOLO-layout dataset that RF-DETR auto-detects.
Outputs the same ``result.json`` contract as ``yolo_cli.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def _write_status(path: Path | None, epoch: int, total: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"epoch": epoch, "epochs": total}), encoding="utf-8"
    )


def _dataset_names(data_dir: Path) -> list[str]:
    payload = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    names = payload.get("names") or []
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(value) for value in names]


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    for key in (f"val/{name}", f"test/{name}", name):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    for key, value in metrics.items():
        if str(key).endswith(f"/{name}"):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _find_best(output: Path) -> Path:
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "last_ema.pth",
    ):
        candidate = output / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no RF-DETR checkpoint in {output}")


def _prune_checkpoints(output: Path, keep: Path | None) -> None:
    keep_resolved = keep.resolve() if keep is not None else None
    for pattern in ("*.pth", "*.ckpt"):
        for path in output.rglob(pattern):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if keep_resolved is not None and resolved == keep_resolved:
                continue
            try:
                path.unlink()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="rfdetr-cli")
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--resolution", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", default="RFDETR2XLarge")
    parser.add_argument("--base", default="")
    parser.add_argument("--result", required=True)
    parser.add_argument("--status", default="")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = run_dir / "train"
    output.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status) if args.status else None
    _write_status(status_path, 0, args.epochs)

    import rfdetr

    cls = getattr(rfdetr, args.model)
    model = cls(pretrain_weights=args.base) if args.base else cls()
    names = _dataset_names(data_dir)

    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "dataset_file": "yolo",
        "epochs": max(1, int(args.epochs)),
        "batch_size": args.batch if args.batch == "auto" else max(1, int(args.batch)),
        "output_dir": str(output),
        "device": "cuda:0",
        "class_names": names,
        "num_workers": 2,
        "run_test": False,
        "tensorboard": False,
        "progress_bar": True,
    }
    if args.resolution:
        block = int(model.model_config.patch_size) * int(
            model.model_config.num_windows
        )
        resolution = (max(block, int(args.resolution)) // block) * block
        train_kwargs["resolution"] = resolution

    _write_status(status_path, 0, args.epochs)
    model.train(**train_kwargs)
    _write_status(status_path, args.epochs, args.epochs)

    best_source = _find_best(output)
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pth"
    if best_path.exists():
        best_path.unlink()
    try:
        os.link(best_source, best_path)
    except OSError:
        shutil.copy2(best_source, best_path)

    metrics: dict[str, Any] = {}
    try:
        evaluated = rfdetr.RFDETR.from_checkpoint(str(best_path), device="cuda:0")
        raw = evaluated.evaluate(
            dataset_dir=str(data_dir),
            split="val",
            batch_size=1,
            device="cuda:0",
        )
        map50 = _metric(raw, "mAP_50")
        map50_95 = _metric(raw, "mAP_50_95")
        if map50 is not None:
            metrics["map50"] = round(map50, 6)
            metrics["seen_map50"] = round(map50, 6)
        if map50_95 is not None:
            metrics["map50_95"] = round(map50_95, 6)
            metrics["seen_map50_95"] = round(map50_95, 6)
        metrics["eval_mode"] = "train20-seen"
    except Exception as exc:
        print(f"RF-DETR evaluate failed: {type(exc).__name__}: {exc}", flush=True)

    # Keep only the production checkpoint; the worker prunes older runs.
    _prune_checkpoints(output, best_path)

    payload = {
        "weights": str(best_path),
        "base_model": args.base or args.model,
        "metrics": metrics,
        "gpu": str(args.gpu),
    }
    Path(args.result).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
