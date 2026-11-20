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
import subprocess
import sys
import time
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


def _wait_stable(path: Path, timeout: float = 180.0) -> bool:
    """Wait until DDP rank 0 has finished writing a checkpoint file."""

    deadline = time.time() + timeout
    last_size = -1
    stable = 0
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            stable = 0
            time.sleep(1)
            continue
        if size == last_size and size > 0:
            stable += 1
            if stable >= 3:
                return True
        else:
            stable = 0
            last_size = size
        time.sleep(1)
    return path.is_file()


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
    parser.add_argument("--devices", default="0")
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

    devices = [value.strip() for value in str(args.devices).split(",") if value.strip()]
    distributed = len(devices) > 1
    model_device = "cpu" if distributed else "cuda:0"

    cls = getattr(rfdetr, args.model)
    if args.base:
        try:
            # Checkpoints written by our DDP runs contain custom objects and
            # must be loaded as trusted local training artifacts.
            model = rfdetr.RFDETR.from_checkpoint(
                args.base, trust_checkpoint=True, device=model_device
            )
        except Exception:
            model = cls(pretrain_weights=args.base, device=model_device)
    else:
        model = cls(device=model_device)
    names = _dataset_names(data_dir)
    batch_value: str | int
    if distributed:
        try:
            batch_value = max(1, int(args.batch))
        except (TypeError, ValueError):
            batch_value = 4
    else:
        batch_value = args.batch if args.batch == "auto" else max(1, int(args.batch))

    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "dataset_file": "yolo",
        "epochs": max(1, int(args.epochs)),
        "batch_size": batch_value,
        "output_dir": str(output),
        "class_names": names,
        "num_workers": 2,
        "run_test": False,
        "tensorboard": False,
        "progress_bar": True,
    }
    if distributed:
        # Lightning DDP over the visible devices (CUDA_VISIBLE_DEVICES is set
        # by train_rfdetr). The batch from the UI is per GPU.
        train_kwargs["devices"] = ",".join(devices)
        train_kwargs["strategy"] = "ddp"
        train_kwargs["grad_accum_steps"] = 1
    else:
        train_kwargs["device"] = "cuda:0"
    if args.resolution:
        block = int(model.model_config.patch_size) * int(
            model.model_config.num_windows
        )
        resolution = (max(block, int(args.resolution)) // block) * block
        train_kwargs["resolution"] = resolution

    _write_status(status_path, 0, args.epochs)
    model.train(**train_kwargs)
    _write_status(status_path, args.epochs, args.epochs)

    if distributed:
        # The parent process can return before rank 0 finishes its final
        # checkpoint callback. Wait for the best-total file first.
        deadline = time.time() + 180
        while time.time() < deadline:
            if (output / "checkpoint_best_total.pth").is_file():
                break
            time.sleep(1)
    best_source = _find_best(output)
    _wait_stable(best_source)
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
        if distributed:
            # Never evaluate from inside the DDP process group; use a clean
            # single-GPU process so collectives cannot target the wrong device.
            eval_script = Path(__file__).with_name("rfdetr_eval_cli.py")
            eval_out = run_dir / "eval_metrics.json"
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            eval_env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(eval_script),
                    "--checkpoint", str(best_path),
                    "--data", str(data_dir),
                    "--out", str(eval_out),
                ],
                env=eval_env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    (completed.stderr or completed.stdout or "").strip()[-2000:]
                )
            raw = json.loads(eval_out.read_text(encoding="utf-8"))
        else:
            evaluated = rfdetr.RFDETR.from_checkpoint(
                str(best_path), device="cuda:0", trust_checkpoint=True
            )
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
    # In DDP the rank-0 callback may still be finalizing checkpoint files, so
    # leave cleanup to the worker after the process exits.
    if not distributed:
        _prune_checkpoints(output, best_path)

    payload = {
        "weights": str(best_path),
        "base_model": args.base or args.model,
        "metrics": metrics,
        "gpu": str(args.devices),
    }
    Path(args.result).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
