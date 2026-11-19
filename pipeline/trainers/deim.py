from __future__ import annotations

import json
from pathlib import Path

from .base import PreparedTrainer, TrainerProfile


class DeimAdapter:
    """Prepare a DEIM/KL730 training run from a frozen COCO snapshot.

    The generated config lives inside the trainer repo so its relative
    ``__include__`` paths keep working. The labeltool environment never imports
    torch or DEIM.
    """

    def prepare(
        self,
        *,
        profile: TrainerProfile,
        snapshot_dir: Path,
        run_dir: Path,
        options: dict,
    ) -> PreparedTrainer:
        repo = Path(profile.repo).resolve()
        template_path = Path(profile.config_template or "").resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"trainer template not found: {template_path}")
        snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        image_root = snapshot["image_root"]
        labels = snapshot["labels"]
        train_ann = snapshot_dir / "train" / "annotations.json"
        val_ann = snapshot_dir / "val" / "annotations.json"
        if not train_ann.exists() or not val_ann.exists():
            raise FileNotFoundError("snapshot is missing train/val COCO files")

        config_dir = repo / "configs" / "auto_label_train"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"run_{run_dir.name}.yml"
        template = template_path.read_text(encoding="utf-8")
        values = {
            "__NUM_CLASSES__": str(len(labels)),
            "__EPOCHES__": str(int(options.get("epochs", 20))),
            "__BATCH_SIZE__": str(int(options.get("batch_size", 8))),
            "__WORKERS__": str(int(options.get("workers", 4))),
            "__IMAGE_ROOT__": image_root,
            "__TRAIN_ANN__": str(train_ann),
            "__VAL_ANN__": str(val_ann),
        }
        for key, value in values.items():
            template = template.replace(key, value)
        config_path.write_text(template, encoding="utf-8")

        command = [
            profile.python,
            "train.py",
            "-c",
            str(config_path.relative_to(repo)),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(int(options.get("seed", 0))),
        ]
        if options.get("use_amp", True):
            command.append("--use-amp")
        if options.get("tuning_checkpoint"):
            command.extend(["--tuning", str(options["tuning_checkpoint"])])
        env = dict(profile.env)
        if options.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(options["gpu"])
        return PreparedTrainer(
            command=command,
            workdir=repo,
            config_path=config_path,
            env=env,
            profile=profile,
            metadata={
                "dataset_format": "coco",
                "image_root": image_root,
                "num_classes": len(labels),
                "labels": labels,
            },
        )


class GenericSubprocessAdapter:
    """YAML-driven adapter for any repo with a stable CLI.

    Command placeholders: {python}, {config}, {run_dir}, {snapshot},
    {image_root}, {train_ann}, {val_ann}, {seed}, {num_classes}.
    """

    def prepare(
        self,
        *,
        profile: TrainerProfile,
        snapshot_dir: Path,
        run_dir: Path,
        options: dict,
    ) -> PreparedTrainer:
        snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        repo = Path(profile.repo).resolve()
        config_path = snapshot_dir / "dataset.yaml"
        values = {
            "python": profile.python,
            "config": str(config_path),
            "run_dir": str(run_dir),
            "snapshot": str(snapshot_dir),
            "image_root": snapshot["image_root"],
            "train_ann": str(snapshot_dir / "train" / "annotations.json"),
            "val_ann": str(snapshot_dir / "val" / "annotations.json"),
            "seed": str(int(options.get("seed", 0))),
            "num_classes": str(len(snapshot["labels"])),
        }
        if not profile.command:
            raise ValueError(f"trainer profile {profile.id!r} has no command template")
        command = [str(part).format(**values) for part in profile.command]
        env = dict(profile.env)
        if options.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(options["gpu"])
        return PreparedTrainer(
            command=command,
            workdir=repo,
            config_path=config_path,
            env=env,
            profile=profile,
            metadata={"dataset_format": "coco", **values},
        )


def adapter_for(profile: TrainerProfile):
    if profile.id.startswith("deim"):
        return DeimAdapter()
    return GenericSubprocessAdapter()
