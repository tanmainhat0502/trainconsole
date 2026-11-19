"""Quick VRAM probe: suggest a max per-GPU batch size for YOLO training.

Usage:
    python gpu_probe.py '{"weights": "path.pt", "imgsz": 640, "gpus": [0, 1, 3]}'

Prints a JSON object on the last stdout line:
    {"per_gpu": {0: 24, ...}, "per_image_bytes": .., "overhead_bytes": ..}

The probe must mirror real Ultralytics training: AMP (FP16), unfrozen params,
and peak allocator memory. A plain FP32 forward/backward overestimates
per-image cost by ~2x and under-suggests the batch (12 instead of ~18 on a
24 GB card). The fixed reserve covers optimizer/EMA state, CUDA context and
cuDNN workspace, calibrated against a real multi-GPU run.
"""

from __future__ import annotations

import json
import sys

import torch

TARGET_FRACTION = 0.90  # aim for ~90% of free VRAM, leaving room for DDP buffers
EXTRA_OVERHEAD = 1536 * 2**20  # CUDA context + cuDNN workspace (~1.5 GB)


def flatten(obj, acc: list) -> None:
    if isinstance(obj, torch.Tensor):
        acc.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            flatten(item, acc)
    elif isinstance(obj, dict):
        for item in obj.values():
            flatten(item, acc)


def main() -> int:
    cfg = json.loads(sys.argv[1])
    weights = cfg["weights"]
    imgsz = int(cfg["imgsz"])
    gpus = [int(g) for g in cfg["gpus"]]
    primary = gpus[0]

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA is not available"}))
        return 1
    available = torch.cuda.device_count()
    missing = [g for g in gpus if g >= available or g < 0]
    if missing:
        print(json.dumps({"error": f"GPU {','.join(str(g) for g in missing)} not found "
                                   f"(cuda devices: {available})"}))
        return 1

    free = {}
    for g in gpus:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(g)
        free[g] = int(free_bytes)

    # Probe on the least-loaded selected GPU so batch 8/16 fit.
    primary = max(gpus, key=lambda g: free[g])

    model = YOLO(weights)
    model.model.to(f"cuda:{primary}")
    model.model.train()
    # YOLO() loads inference-ready weights with frozen params; real training
    # allocates gradients, so enable them before probing.
    model.model.requires_grad_(True)
    params_bytes = sum(p.numel() * p.element_size() for p in model.model.parameters())
    # FP32 master weights + grads + SGD momentum + EMA copy + safety.
    reserve = params_bytes * 5 + EXTRA_OVERHEAD

    def probe(batch: int) -> int:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(primary)
        x = torch.zeros(batch, 3, imgsz, imgsz, device=f"cuda:{primary}")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.model(x)
        tensors: list = []
        flatten(out, tensors)
        floats = [t for t in tensors if t.dtype.is_floating_point]
        # Reduce before .float() so the loss does not create full-size fp32
        # copies of every detection head output (that was the old overestimate).
        loss = sum(t.sum().float() for t in floats) * 1024.0
        loss.backward()
        peak = torch.cuda.max_memory_allocated(primary)
        del x, out, tensors, floats, loss
        torch.cuda.empty_cache()
        return int(peak)

    try:
        p8 = probe(8)
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"error": "OOM even at batch 8; free some VRAM first"}))
        return 1

    a: float
    c: float
    try:
        p16 = probe(16)
        a = max((p16 - p8) / 8.0, 1.0)
        c = max(p8 - a * 8.0, 0.0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            p4 = probe(4)
            a = max((p8 - p4) / 4.0, 1.0)
            c = max(p8 - a * 8.0, 0.0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            a = max(p8 / 8.0, 1.0)
            c = 0.0
    torch.cuda.empty_cache()

    if a <= 0:
        print(json.dumps({"error": "probe failed to measure per-image cost"}))
        return 1

    per_gpu = {}
    for g in gpus:
        budget = free[g] * TARGET_FRACTION - c - reserve
        per_gpu[g] = max(0, int(budget / a) // 2 * 2)

    result = {
        "per_gpu": {str(g): per_gpu[g] for g in per_gpu},
        "suggest": min(per_gpu.values()) if per_gpu else 0,
        "free_bytes": {str(g): free[g] for g in free},
        "per_image_bytes": int(a),
        "overhead_bytes": int(c + reserve),
        "fraction": TARGET_FRACTION,
        "method": "amp-reserved-v2",
        "gpus": gpus,
        "weights": weights,
        "imgsz": imgsz,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
