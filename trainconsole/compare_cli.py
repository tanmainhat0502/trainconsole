"""Compare YOLO26 and RF-DETR on random pool images (read-only).

Used by the Compare modal. Supports multiple GPUs and a per-GPU batch.
No CVAT writes; the pool is only read.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_COLOR = (0, 255, 128)
YOLO_COLOR = (0, 220, 255)
DETR_COLOR = (255, 70, 70)
OUTLINE = (0, 0, 0)


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def _nms(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    iou_threshold: float,
    containment_threshold: float,
) -> list[list[float]]:
    ordered = sorted(
        zip(boxes, scores, labels),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    kept: list[tuple[list[float], float, int]] = []
    for box, score, label in ordered:
        duplicate = False
        for other_box, _other_score, other_label in kept:
            if int(label) != int(other_label):
                continue
            inter = _intersection(box, other_box)
            if inter <= 0:
                continue
            union = _area(box) + _area(other_box) - inter
            iou = inter / union if union > 0 else 0.0
            smaller = min(_area(box), _area(other_box))
            containment = inter / smaller if smaller > 0 else 0.0
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, score, label))
    return [item[0] for item in kept]


def _coco_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_name: dict[str, dict] = {}
    by_basename: dict[str, dict] = {}
    for image in payload.get("images", []):
        name = str(image.get("file_name") or "").replace("\\", "/").lstrip("./")
        by_name[name] = image
        by_basename.setdefault(Path(name).name, image)
    annotations: dict[int, list[dict]] = {}
    for annotation in payload.get("annotations", []):
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    return {"by_name": by_name, "by_basename": by_basename, "annotations": annotations}


def _gt_boxes(index: dict[str, Any], relative: str) -> list[list[float]]:
    image = index["by_name"].get(relative) or index["by_basename"].get(
        Path(relative).name
    )
    if image is None:
        return []
    width = float(image.get("width") or 0)
    height = float(image.get("height") or 0)
    boxes: list[list[float]] = []
    for annotation in index["annotations"].get(int(image["id"]), []):
        bbox = annotation.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, box_w, box_h = (float(value) for value in bbox)
        x1 = max(0.0, min(x, width)) if width else x
        y1 = max(0.0, min(y, height)) if height else y
        x2 = max(x1, min(x + box_w, width)) if width else x + box_w
        y2 = max(y1, min(y + box_h, height)) if height else y + box_h
        if x2 - x1 >= 0.5 and y2 - y1 >= 0.5:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _draw(
    image: Image.Image,
    gt: list[list[float]],
    yolo: list[list[float]],
    detr: list[list[float]],
) -> Image.Image:
    if image.width > 960:
        scale = 960 / image.width
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    else:
        scale = 1.0
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def boxes(items: list[list[float]], color: tuple[int, int, int]) -> None:
        for box in items:
            scaled = [value * scale for value in box]
            draw.rectangle(scaled, outline=OUTLINE, width=6)
            draw.rectangle(scaled, outline=color, width=4)

    boxes(gt, GT_COLOR)
    boxes(yolo, YOLO_COLOR)
    boxes(detr, DETR_COLOR)
    draw.text(
        (8, 8),
        "GT=green  YOLO=blue  DETR=red",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        font=font,
    )
    return image


def _result_boxes(result: Any, args: argparse.Namespace) -> list[list[float]]:
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    return _nms(
        [[float(value) for value in box] for box in boxes.xyxy.tolist()],
        [float(value) for value in boxes.conf.tolist()],
        [int(value) for value in boxes.cls.tolist()],
        float(args.nms_iou),
        float(args.nms_containment),
    )


def _detr_boxes(detections: Any, args: argparse.Namespace) -> list[list[float]]:
    return _nms(
        [[float(value) for value in box] for box in detections.xyxy.tolist()],
        [float(value) for value in detections.confidence.tolist()],
        [int(value) for value in detections.class_id.tolist()],
        float(args.nms_iou),
        float(args.nms_containment),
    )


def _process_chunk(
    chunk: list[tuple[int, Path]],
    gpu: int,
    args: argparse.Namespace,
    pool: Path,
    index: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    from ultralytics import YOLO

    yolo = YOLO(args.yolo)

    from rfdetr import RFDETR

    detr = RFDETR.from_checkpoint(
        args.detr, device=f"cuda:{gpu}", trust_checkpoint=True
    )
    output: list[tuple[int, dict[str, Any]]] = []
    batch_size = max(1, int(args.batch))
    for offset in range(0, len(chunk), batch_size):
        batch = chunk[offset : offset + batch_size]
        images = []
        for _original_index, path in batch:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        source = images if len(images) > 1 else images[0]
        yolo_results = yolo.predict(
            source=source,
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            device=gpu,
            verbose=False,
        )
        if not isinstance(yolo_results, list):
            yolo_results = [yolo_results]
        raw = detr.predict(source, threshold=float(args.conf))
        detr_results = raw if isinstance(raw, list) else [raw]
        for (original_index, path), image, yolo_result, detections in zip(
            batch, images, yolo_results, detr_results
        ):
            relative = path.relative_to(pool).as_posix()
            gt = _gt_boxes(index, relative)
            yolo_boxes = _result_boxes(yolo_result, args)
            detr_boxes = _detr_boxes(detections, args)
            rendered = _draw(image, gt, yolo_boxes, detr_boxes)
            buffer = io.BytesIO()
            rendered.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            output.append(
                (
                    original_index,
                    {
                        "name": relative,
                        "image": f"data:image/jpeg;base64,{encoded}",
                        "gt": len(gt),
                        "yolo": len(yolo_boxes),
                        "detr": len(detr_boxes),
                    },
                )
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare-cli")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--detr", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--nms-containment", type=float, default=0.85)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pool = Path(args.pool).resolve()
    images = [
        path
        for path in sorted(pool.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise SystemExit(f"no images in {pool}")
    count = max(1, min(int(args.count), len(images)))
    sampled = random.Random(args.seed).sample(images, count)
    index = _coco_index(Path(args.coco))
    gpus = [int(value) for value in args.gpus.replace(" ", ",").split(",") if value]
    if not gpus:
        gpus = [0]

    chunks = [
        [(position, path) for position, path in enumerate(sampled) if position % len(gpus) == slot]
        for slot in range(len(gpus))
    ]
    results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_process_chunk, chunk, gpu, args, pool, index)
            for gpu, chunk in zip(gpus, chunks)
            if chunk
        ]
        for future in futures:
            results.extend(future.result())

    results.sort(key=lambda item: item[0])
    payload = {"images": [item[1] for item in results], "count": len(results)}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"compared {len(results)} images on GPUs {gpus}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
