from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

from PIL import Image

from ..config import settings
from ..cvat import client as cvat_client
from ..cvat import jobs_by_ids, list_jobs, state

Progress = Callable[[int, str], None]


def _area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _dedupe_boxes(
    boxes: list[dict[str, Any]],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[dict[str, Any]]:
    """Class-wise NMS for end-to-end detectors.

    YOLO26 predicts without NMS (``model.end2end``), so duplicate or nested
    boxes of the same class survive ``model.predict``. Keep the highest-score
    box and drop any later box that overlaps it heavily or is almost fully
    contained in it.
    """

    ordered = sorted(boxes, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if existing["label_id"] != candidate["label_id"]:
                continue
            inter = _intersection(candidate["points"], existing["points"])
            if inter <= 0:
                continue
            area_candidate = _area(candidate["points"])
            area_existing = _area(existing["points"])
            union = area_candidate + area_existing - inter
            iou = inter / union if union > 0 else 0.0
            smaller = min(area_candidate, area_existing)
            containment = inter / smaller if smaller > 0 else 0.0
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def eligible_jobs(project_id: int, task_ids: list[int] | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(project_id=project_id, task_ids=task_ids)
        if job["state"] == "new"
    ]


def inference_job_ids(
    project_id: int,
    task_ids: list[int] | None,
    job_ids: list[int] | None = None,
) -> list[int]:
    """Job IDs the current inference run would touch; used to split GPUs."""

    if job_ids:
        requested = jobs_by_ids(job_ids)
        wrong = [
            item["id"] for item in requested if item["project_id"] != int(project_id)
        ]
        if wrong:
            raise ValueError(
                f"job(s) {wrong} do not belong to project {project_id}"
            )
        return [int(item["id"]) for item in requested if item["state"] == "new"]
    return [int(job["id"]) for job in eligible_jobs(project_id, task_ids)]


def _class_names(model: Any, fallback: list[str]) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return list(fallback)


def run_inference(
    *,
    model_path: str,
    names: list[str],
    project_id: int,
    task_ids: list[int] | None,
    job_ids: list[int] | None = None,
    family: str = "yolo26",
    batch: int = 1,
    confidence: float = 0.25,
    imgsz: int = 640,
    gpu: str = "0",
    progress: Progress | None = None,
) -> dict[str, Any]:
    if family == "rfdetr":
        from rfdetr import RFDETR

        device = f"cuda:{gpu}" if str(gpu).isdigit() else "cuda:0"
        model = RFDETR.from_checkpoint(model_path, device=device)
        class_names = list(getattr(model, "class_names", None) or names)
    else:
        from ultralytics import YOLO

        model = YOLO(model_path)
        class_names = _class_names(model, names)
    skipped = 0
    if job_ids:
        requested = jobs_by_ids(job_ids)
        wrong_project = [
            item["id"] for item in requested if item["project_id"] != int(project_id)
        ]
        if wrong_project:
            raise ValueError(
                f"job(s) {wrong_project} do not belong to project {project_id}"
            )
        jobs = [item for item in requested if item["state"] == "new"]
        skipped = len(requested) - len(jobs)
    else:
        jobs = eligible_jobs(project_id, task_ids)
    counts = {
        "jobs": len(jobs),
        "processed": 0,
        "skipped": skipped,
        "images": 0,
        "boxes": 0,
        "model": model_path,
    }
    if job_ids:
        counts["requested_job_ids"] = [int(job_id) for job_id in job_ids]
    if progress:
        progress(5, f"{len(jobs)} job(s) new")

    with cvat_client() as c:
        for index, job_row in enumerate(jobs):
            job_id = int(job_row["id"])
            job = c.jobs.retrieve(job_id)
            if state(job) != "new":
                counts["skipped"] += 1
                continue
            task = c.tasks.retrieve(job.task_id)
            label_ids = {label.name: int(label.id) for label in task.get_labels()}
            shapes: list[dict[str, Any]] = []
            start, stop = int(job.start_frame), int(job.stop_frame)
            batch = max(1, int(batch))
            frames = list(range(start, stop + 1))
            for offset in range(0, len(frames), batch):
                batch_frames = frames[offset : offset + batch]
                images = []
                for frame in batch_frames:
                    stream = None
                    try:
                        stream = task.get_frame(frame)
                        images.append(
                            Image.open(
                                BytesIO(
                                    stream.read()
                                    if hasattr(stream, "read")
                                    else bytes(stream)
                                )
                            ).convert("RGB")
                        )
                    finally:
                        if stream is not None and hasattr(stream, "close"):
                            stream.close()
                if family == "rfdetr":
                    source = images if len(images) > 1 else images[0]
                    raw = model.predict(source, threshold=float(confidence))
                    detections_list = raw if isinstance(raw, list) else [raw]
                else:
                    source = images if len(images) > 1 else images[0]
                    results = model.predict(
                        source=source,
                        conf=float(confidence),
                        imgsz=int(imgsz),
                        iou=float(settings.nms_iou),
                        max_det=int(settings.max_detections),
                        device=gpu,
                        verbose=False,
                    )
                    detections_list = results if isinstance(results, list) else [results]
                for frame, detections in zip(batch_frames, detections_list):
                    if family == "rfdetr":
                        per_name = None
                        if getattr(detections, "data", None):
                            per_name = detections.data.get("class_name")
                        frame_boxes = []
                        for det_index in range(len(detections)):
                            class_id = int(detections.class_id[det_index])
                            label_name = (
                                str(per_name[det_index])
                                if per_name is not None
                                else (
                                    class_names[class_id]
                                    if class_id < len(class_names)
                                    else None
                                )
                            )
                            label_id = label_ids.get(label_name)
                            if label_id is None:
                                continue
                            x1, y1, x2, y2 = [
                                float(value)
                                for value in detections.xyxy[det_index]
                            ]
                            frame_boxes.append(
                                {
                                    "frame": frame,
                                    "label_id": label_id,
                                    "type": "rectangle",
                                    "points": [x1, y1, x2, y2],
                                    "score": float(detections.confidence[det_index]),
                                }
                            )
                    else:
                        boxes = getattr(detections, "boxes", None)
                        if boxes is None:
                            continue
                        frame_boxes = []
                        for box in boxes:
                            class_id = (
                                int(box.cls[0].item())
                                if hasattr(box.cls[0], "item")
                                else int(box.cls[0])
                            )
                            if class_id >= len(class_names):
                                continue
                            label_id = label_ids.get(class_names[class_id])
                            if label_id is None:
                                continue
                            x1, y1, x2, y2 = [
                                float(value) for value in box.xyxy[0].tolist()
                            ]
                            score = (
                                float(box.conf[0].item())
                                if hasattr(box.conf[0], "item")
                                else float(box.conf[0])
                            )
                            frame_boxes.append(
                                {
                                    "frame": frame,
                                    "label_id": label_id,
                                    "type": "rectangle",
                                    "points": [x1, y1, x2, y2],
                                    "score": score,
                                }
                            )
                    kept = _dedupe_boxes(
                        frame_boxes,
                        iou_threshold=float(settings.nms_iou),
                        containment_threshold=float(settings.nms_containment),
                    )
                    shapes.extend(kept)
                    counts["images"] += 1
                    counts["boxes"] += len(kept)

            # Re-check right before writing: the user may have started this job
            # while inference was running.
            job = c.jobs.retrieve(job_id)
            if state(job) != "new":
                counts["skipped"] += 1
                continue
            fresh = job.get_annotations()
            if not shapes:
                counts["skipped"] += 1
                continue
            from cvat_sdk import models
            from cvat_sdk.core.proxies.annotations import AnnotationUpdateAction

            # Replace old bbox (e.g. imported COCO labels) with fresh
            # predictions, but keep masks/polylines such as the seatbelt mask.
            stale = [
                shape
                for shape in (getattr(fresh, "shapes", None) or [])
                if str(getattr(shape, "type", "")).lower() in {"rectangle", "box"}
            ]
            if stale:
                job.update_annotations(
                    models.PatchedLabeledDataRequest(
                        shapes=[models.LabeledShapeRequest(**item.to_dict()) for item in stale]
                    ),
                    action=AnnotationUpdateAction.DELETE,
                )
                counts["replaced"] = counts.get("replaced", 0) + len(stale)
            job.update_annotations(
                models.PatchedLabeledDataRequest(
                    shapes=[models.LabeledShapeRequest(**shape) for shape in shapes]
                ),
                action=AnnotationUpdateAction.CREATE,
            )
            counts["processed"] += 1
            if progress:
                progress(
                    5 + int(90 * (index + 1) / max(1, len(jobs))),
                    f"job {job_id}: {len(shapes)} boxes",
                )
    return counts
