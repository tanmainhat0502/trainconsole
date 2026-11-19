from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cvat import client, state


def export_coco(project_id: int, scope: str = "completed") -> dict[str, Any]:
    """Merge CVAT's per-job COCO exports into a single COCO 1.0 document.

    ``scope`` is either ``completed`` (only finished jobs) or ``all`` (every job
    in the project). Images and annotations are re-numbered so several jobs can
    live in one JSON file.
    """

    scope = "all" if scope == "all" else "completed"
    merged: dict[str, Any] = {
        "info": {
            "description": f"Train Console label export ({scope})",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    category_ids: dict[str, int] = {}
    image_id = 0
    annotation_id = 0
    job_count = 0

    with client() as c:
        for task in c.tasks.list(project_id=project_id):
            for job in c.jobs.list(task_id=int(task.id)):
                if scope == "completed" and state(job) != "completed":
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    archive = Path(tmp) / "coco.zip"
                    job.export_dataset("COCO 1.0", str(archive), include_images=False)
                    with zipfile.ZipFile(archive) as zf:
                        name = next(
                            item for item in zf.namelist() if item.endswith(".json")
                        )
                        payload = json.loads(zf.read(name))
                job_count += 1

                local_images: dict[int, int] = {}
                local_categories = {
                    int(category["id"]): category
                    for category in payload.get("categories", [])
                }
                for image in payload.get("images", []):
                    image_id += 1
                    local_images[int(image["id"])] = image_id
                    file_name = image.get("file_name") or f"frame_{image_id}.jpg"
                    merged["images"].append(
                        {
                            **image,
                            "id": image_id,
                            "file_name": f"task_{task.id}/job_{job.id}/{file_name}",
                        }
                    )
                for annotation in payload.get("annotations", []):
                    local_category = local_categories.get(int(annotation["category_id"]))
                    if local_category is None:
                        continue
                    category_name = str(local_category.get("name") or "unknown")
                    category_id = category_ids.get(category_name)
                    if category_id is None:
                        category_id = len(category_ids) + 1
                        category_ids[category_name] = category_id
                        merged["categories"].append(
                            {
                                "id": category_id,
                                "name": category_name,
                                "supercategory": local_category.get("supercategory", ""),
                            }
                        )
                    merged_image_id = local_images.get(int(annotation["image_id"]))
                    if merged_image_id is None:
                        continue
                    annotation_id += 1
                    merged["annotations"].append(
                        {
                            **annotation,
                            "id": annotation_id,
                            "image_id": merged_image_id,
                            "category_id": category_id,
                        }
                    )

    merged["info"]["jobs"] = job_count
    return merged
