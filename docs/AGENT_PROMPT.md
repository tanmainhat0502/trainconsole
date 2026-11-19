# Agent prompt — prepare data for this repository

Copy the block below into your coding agent (Claude Code, OpenCode, Cursor,
etc.), fill in the paths, and let it prepare the dataset.

---

```text
You are preparing a dataset for the "CVAT + Train Console" pipeline in this
repository.

INPUT: <absolute path of the raw dataset>
TARGET POOL ROOT: <repo>/data/pool
TARGET CLASSES: <e.g. ecigar>
CVAT PROJECT NAME: <e.g. my_project>

Goal: produce a pool folder and a COCO 1.0 label file that this repository can
upload to CVAT and train on.

Requirements:

1. Images
   - Copy or symlink every image into TARGET POOL ROOT, preserving source and
     clip folders:
       <pool>/<source>/<clip>/<image file>
   - source = camera/session, clip = recording/segment.
   - Never modify or resize the source images.
   - Allowed extensions: .jpg .jpeg .png .bmp .webp
   - Do not copy images that are not referenced by the annotations unless the
     user explicitly wants unlabeled images in the pool.

2. Labels (COCO 1.0)
   - Write TARGET POOL/labels.json.
   - "images": one entry per image with id, file_name, width, height.
   - "file_name" MUST be relative to TARGET POOL ROOT and use forward slashes,
     exactly matching the path where the image was copied.
   - "annotations": one entry per object:
       - bbox = [x, y, width, height] in ABSOLUTE PIXELS (COCO xywh).
       - Do NOT normalize, scale, or resize the boxes.
       - area = width * height.
       - iscrowd = 0.
   - "categories": id and name. Names must exactly match TARGET CLASSES, unless
     a different source name is required; in that case also update
     label_aliases in trainconsole/projects.py.
   - Coordinate origin is top-left. Clamp boxes to the image bounds if needed.

3. Project registration
   - Add an entry to _EXTRA_PROJECTS in trainconsole/projects.py:
       _EXTRA_PROJECTS["<CVAT PROJECT NAME>"] = ProjectConfig(
           name="<CVAT PROJECT NAME>",
           pool_root=ROOT / "data" / "pool",
           labels=("<class 1>", "<class 2>"),
           coco=ROOT / "data" / "pool" / "labels.json",
           label_aliases={"<source name>": "<cvat label>"},
       )
   - If the CVAT project does not exist yet, the console creates it on the
     first batch with the labels above.

4. Validation (run it and paste the output)
   - Count images, annotations and categories.
   - Verify every file_name exists under the pool root.
   - Verify every category name is in the CVAT label list.
   - Verify every bbox is inside its image and has width/height > 0.
   - Check there are no duplicate image ids and no duplicate file names.

5. Report
   - Write docs/DATA_REPORT.md with:
       total images, total boxes, classes and counts, source/clip counts,
       images without annotations, skipped/duplicate files, and any assumptions.

Do not touch assets/, .env, var/, or any weights. Do not commit anything.
```

---

## Adding a project without an agent

1. Copy images to `data/<name>/images/<source>/<clip>/...`.
2. Put the COCO file next to it (`data/<name>/labels.json`) or inside the pool
   root as `labels.json`.
3. Add the `ProjectConfig` block in `trainconsole/projects.py`.
4. Restart the Train Console (`./run_console.sh`) so the new config is loaded.
5. In Management, pick the project, create a batch with **Import existing COCO
   labels** checked.

## Small sanity check

```bash
python - <<'PY'
import json, sys
from pathlib import Path

pool = Path("data/pool")
data = json.loads((pool / "labels.json").read_text())
assert len({i["id"] for i in data["images"]}) == len(data["images"])
assert len({i["file_name"] for i in data["images"]}) == len(data["images"])
for img in data["images"]:
    assert (pool / img["file_name"]).is_file(), img["file_name"]
print("OK", len(data["images"]), "images", len(data["annotations"]), "boxes")
PY
```
