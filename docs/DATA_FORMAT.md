# Data format

## 1. Pool folder

The console samples from a **pool root**. Every image file below it is a
candidate.

```text
data/pool/
├── labels.json                  # optional COCO ground truth
├── 13_203_0030/
│   └── e_cigarette-012/
│       ├── 20260918_164427_2654.jpg
│       └── ...
└── 177_202_0011/
    └── e_cigarette-011/
        └── ...
```

- Level 1 folder = `source` (camera/session). Sources get a fair quota.
- Level 2 folder = `clip`. Max 12 images per clip and min 10-frame gap when
  the diversity sampler is used.
- Any nesting deeper than 2 is allowed; images are found recursively.

Set the default pool in `.env`:

```text
TC_POOL_ROOT=./data/pool
```

## 2. COCO labels (optional)

If the pool already has ground truth, add a COCO 1.0 file named
`labels.json` or `labels_instance_split.json` in the pool root, or configure a
custom path per project in `trainconsole/projects.py`.

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "13_203_0030/e_cigarette-012/20260918_164427_2654.jpg",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 8,
      "bbox": [354.29, 374.89, 230.86, 83.43],
      "area": 19260.1,
      "iscrowd": 0
    }
  ],
  "categories": [
    { "id": 8, "name": "e_cigarette" }
  ]
}
```

Rules:

| Field | Rule |
|---|---|
| `file_name` | Path **relative to the pool root**, exact case |
| `bbox` | `[x, y, width, height]`, **absolute pixels**, not normalized |
| `width`,`height` | Image size in pixels |
| `category.name` | Must match the CVAT label name, or add an alias |
| `iscrowd` | `0` (crowd regions are not imported) |
| Coordinates | No scaling, no resizing; out-of-range boxes are clamped |

The importer matches images **by `file_name`**, so annotation order does not
need to match the upload order. Frame numbers in CVAT follow the order in which
the console uploads the selected images.

## 3. New project with its own labels

Create the CVAT project with the desired labels (or let the console create it
on the first batch), then register it in `trainconsole/projects.py`:

```python
_EXTRA_PROJECTS["my_project"] = ProjectConfig(
    name="my_project",
    pool_root=ROOT / "data" / "my_project" / "images",
    labels=("ecigar",),
    coco=ROOT / "data" / "my_project" / "labels.json",
    label_aliases={"e_cigarette": "ecigar"},
    take_all=True,  # default the "Force all images" checkbox to on
)
```

`label_aliases` maps COCO category names to CVAT label names. CVAT label names
must already exist on the project; the console does not add labels to an
existing project.

## 4. Masks

YOLO training uses boxes only. Labels listed in `TC_EXCLUDE_FROM_DETECTION`
(default: `seatbelt`) are removed from the YOLO dataset but remain in CVAT.
Use SAM2 for those masks.

## 5. Validate before uploading

```bash
python - <<'PY'
import json
from pathlib import Path

pool = Path("data/pool")
data = json.loads((pool / "labels.json").read_text())
missing = [i["file_name"] for i in data["images"] if not (pool / i["file_name"]).is_file()]
print("images:", len(data["images"]), "annotations:", len(data["annotations"]))
print("missing files:", len(missing), missing[:5])
if missing:
    raise SystemExit(1)
print("categories:", [c["name"] for c in data["categories"]])
PY
```
