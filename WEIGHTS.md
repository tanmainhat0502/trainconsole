# Weights

Everything needed is downloaded from public sources. No upload is required.

## YOLO26x base weights

| Item | Source | Where to set |
|---|---|---|
| `yolo26x.pt` | `https://github.com/ultralytics/assets/releases` (Ultralytics downloads it automatically) | `TC_MODEL` in `.env` |

```python
from ultralytics import YOLO
YOLO("yolo26x.pt")   # downloads if missing
```

To use a local file:

```text
TC_MODEL=/absolute/path/to/yolo26x.pt
```

## Trained checkpoints

Training writes checkpoints under:

```text
var/trainconsole/train_runs/<run_id>/train/weights/best.pt
```

The console records them in its SQLite database and automatically uses the
newest one for the next training/inference. No path needs to be configured.

To move existing runs to another machine, copy the whole `var/` folder together
with `var/trainconsole.db`, or retrain.

## Optional offline copies

`assets/` is gitignored and only for offline machines:

```text
assets/
└── weights/
    └── yolo26x.pt
```

CVAT/SAM2 are not part of this package — use the existing CVAT deployment.
