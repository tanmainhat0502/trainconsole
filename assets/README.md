# Optional offline assets (gitignored)

You do **not** need to put anything here for a normal install. YOLO26x is
downloaded by Ultralytics on the first training run:

```text
https://github.com/ultralytics/assets/releases
```

Use this folder only if the target machine has no internet:

```text
assets/
└── weights/
    └── yolo26x.pt
```

Then set:

```text
TC_MODEL=/absolute/path/to/assets/weights/yolo26x.pt
```

Trained checkpoints are generated automatically under
`var/trainconsole/train_runs/`.

CVAT and SAM2 are **not** part of this package; use your existing CVAT.
Nothing in this folder is committed to git.
