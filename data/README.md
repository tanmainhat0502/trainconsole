# Data folder

Put the raw image pool and its labels here.

Recommended layout (matches `TC_POOL_ROOT=./data/pool`):

```text
data/
└── pool/
    ├── labels.json                 # optional COCO 1.0 ground truth
    ├── <source>/<clip>/*.jpg       # e.g. 13_203_0030/e_cigarette-012/*.jpg
    ├── <source>/<clip>/*.jpg
    └── ...
```

Rules:

- Every image **under the pool root** is a candidate sample.
- The first folder level becomes the "source" (camera/session).
- The second level becomes the "clip" (used for gap/cap rules).
- `labels.json` (or `labels_instance_split.json`) is optional and must use
  `file_name` paths **relative to the pool root**.
- If you keep the COCO file somewhere else, set it per project in
  `trainconsole/projects.py` (`ProjectConfig.coco`).

For a second project, use a separate folder and register it in
`trainconsole/projects.py` — see `docs/DATA_FORMAT.md` and
`docs/AGENT_PROMPT.md`.

> Do not put weights or Docker images here. Use `assets/` for those.
