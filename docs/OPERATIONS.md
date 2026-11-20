# Operations

## Services

| Service | URL | Runs as |
|---|---|---|
| CVAT (existing) | `http://<cvat-host>:<cvat-ui-port>` | your existing CVAT |
| CVAT API | `http://<cvat-host>:<cvat-api-port>` | your existing CVAT |
| Train Console | `http://<host>:9080` | host process (`uvicorn`) |
| Worker | — | host process (`deploy/tc_worker.sh`) |

Train Console login: `admin` / `TC_ADMIN_PASSWORD` (default `admin`).

The console reads CVAT through `TC_CVAT_URL`; job links use
`TC_CVAT_UI_URL`.

## Management

- **Project** dropdown lists every CVAT project.
- **KPIs**: images in pool, on CVAT, labeled, unlabeled, not on CVAT.
- **+ New batch**:
  - pool root from the selected project config;
  - *Force all images* takes all remaining images, ordered round-robin by
    source, so each job covers many clips;
  - unchecked + `Images < pool size` runs the diversity sampler (KMeans medoid,
    10% audit, max 12/clip, min frame gap 10, near-duplicate removal);
  - *Import existing COCO labels* imports boxes by `file_name`, no scaling.
- **Assignee**: type the first letters to search a user; selection saves
  directly to CVAT.
- **Manage users**: create/delete CVAT accounts; click a row to see the stored
  password. Old accounts have no stored password; **Set password** runs
  `docker exec cvat_server python manage.py changepassword`, so it only works
  when the console runs on the CVAT Docker host.
- **User progress**: completed frames / total per assignee.

## Training

- **Model**: choose **YOLO26x** or **RF-DETR-2XL** in the train form (and in
  the Infer dialog). YOLO uses per-GPU batch; RF-DETR uses `batch_size="auto"`.
- **Train & Infer**: trains on all `completed` jobs of the project, then queues
  inference for `new` jobs with the new weights. Job exports are cached in
  `var/trainconsole/job_cache` and reused while the job's CVAT `updated_date`
  is unchanged, so a later train only exports new or re-annotated jobs.
  Delete that folder to force a full re-export (`TC_EXPORT_CACHE=0` disables
  reuse).
- After a **successful** run, only the newest checkpoint per **project + model
  family** is kept
  under `train_runs/*/train/weights`; older `best.pt` / `last.pt` files are
  deleted. Datasets and logs are kept. Set `TC_PRUNE_OLD_WEIGHTS=0` to keep
  every checkpoint. Old `ModelVersion` rows stay as `retired` for history, but
  their files are gone — enable the setting if you need rollback/promotion of
  old models. Training YOLO does not delete the newest RF-DETR checkpoint and
  vice versa, so Compare always has both.
- **Inference GPU/batch**: manual Infer uses the GPU list and per-GPU batch
  from the form; auto-infer inherits the training run's settings. With several
  GPUs, jobs are split across GPUs and each GPU uses the same batch.
- **RF-DETR-2XL**: training uses PyTorch Lightning DDP across the whole GPU
  list; the batch from the form is per GPU. Inference also uses the GPU list
  (jobs are sharded across GPUs). Set `TC_RFDETR_PYTHON` to its env
  (Python >= 3.10 with `ultralytics` and `rfdetr[train,plus]`). The worker interpreter
  (`TC_WORKER_PYTHON`) must also have `ultralytics` + `rfdetr` for RF-DETR
  inference. Training uses the same YOLO-format dataset (seatbelt excluded by
  `TC_EXCLUDE_FROM_DETECTION`).
- **Compare 50**: requires one checkpoint of each family. It reads 50 random
  images from the project pool, uses the GPU list and per-GPU batch from the
  form, draws GT (green), YOLO (blue) and DETR (red), and never writes to
  CVAT. The modal offers **View last compare** (cached result) or **Run new
  50 images** (fresh inference).
- **Infer**: inference only, newest weights of the selected family, no training.
- **Export labels ▾**: COCO JSON for `completed` jobs only or `all` jobs.
- Progress is per epoch (`epoch x/y`, 45% → 95% → 100%).
- **Cancel** stops the YOLO subprocess and frees the GPU.
- Weight policy:
  - first training: `yolo26x.pt`;
  - later: newest trained weights of the project (tuning continues).
- Evaluation: `20%` of the same training images (seen). `mAP50` measures
  memorization, not generalization.
- **GPU** — single id or comma-separated (e.g. `0,1,3`). Training uses
  Ultralytics DDP with 1 process per GPU. The batch you enter is **per GPU**
  (batch 14 on 3 GPUs → 14 on each GPU). The **Test** button next to the GPU
  field probes free VRAM for YOLO and calibrates the RF-DETR batch from the
  last RF-DETR run's memory and current free VRAM. The YOLO probe mirrors real
  training (AMP FP16) and targets ~90% of free VRAM. A shared GPU gets a lower
  suggestion. Set the Batch field to the Test suggestion before training. Inference uses the
  first GPU of the list.

## Inference

The **Infer** button opens a dialog with two modes:

- **All jobs in state new** — uses the Task IDs field above (empty = all tasks).
- **Specific job IDs** — comma-separated list, e.g. `82, 84, 90`.

Only jobs in state `new` are touched; targeted IDs in other states are counted
as skipped:

| Job state | Behaviour |
|---|---|
| `new` | boxes are replaced by fresh predictions; job stays `new` |
| `in progress` | skipped (manual work) |
| `completed` | skipped |
| has masks/polygons | boxes replaced, masks kept |

Before writing, the worker re-reads the job state and annotations. YOLO26 is
end-to-end (no built-in NMS), so the console applies its own class-wise NMS
plus a containment filter (`TC_NMS_IOU`, `TC_NMS_CONTAINMENT`, `TC_MAX_DET`).

## Worker supervision

`deploy/tc_worker.sh` keeps one worker alive:

- restarts it 5 s after a crash;
- cron `* * * * *` restarts the supervisor if needed;
- cron `@reboot` starts it after a reboot;
- on startup, a run left `running` by a dead worker is re-queued
  (`requeued after worker restart`); any orphaned training process tree from
  the previous worker is killed first;
- **Cancel** kills the whole DDP process tree (launcher + all ranks), not just
  the parent process.

```cron
@reboot /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
* * * * * /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
```

Set `TC_WORKER_PYTHON` to the interpreter that has the console dependencies.

## Useful checks

```bash
# API
curl -s http://localhost:9080/health

# worker
pgrep -af 'trainconsole.worker'

# training log
tail -f var/trainconsole/train_runs/<run_id>/train.log

# GPU
nvidia-smi
```
