# Hướng dẫn triển khai Train Console

Package này **chỉ chứa training management**. CVAT đã có sẵn, không deploy lại.

## 0. Yêu cầu

- Một CVAT instance đang chạy (server A), có tài khoản admin.
- Biết 2 URL:
  - **CVAT API**: ví dụ `http://cvat-host:51102`
  - **CVAT UI**: URL user mở, ví dụ `http://cvat-host:8081`
- Máy chạy console (server B) truy cập được CVAT API.
- Conda/Minconda, `flock`.
- GPU NVIDIA nếu train trên máy này.
- Port console mặc định: **9080**.

> Không cần Docker cho console. Docker chỉ cần nếu bạn muốn dùng tính năng
> reset password qua `docker exec cvat_server` (khi console chạy cùng máy CVAT).

## 1. Đặt package lên server

```bash
cd ~/code
# copy folder handoff lên đây
cd handoff
```

## 2. Tạo `.env`

```bash
cp .env.example .env
nano .env
```

Điền:

```text
TC_CVAT_URL=http://<cvat-host>:51102
TC_CVAT_UI_URL=http://<cvat-host>:8081
TC_CVAT_USERNAME=admin
TC_CVAT_PASSWORD=<mật khẩu CVAT>
TC_GPU=3
TC_TRAIN_PYTHON=/home/<user>/anaconda3/envs/tanmai/bin/python
TC_POOL_ROOT=./data/pool
```

Ghi chú:

- `TC_CVAT_UI_URL` là URL **trình duyệt user mở được**, console dùng để tạo link
  job. Nếu user mở từ máy khác, dùng IP/DNS, không dùng `localhost`.
- `TC_GPU` là GPU cho YOLO.
- `TC_TRAIN_PYTHON` là python của env train (ultralytics + torch).

## 3. Bỏ data

```text
handoff/data/pool/
├── labels.json                   # optional COCO
└── <source>/<clip>/*.jpg
```

Format chi tiết: `docs/DATA_FORMAT.md`.
Muốn agent format hộ: copy prompt trong `docs/AGENT_PROMPT.md`.

Nếu muốn tách project riêng (pool/label khác), thêm `ProjectConfig` trong
`trainconsole/projects.py`.

## 4. Cài môi trường

### Console + worker

```bash
conda create -n trainconsole python=3.11 -y
conda activate trainconsole
pip install -r requirements-console.txt
```

### Train

```bash
conda create -n tanmai python=3.10 -y
conda activate tanmai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

Trỏ `TC_TRAIN_PYTHON` tới python của env `tanmai`.

### RF-DETR-2XL (tuỳ chọn)

```bash
conda create -n rfdetr python=3.10 -y
conda activate rfdetr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-console.txt
pip install ultralytics "rfdetr[train,plus]"
```

Trỏ `TC_RFDETR_PYTHON` và `TC_WORKER_PYTHON` tới python của env `rfdetr`.
`TC_WORKER_PYTHON` cần có cả `ultralytics` + `rfdetr` để inference RF-DETR.
Lưu ý: RF-DETR-2XL dùng license Roboflow PML-1.0; self-host commercial cần
Enterprise add-on.

## 5. Chạy

```bash
conda activate trainconsole
./run_console.sh
```

Cửa sổ khác:

```bash
TC_WORKER_PYTHON=$(which python) ./deploy/tc_worker.sh
```

Nên thêm cron để worker tự restart:

```cron
@reboot /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
* * * * * /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
```

## 6. Sử dụng

Mở:

```text
http://<server-B>:9080
```

Login: `admin` / `TC_ADMIN_PASSWORD` (mặc định `admin`).

1. **Management** → chọn project CVAT.
2. **+ New batch** → pool, số ảnh, job size, assign user, import COCO.
3. **Training** → chọn model **YOLO26x** hoặc **RF-DETR-2XL**; **Test** hiển
   thị batch YOLO và batch RF-DETR đã calibrate theo VRAM trống + lần train
   gần nhất; nhập batch đó vào ô Batch. **Compare 50** cho
   chọn **View last compare** (kết quả cũ) hoặc **Run new 50 images**; dùng
   GPU/batch trên form, không ghi CVAT.
3. **Preview sample** → **Create CVAT task/jobs** (đẩy sang CVAT đã có).
4. User label/review trong CVAT.
5. **Training** → **Train** (chỉ train); **Infer** là nút riêng.
6. Theo dõi tiến độ từng epoch, **Cancel** nếu cần.
7. **Export labels** để tải COCO JSON.
8. Quản lý user trong **Manage users** (tạo/xoá, xem password).

## 7. Kiểm tra nhanh

```bash
curl -s http://localhost:9080/health
pgrep -af 'trainconsole.worker'
tail -f var/trainconsole/train_runs/<run_id>/train.log
nvidia-smi
```

## 8. Đổi port console

Mặc định `9080`.

1. `.env`:
   ```text
   TC_CONSOLE_PORT=9090
   ```
2. Chạy lại `./run_console.sh`.

`TC_CVAT_UI_URL` không đổi (đó là CVAT có sẵn).

## 9. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| Console không login được | Kiểm tra `TC_CVAT_URL`, `TC_CVAT_USERNAME/PASSWORD` |
| Train `queued` mãi | Worker chưa chạy: `pgrep -af trainconsole.worker` |
| `no completed CVAT jobs` | Chưa có job nào `completed` |
| `no images left in the pool` | Ảnh đã upload hết, hoặc task CVAT đã bị xoá (preview lại để tự dọn) |
| Job links mở sai | `TC_CVAT_UI_URL` phải là URL user truy cập được |
| Reset password lỗi | Console phải chạy trên máy có Docker CVAT, hoặc dùng admin UI của CVAT |
| Port bận | Đổi `TC_CONSOLE_PORT` |

## 10. File nặng

Không cần upload. Xem `WEIGHTS.md`: YOLO26x tự tải, checkpoint train sinh ra
trong `var/trainconsole/`. `assets/` chỉ dùng khi offline.
