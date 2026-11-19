# Handoff checklist — Train Console

## A. Gửi những gì

Gửi **toàn bộ folder `handoff/`**, trừ:

```text
.env
var/
data/        (dữ liệu thật)
assets/      (weights offline, nếu có)
```

Không cần gửi CVAT/Docker/SAM2 — CVAT đã có sẵn.

## B. Điều kiện trước khi bắt đầu

- CVAT đang chạy, có tài khoản admin.
- Biết:
  - **CVAT API URL**, ví dụ `http://cvat-host:51102`
  - **CVAT UI URL** mà user mở, ví dụ `http://cvat-host:8081`
- Máy chạy console truy cập được CVAT API.
- Conda, `flock`, GPU nếu train.

## C. Các bước làm

### 1. Nhận package

```bash
cd ~/code
# copy folder handoff vào đây
cd handoff
```

### 2. Tạo `.env`

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

- `TC_CVAT_UI_URL` phải là URL user mở được (không dùng `localhost` nếu user ở
  máy khác).
- Console chạy port **9080** (`TC_CONSOLE_PORT`).

### 3. Bỏ data

```text
data/pool/
├── labels.json                   # optional COCO
└── <source>/<clip>/*.jpg
```

Format: `docs/DATA_FORMAT.md`.
Agent format hộ: `docs/AGENT_PROMPT.md`.

### 4. Cài môi trường

```bash
conda create -n trainconsole python=3.11 -y
conda activate trainconsole
pip install -r requirements-console.txt

conda create -n tanmai python=3.10 -y
conda activate tanmai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

### 5. Chạy console + worker

```bash
conda activate trainconsole
./run_console.sh

# cửa sổ khác
TC_WORKER_PYTHON=$(which python) ./deploy/tc_worker.sh
```

Cron nên thêm:

```cron
@reboot /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
* * * * * /path/to/handoff/deploy/tc_worker.sh >/dev/null 2>&1
```

### 6. Sử dụng

```text
http://<server-B>:9080
```

Login: `admin` / `TC_ADMIN_PASSWORD`.

1. Management → chọn project.
2. **+ New batch** → pool, số ảnh, job size, assign user, import COCO.
3. Preview → Create CVAT task/jobs.
4. User label/review trong CVAT có sẵn.
5. Training → **Train & Infer**.
6. Theo dõi tiến độ, Cancel nếu cần.
7. Export labels nếu cần COCO.

## D. Kiểm tra nhanh

```bash
curl -s http://localhost:9080/health
pgrep -af 'trainconsole.worker'
tail -f var/trainconsole/train_runs/<run_id>/train.log
nvidia-smi
```

## E. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| Không login/không thấy project | Sai `TC_CVAT_URL` hoặc credentials |
| Train `queued` mãi | Worker chưa chạy |
| `no completed CVAT jobs` | Chưa có job `completed` |
| `no images left in the pool` | Ảnh đã upload hết / task đã xoá (preview lại) |
| Link job mở sai | `TC_CVAT_UI_URL` phải là URL user mở được |
| Reset password lỗi | Console phải chạy trên máy Docker của CVAT, hoặc dùng admin UI CVAT |
| Port bận | Đổi `TC_CONSOLE_PORT` |

## F. Tin nhắn ngắn để gửi kèm

```text
Mình gửi bạn package Train Console (CVAT đã có sẵn, không cần deploy lại).

Bạn mở README.md, sau đó làm theo docs/SETUP_GUIDE.md.
Tóm tắt:
1. Copy .env.example -> .env, điền:
   TC_CVAT_URL, TC_CVAT_UI_URL, TC_CVAT_PASSWORD, TC_GPU, TC_TRAIN_PYTHON.
2. Bỏ ảnh vào data/pool/<source>/<clip>/, labels.json nếu có.
3. Cài requirements-console.txt và requirements-train.txt.
4. Chạy ./run_console.sh và ./deploy/tc_worker.sh.
5. Mở http://<server>:9080 (admin/admin).

Không cần gửi weights; YOLO tự tải (WEIGHTS.md).
```
