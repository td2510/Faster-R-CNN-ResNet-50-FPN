# Track C — Faster R-CNN

Track C là thí nghiệm mở rộng của Đề tài 4: so sánh YOLO one-stage với Faster R-CNN two-stage. Các mục YOLO, phân tích ngưỡng và phân tích lỗi vẫn là phần bắt buộc do Track B phụ trách.

## Dữ liệu dùng chung với Track B

Không cần tải ZIP từ Google Drive. Cách ưu tiên là tải trực tiếp đúng Roboflow project/version mà Track A đã dùng:

1. Tạo API key riêng trong Roboflow.
2. Ghi duy nhất key đó vào `incoming/roboflow_api_key.txt` (file này đã được git-ignore).
3. Chạy:

```powershell
.\.venv-track-c\Scripts\python.exe track_c_faster_rcnn.py roboflow
.\.venv-track-c\Scripts\python.exe track_c_faster_rcnn.py subset
.\.venv-track-c\Scripts\python.exe track_c_faster_rcnn.py audit
```

`subset` dùng seed 42 để chọn cố định 3.000 ảnh train từ danh sách đã sắp xếp, đồng thời giữ nguyên validation/test. Kết quả được ghi vào `data/BTL_DeTai4_9000/runs_rcnn/subset_manifest.csv` và `subset_report.json`. Track B phải dùng chính manifest/subset này nếu muốn so sánh mAP có kiểm soát; không nên so YOLO train trên 9.000 ảnh với Faster R-CNN train trên 3.000 ảnh như hai baseline tương đương.

Nếu tổng số ảnh sau khi giữ validation/test vượt giới hạn 5.000 của đề bài, pipeline sẽ dừng thay vì âm thầm vi phạm. Khi đó giảm riêng số ảnh train dựa trên số lượng audit thực tế.

ZIP từ Drive vẫn là phương án dự phòng: đặt ZIP vào `incoming/`, rồi chạy `extract`, `subset`, `audit`.

## Môi trường

Môi trường `.venv-track-c` đã được kiểm tra trên RTX 4060 Laptop với PyTorch CUDA:

```powershell
python -m venv .venv-track-c
.\.venv-track-c\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
.\.venv-track-c\Scripts\python.exe -m pip install -r requirements-track-c.txt
```

## Train, resume và giám sát

Chạy nền toàn bộ audit, smoke test, train 18 epoch, evaluate và benchmark:

```powershell
.\start_track_c.ps1
.\status_track_c.ps1
```

Runner thử lần lượt batch 4 FP32, batch 4 AMP, rồi batch 2 AMP với gradient accumulation 2. Checkpoint atomic được lưu sau mỗi epoch tại `data/BTL_DeTai4_9000/runs_rcnn/fasterrcnn_ckpt.pt`; chạy lại sẽ resume.

Trạng thái gần nhất nằm trong `data/BTL_DeTai4_9000/runs_rcnn/heartbeat.json`; stdout/stderr và PID cũng nằm trong cùng thư mục kết quả.

## Benchmark Track B

Đặt checkpoint YOLO vào `inputs/track_b/best.pt`, sau đó chạy:

```powershell
.\.venv-track-c\Scripts\python.exe -u track_c_faster_rcnn.py benchmark
```

Cả hai model được đánh giá lại trên RTX 4060, cùng validation và 100 ảnh test. Không so sánh trực tiếp thời gian train vì Track B dùng T4 còn Track C dùng RTX 4060.

## Run canonical Track A

Run dùng đúng subset 3.000 ảnh Track A nằm trong cấu hình `track_c_config_track_a.json`; train/val/test được dựng tại `data/BTL_DeTai4/dataset_track_a`. Junction `train` trỏ tới `data/BTL_DeTai4_9000/dataset/train_subset_3000`. Artifacts nằm trong `data/BTL_DeTai4_9000/runs_rcnn/track_a_subset` và đây là run Faster R-CNN canonical để đối chiếu Track B.

Kết quả tổng hợp, kiểm chứng checkpoint và bảng benchmark nằm tại:

- `data/BTL_DeTai4_9000/runs_rcnn/TRACK_C_RESULTS.md`
- `data/BTL_DeTai4_9000/runs_rcnn/comparison_fasterrcnn_subsets.csv`
- `data/BTL_DeTai4_9000/runs_rcnn/comparison_rtx4060.csv`
- `data/BTL_DeTai4_9000/runs_rcnn/checkpoint_verification.json`

Checkpoint YOLO subset đã có tại `data/BTL_DeTai4_9000/runs_yolo/baseline_640_subset3000/weights/best.pt`. Bảng chính thức trong `data/BTL_DeTai4_9000/runs_rcnn/track_a_subset/comparison_rtx4060.csv` đã được benchmark lại trên cùng RTX 4060, cùng validation và 100 ảnh test preload.
