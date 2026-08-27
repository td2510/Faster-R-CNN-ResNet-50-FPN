# Kế hoạch chạy trong 2 ngày — Đề tài 4 (YOLOv8 vs Faster R-CNN)

> Giả định: nhóm 3 người, mỗi ngày làm **08:00 → 18:00** (10 tiếng, có nghỉ trưa). Nếu nhóm chỉ có 2 người, gộp Track A vào Track B.
> So với bản 1 ngày, phiên bản này **không cắt phạm vi** nữa — dùng gần đúng số liệu đề bài gốc: ~3.000+ ảnh, YOLOv8 50 epoch, Faster R-CNN 15–20 epoch, và **train thật** ở cả 3 độ phân giải (416/640/960) thay vì chỉ đổi lúc inference.

---

## 0. Dataset đã chọn sẵn: Vietnam Traffic Sign Detection (Roboflow Universe)

Trong 3 hướng đề bài gợi ý (biển báo giao thông / mũ bảo hiểm / phương tiện giao thông đô thị), chọn **biển báo giao thông** vì có sẵn bộ dữ liệu tiếng Việt, đúng định dạng YOLO, số ảnh vừa khít khoảng khuyến nghị:

- **Tên dataset:** Vietnam-Traffic-Sign-Detection
- **Link:** https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8
- **Workspace slug:** `vietnam-traffic-sign-detection`
- **Project slug:** `vietnam-traffic-sign-detection-2i2j8`
- **Quy mô:** ~3.680 ảnh biển báo giao thông Việt Nam, đã chia sẵn train/valid/test
- **Lớp:** **đã xác nhận `nc: 58`** sau khi tải version 5 (đúng như dự đoán, khác với con số 59 hiển thị trên trang web — chênh lệch 1 lớp là do lý do đã nêu ở mục "Xác nhận" bên dưới). Danh sách đầy đủ 58 lớp (mã biển báo chuẩn Việt Nam):
  ```
  DP.135, P.102, P.103a, P.103b, P.103c, P.104, P.106a, P.106b, P.107a, P.112,
  P.115, P.117, P.123a, P.123b, P.124a, P.124b, P.124c, P.125, P.127, P.128,
  P.130, P.131a, P.137, P.245a, R.301c, R.301d, R.301e, R.302a, R.302b, R.303,
  R.407a, R.409, R.425, R.434, S.509a, W.201a, W.201b, W.202a, W.202b, W.203b,
  W.203c, W.205a, W.205b, W.205d, W.207a, W.207b, W.207c, W.208, W.209, W.210,
  W.219, W.221b, W.224, W.225, W.227, W.233, W.235, W.245a
  ```
- **Định dạng export:** hỗ trợ sẵn YOLOv8 (cũng có YOLOv5/v7/v9/v11) → không cần tự convert
- **Version dùng để tải:** **v5** (đã xác nhận trên trang — không phải v6 như suy đoán ban đầu từ kết quả tìm kiếm).
- **License:** Public Domain. Trích dẫn bắt buộc đưa vào README (mục "Tài liệu tham khảo" / khai báo nguồn dữ liệu):

```bibtex
@misc{ vietnam-traffic-sign-detection-2i2j8_dataset,
  title = { Vietnam-Traffic-Sign-Detection Dataset },
  type = { Open Source Dataset },
  author = { Vietnam traffic sign detection },
  howpublished = { \url{ https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8 } },
  url = { https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8 },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2026 },
  month = { aug },
  note = { visited on 2026-08-22 },
}
```

**Đã xác nhận (không cần làm lại):** tải version 5 về, mở `data.yaml` → `nc: 58` (khác 1 lớp so với con số 59 hiển thị trên trang "Classes", đúng như dự đoán — do trang Classes liệt kê theo lịch sử gán nhãn toàn dataset trong khi version cụ thể chỉ giữ một tập con). Toàn bộ code Faster R-CNN bên dưới đã dùng sẵn `NUM_CLASSES = 58`.

**Phương án dự phòng nếu dataset trên bị gỡ/đổi quyền truy cập:** vào https://universe.roboflow.com/search, gõ `vietnam traffic sign`, lọc format = YOLOv8, sắp xếp theo số ảnh, chọn dataset gần 3.000 ảnh nhất. Cách tải giữ nguyên như hướng dẫn bên dưới, chỉ đổi `workspace`/`project`.

**Không cần "fork" dataset về workspace Roboflow riêng của bạn.** Fork trên Roboflow là clone dataset sang tài khoản của bạn để tự sửa/gán nhãn lại — không cần thiết ở đây vì bạn chỉ dùng nguyên trạng. Chỉ cần API key của chính bạn (tài khoản Roboflow miễn phí bất kỳ) gọi thẳng vào `workspace`/`project` gốc như code bên dưới là tải được, vì đây là dataset public trên Universe. Cũng không cần tải về máy cá nhân rồi upload thủ công lên Drive — lệnh `download()` chạy ngay trong Colab tải thẳng vào ổ đĩa phiên Colab, sau đó lệnh `cp -r` ở Bước 3 copy thẳng từ đó sang Google Drive đã mount — toàn bộ nằm trong 1 notebook, không cần thao tác tay ở máy tính cá nhân.

---

## Phân vai (3 track chạy song song khi có thể)

| Track | Người | Việc chính |
|---|---|---|
| **A — Dữ liệu & phân tích lỗi** | Người 1 | Tải/thống kê dữ liệu, viết script phân tích lỗi, tổng hợp cuối |
| **B — YOLOv8** | Người 2 | Fine-tune YOLOv8n ở 3 độ phân giải, threshold sweep, inference |
| **C — Faster R-CNN** | Người 3 | Fine-tune Faster R-CNN (torchvision) với checkpoint/resume, đo mAP/FPS |

**Nguyên tắc bắt buộc:** B và C dùng **cùng bộ ảnh test** và **cùng loại GPU (Colab T4)** để so sánh FPS/mAP có ý nghĩa. Track A tải dữ liệu **một lần**, up lên **1 folder Drive dùng chung**, B/C mount chung folder đó.

---

## Ghi log thí nghiệm tự động (dùng chung cho Track B & C)

Đề bài bắt buộc có **nhật ký thí nghiệm ghi lại mọi lượt chạy** (ngày giờ, cấu hình, seed, kết quả). Thay vì chép tay vào Excel sau mỗi lần train/val (dễ quên, dễ sai), Track B và Track C **cùng dùng 1 hàm ghi log**, ghi vào **chung 1 file `experiment_log.csv`** trên Drive — cuối cùng có 1 bảng log duy nhất cho cả 2 model, không phải 2 file rời rạc.

**Copy y nguyên đoạn này vào notebook của cả Track B lẫn Track C**, đặt thành 1 cell riêng ngay sau `drive.mount(...)` — chạy 1 lần đầu notebook, trước khi train:

```python
import csv, os, datetime

log_path = "/content/drive/MyDrive/BTL_DeTai4/experiment_log.csv"
FIELDS = ["timestamp", "model", "run_name", "imgsz", "epochs", "batch",
          "seed", "conf", "iou", "precision", "recall", "map50", "map50_95",
          "fps", "num_params", "train_time_min", "notes"]

def log_experiment(row: dict):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
```

**Lưu ý:**
- Cell này **chỉ định nghĩa hàm**, chưa ghi gì cả — phải **gọi `log_experiment({...})`** sau mỗi lần `train()`/`val()`/đo FPS thì mới thực sự thêm 1 dòng vào CSV. Các chỗ cần gọi đã đánh dấu **📝 Ghi log** ở từng bước bên dưới.
- Nếu Colab bị ngắt/restart, phải chạy lại cell định nghĩa này trước khi gọi `log_experiment` lần nữa (file CSV trên Drive không mất, chỉ mất hàm trong RAM).
- Track B và Track C chạy song song vẫn an toàn khi cùng ghi vào 1 file — mỗi lần gọi chỉ append đúng 1 dòng rồi đóng file ngay, khả năng ghi đè gần như không xảy ra.

---

## Lưu ý khi Track B và Track C chạy độc lập, song song

1. **Dùng 2 tài khoản Google khác nhau cho 2 người** (không phải 2 tab của cùng 1 tài khoản) — Colab free tính quota GPU theo tài khoản, nếu B và C cùng dùng 1 tài khoản sẽ tranh nhau giới hạn sử dụng, dễ bị "usage limit reached" giữa chừng.
2. **Không sửa/xóa/thêm ảnh vào folder `dataset/` trên Drive** trong lúc cả 2 track đang chạy — cả hai đều gọi `sorted(glob.glob(f"{base}/test/images/*"))[:100]` để lấy đúng 100 ảnh test giống nhau; nếu thư mục bị thay đổi giữa lúc B và C chạy, thứ tự `sorted()` có thể lệch, khiến so sánh FPS không còn công bằng (không cùng 1 tập ảnh).
3. **Đợi cả 2 track ghi log xong rồi mới đọc `experiment_log.csv` để tổng hợp** (bước 14:00–15:30 Ngày 2) — không đọc file này trong lúc 1 trong 2 track còn đang chạy vòng lặp ghi log (sweep threshold), vì có thể đọc trúng lúc file đang được mở để append, dễ gây lỗi parse dòng dở dang.
4. **Cả 2 phải log cùng schema** (dùng đúng 1 hàm `log_experiment`/`FIELDS` đã cho, copy y nguyên, không tự sửa tên cột) — nếu 1 bên đổi tên cột, khi gộp bảng ở bước tổng hợp (`pd.read_csv` + `groupby`) sẽ bị lệch cột hoặc NaN.
5. **Không đổi tên hạng mục lớp (`class names`) khác đi giữa 2 model** — cả YOLOv8 (đọc trực tiếp từ `data.yaml`) và Faster R-CNN (tự map `label + 1`) phải cùng quy ước đánh số lớp 0-based → nếu lệch, so sánh per-class giữa 2 model (nếu làm) sẽ sai mà không báo lỗi.

---

# NGÀY 1

## 08:00–08:30 — Kickoff (cả nhóm)

1. Tạo GitHub repo trống, mọi người có quyền push.
2. Xác nhận dataset ở mục 0 (mở link, kiểm tra version/license/số lớp).
3. Người 1 (Track A) đăng ký Roboflow miễn phí, lấy API key: `Settings → Roboflow API → Private API Key`.
4. Chốt vai trò B/C.

## 08:30–09:15 — Track A: tải dữ liệu, đưa lên Drive dùng chung

Notebook Colab riêng, không cần GPU (Runtime = CPU cho khởi động nhanh).

**Bước 1 — Tải dataset đã chọn:**
```python
!pip install roboflow -q

from roboflow import Roboflow
rf = Roboflow(api_key="DÁN_API_KEY_CỦA_BẠN")
project = rf.workspace("vietnam-traffic-sign-detection").project("vietnam-traffic-sign-detection-2i2j8")
dataset = project.version(5).download("yolov8")
print(dataset.location)
```

**Bước 2 — Kiểm tra data.yaml (số lớp, tên lớp):**
```python
!cat {dataset.location}/data.yaml
```
Ghi lại giá trị `nc:` (số lớp) — dùng cho Track C ở bước sau.

**Bước 3 — Đưa lên Drive dùng chung:**
```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/BTL_DeTai4/dataset"
!cp -r {dataset.location}/* "/content/drive/MyDrive/BTL_DeTai4/dataset/"
```
→ Share folder `BTL_DeTai4` cho 2 bạn còn lại (Share → Anyone with link → Editor, hoặc thêm email).

**Bước 4 — Thống kê phân bố lớp & kích thước box:**
```python
import os, glob
from collections import Counter

label_dir = f"{dataset.location}/train/labels"
class_count = Counter()
size_count = Counter()

for f in glob.glob(f"{label_dir}/*.txt"):
    with open(f) as fh:
        for line in fh:
            cls, xc, yc, w, h = map(float, line.split())
            class_count[int(cls)] += 1
            area = w * h
            if area < (32/640)**2:
                size_count['small'] += 1
            elif area < (96/640)**2:
                size_count['medium'] += 1
            else:
                size_count['large'] += 1

print(class_count)
print(size_count)
```
Vẽ 2 biểu đồ cột (matplotlib) từ 2 Counter trên, lưu vào Drive.

**09:15** → Báo Track B, C: dữ liệu sẵn sàng, gửi số lớp (`nc`) đã đọc được.

---

## 09:15 → hết ngày 1 — Track C: Faster R-CNN chạy nền xuyên suốt

Faster R-CNN là track tốn thời gian máy nhất (ước 5–9 giờ cho 15–20 epoch với ~3.000+ ảnh), nên **bắt đầu ngay và để chạy nền** — không cần canh liên tục, chỉ cần không đóng hẳn tab trình duyệt.

**Bước 1 — Runtime GPU T4**, mount Drive, cài thư viện:
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install torchmetrics -q
import torch, torchvision
print(torch.__version__, torchvision.__version__)
```
→ Ngay sau cell này, dán cell định nghĩa `log_experiment` (xem mục "Ghi log thí nghiệm tự động" ở trên) — chạy 1 lần.

**Bước 2 — Dataset đọc nhãn YOLO (chuyển sang box pixel tuyệt đối cho torchvision):**
```python
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.transforms = transforms or T.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    boxes.append([
                        (xc - bw / 2) * w, (yc - bh / 2) * h,
                        (xc + bw / 2) * w, (yc + bh / 2) * h,
                    ])
                    labels.append(int(cls) + 1)  # 0 = background
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return self.transforms(img), target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))
```

**⚠️ Copy dataset ra ổ local trước khi train (bắt buộc, tránh lỗi tốc độ đọc Drive):** Google Drive mount qua Colab (FUSE) đọc rất chậm và **dao động thất thường giữa các phiên** (có phiên đo được 73MB/s, có phiên chỉ 0.1MB/s cho cùng 1 thao tác — do mỗi lần ngắt/kết nối lại là 1 máy ảo mới, độ trễ mạng tới Drive khác nhau, không kiểm soát được). Với Track C đọc **từng ảnh một** bằng `PIL.Image.open()` (không có cơ chế cache như Ultralytics), rủi ro này còn nặng hơn Track B. Copy 1 lần ra ổ local (`/content/`, SSD gắn trực tiếp vào VM) để loại bỏ hoàn toàn phụ thuộc vào tốc độ Drive:
```python
!mkdir -p /content/dataset_local
!cp -r /content/drive/MyDrive/BTL_DeTai4/dataset/* /content/dataset_local/

base = "/content/dataset_local"   # đọc dữ liệu train/val từ đây, không đọc trực tiếp từ Drive
train_ds = YoloDataset(f"{base}/train/images", f"{base}/train/labels")
val_ds = YoloDataset(f"{base}/valid/images", f"{base}/valid/labels")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)
```
Đã thêm `num_workers=2` (mặc định là 0 = đơn luồng) để đọc ảnh song song 2 luồng — kết hợp với đọc từ local, giúp giảm đáng kể thời gian mỗi epoch so với đọc trực tiếp từ Drive.

**Lưu ý:** `/content/` là ổ đĩa tạm của phiên Colab — mất hết khi phiên bị ngắt (gập máy, mất mạng...). Nếu bị ngắt phiên giữa chừng, phải **chạy lại toàn bộ từ Bước 1** (bao gồm cả lệnh copy này) trước khi resume từ checkpoint ở Bước 4 — checkpoint (`fasterrcnn_ckpt.pt`) vẫn an toàn vì nó lưu trên Drive, chỉ có bản copy dữ liệu local là phải tạo lại mỗi phiên mới.

**Bước 3 — Khởi tạo model (nhớ sửa `NUM_CLASSES` theo `nc` Track A vừa báo):**
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 58   # đã xác nhận nc: 58 trong data.yaml (version 5)
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
model.to('cuda')
```

**Bước 4 — Train với checkpoint/resume** (quan trọng vì chạy 5–9 giờ, cần chống mất tiến trình nếu Colab ngắt phiên):
```python
import os, time, torch

torch.manual_seed(42)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

ckpt_path = "/content/drive/MyDrive/BTL_DeTai4/fasterrcnn_ckpt.pt"
start_epoch = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resume từ epoch {start_epoch}")

NUM_EPOCHS = 18   # 15-20 epoch tùy thời gian còn lại
model.train()
t0 = time.time()
for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_loss = 0
    for imgs, targets in train_loader:
        imgs = [i.to('cuda') for i in imgs]
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(
        {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
        ckpt_path,
    )

print(f"Tổng thời gian train phiên này: {(time.time()-t0)/60:.1f} phút")
torch.save(model.state_dict(), "/content/drive/MyDrive/BTL_DeTai4/fasterrcnn_baseline.pt")
```
Nếu Colab ngắt kết nối giữa chừng: mở lại notebook, chạy lại từ Bước 1 — Bước 4 tự đọc `ckpt_path` và resume đúng epoch còn dang dở, không mất công.

> **Ước lượng thời gian:** ~3.000–3.700 ảnh × 18 epoch × batch 4 trên T4 ≈ **6–8 giờ**. Nếu chưa xong cuối ngày 1, để chạy tiếp đầu ngày 2 (checkpoint đã lưu, không sao).

---

## 09:15–12:00 — Track B: YOLOv8 baseline (imgsz=640, 50 epoch)

**Bước 1 — Runtime GPU T4**, cài đặt:
```python
!pip install ultralytics -q
from ultralytics import YOLO
import torch, random, numpy as np

from google.colab import drive
drive.mount('/content/drive')

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

data_yaml_drive = "/content/drive/MyDrive/BTL_DeTai4/dataset/data.yaml"
```
→ Ngay sau cell này, dán cell định nghĩa `log_experiment` (xem mục "Ghi log thí nghiệm tự động" ở trên) — chạy 1 lần.

Kiểm tra `data_yaml_drive` — nếu đường dẫn `train:`/`val:` là đường dẫn tuyệt đối cũ (do Roboflow ghi sẵn đường dẫn lúc tải ở máy Track A), sửa lại thành tương đối `train/images`, `valid/images`, `test/images` cho khớp cấu trúc thư mục trên Drive (xem code sửa `data.yaml` bằng `pyyaml` đã hướng dẫn trước đó — sửa 1 lần trên bản Drive, vì đây là bản gốc dùng chung cho cả Track B và Track C).

**⚠️ Copy dataset ra ổ local trước khi train (bắt buộc, tránh lỗi tốc độ đọc Drive):** Google Drive mount qua Colab (FUSE) đọc rất chậm và **dao động thất thường giữa các phiên** (có phiên đo được 73MB/s, có phiên chỉ 0.1MB/s cho cùng 1 thao tác — do mỗi lần ngắt/kết nối lại là 1 máy ảo mới, độ trễ mạng tới Drive khác nhau, không kiểm soát được). Với tập train 9.536 ảnh (đã tăng cường sẵn từ Roboflow), đọc trực tiếp từ Drive mỗi epoch có thể khiến 1 epoch mất cả giờ đồng hồ. Copy 1 lần ra ổ local (`/content/`, SSD gắn trực tiếp vào VM) để loại bỏ hoàn toàn phụ thuộc vào tốc độ Drive:
```python
!mkdir -p /content/dataset_local
!cp -r /content/drive/MyDrive/BTL_DeTai4/dataset/* /content/dataset_local/

base = "/content/dataset_local"
data_yaml = f"{base}/data.yaml"
```
Từ đây trở đi, mọi lệnh train/predict của Track B dùng `data_yaml` (bản local, nhanh và ổn định) — chỉ riêng kết quả (weights, log, ảnh biểu đồ) mới ghi ra Drive (`project='/content/drive/MyDrive/BTL_DeTai4/runs_yolo'`, không đổi) vì ghi ít lần hơn đọc rất nhiều nên không phải bottleneck.

**Lưu ý:** `/content/` là ổ đĩa tạm, mất hết khi phiên Colab bị ngắt (gập máy, mất mạng...) — nếu bị ngắt giữa chừng, phải chạy lại **toàn bộ Bước 1** (kể cả lệnh copy này) trước khi train tiếp.

**Bước 2 — Train baseline 640, đủ 50 epoch:**
```python
import time
t0 = time.time()

model = YOLO('yolov8n.pt')
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    seed=42,
    project='/content/drive/MyDrive/BTL_DeTai4/runs_yolo',
    name='baseline_640',
    patience=15,
)
train_time_640 = (time.time() - t0) / 60
```
Với ~3.000–3.700 ảnh, 50 epoch trên T4 ≈ **1–2 giờ** (đúng số liệu ước tính trong đề bài).

**Bước 3 — Lấy mAP baseline:**
```python
metrics = model.val()
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
```
`PR_curve.png` và `confusion_matrix.png` tự lưu trong `runs_yolo/baseline_640/` — dùng luôn cho yêu cầu "precision-recall curve từng lớp".

**📝 Ghi log:**
```python
num_params = sum(p.numel() for p in model.model.parameters())

log_experiment({
    "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
    "model": "YOLOv8n", "run_name": "baseline_640",
    "imgsz": 640, "epochs": 50, "batch": 16, "seed": 42,
    "conf": 0.25, "iou": 0.5,
    "precision": metrics.box.mp, "recall": metrics.box.mr,
    "map50": metrics.box.map50, "map50_95": metrics.box.map,
    "fps": "", "num_params": num_params,
    "train_time_min": round(train_time_640, 1),
    "notes": "baseline 640, đủ 50 epoch"
})
```

## 12:00–13:00 — Nghỉ trưa
Track C vẫn chạy nền (không cần canh cả nhóm, 1 người giữ tab mở là đủ).

---

## 13:00–15:30 — Track B: threshold sweep + train thêm 2 resolution (416, 960)

**Bước 1 — Threshold sweep confidence:**
```python
import pandas as pd

conf_values = [0.1, 0.25, 0.4, 0.5, 0.7, 0.9]
rows = []
for c in conf_values:
    m = model.val(conf=c, iou=0.5, split='val')
    rows.append({'conf': c, 'precision': m.box.mp, 'recall': m.box.mr, 'map50': m.box.map50})
    log_experiment({   # 📝 Ghi log — 1 dòng cho mỗi giá trị conf
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "model": "YOLOv8n", "run_name": "conf_sweep",
        "imgsz": 640, "epochs": 50, "batch": 16, "seed": 42,
        "conf": c, "iou": 0.5,
        "precision": m.box.mp, "recall": m.box.mr,
        "map50": m.box.map50, "map50_95": m.box.map,
        "fps": "", "num_params": "", "train_time_min": "",
        "notes": f"conf sweep c={c}"
    })
pd.DataFrame(rows).to_csv('/content/drive/MyDrive/BTL_DeTai4/conf_sweep.csv', index=False)
```

**Bước 2 — Threshold sweep NMS iou:**
```python
iou_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
rows = []
for i in iou_values:
    m = model.val(conf=0.25, iou=i, split='val')
    rows.append({'nms_iou': i, 'precision': m.box.mp, 'recall': m.box.mr, 'map50': m.box.map50})
    log_experiment({   # 📝 Ghi log — 1 dòng cho mỗi giá trị iou
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "model": "YOLOv8n", "run_name": "nms_sweep",
        "imgsz": 640, "epochs": 50, "batch": 16, "seed": 42,
        "conf": 0.25, "iou": i,
        "precision": m.box.mp, "recall": m.box.mr,
        "map50": m.box.map50, "map50_95": m.box.map,
        "fps": "", "num_params": "", "train_time_min": "",
        "notes": f"nms iou sweep i={i}"
    })
pd.DataFrame(rows).to_csv('/content/drive/MyDrive/BTL_DeTai4/nms_sweep.csv', index=False)
```

**Vẽ 2 biểu đồ đường (bắt buộc theo đề bài):**
```python
import matplotlib.pyplot as plt

df_conf = pd.read_csv('/content/drive/MyDrive/BTL_DeTai4/conf_sweep.csv')
plt.figure(figsize=(7,5))
plt.plot(df_conf['conf'], df_conf['precision'], marker='o', label='Precision')
plt.plot(df_conf['conf'], df_conf['recall'], marker='o', label='Recall')
plt.plot(df_conf['conf'], df_conf['map50'], marker='o', label='mAP@0.5')
plt.xlabel('Confidence threshold'); plt.ylabel('Giá trị'); plt.legend()
plt.title('Ảnh hưởng ngưỡng confidence')
plt.savefig('/content/drive/MyDrive/BTL_DeTai4/conf_sweep_plot.png')
plt.show()

df_nms = pd.read_csv('/content/drive/MyDrive/BTL_DeTai4/nms_sweep.csv')
plt.figure(figsize=(7,5))
plt.plot(df_nms['nms_iou'], df_nms['precision'], marker='o', label='Precision')
plt.plot(df_nms['nms_iou'], df_nms['recall'], marker='o', label='Recall')
plt.plot(df_nms['nms_iou'], df_nms['map50'], marker='o', label='mAP@0.5')
plt.xlabel('NMS IoU threshold'); plt.ylabel('Giá trị'); plt.legend()
plt.title('Ảnh hưởng ngưỡng NMS')
plt.savefig('/content/drive/MyDrive/BTL_DeTai4/nms_sweep_plot.png')
plt.show()
```

**Bước 3 — Train thật ở 2 resolution còn lại** (giờ có đủ thời gian, không cần chỉ inference-sweep như bản 1 ngày):
```python
for sz in [416, 960]:
    t0 = time.time()
    m = YOLO('yolov8n.pt')
    batch = 16 if sz <= 640 else 8   # imgsz=960 tốn VRAM hơn, giảm batch tránh OOM trên T4
    m.train(
        data=data_yaml, epochs=50, imgsz=sz, batch=batch, seed=42,
        project='/content/drive/MyDrive/BTL_DeTai4/runs_yolo', name=f'baseline_{sz}',
    )
    train_time_sz = (time.time() - t0) / 60
    metrics = m.val()
    print(sz, metrics.box.map50, metrics.box.map)

    log_experiment({   # 📝 Ghi log — 1 dòng cho mỗi resolution
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "model": "YOLOv8n", "run_name": f"baseline_{sz}",
        "imgsz": sz, "epochs": 50, "batch": batch, "seed": 42,
        "conf": 0.25, "iou": 0.5,
        "precision": metrics.box.mp, "recall": metrics.box.mr,
        "map50": metrics.box.map50, "map50_95": metrics.box.map,
        "fps": "", "num_params": sum(p.numel() for p in m.model.parameters()),
        "train_time_min": round(train_time_sz, 1),
        "notes": f"train lại toàn bộ ở imgsz={sz}"
    })
```
Mỗi lần train ~45–90 phút tùy resolution → làm được cả 2 trong khung 13:00–15:30 (có thể lấn sang chiều muộn nếu ảnh 960 chậm hơn dự kiến).

---

## 15:30–16:00 — Track B: minh họa trực quan bắt buộc (IoU, NMS, ảnh hưởng confidence)

Đề bài yêu cầu **"giải thích bằng lời và bằng ví dụ minh họa từ chính mô hình của nhóm: anchor box, IoU, NMS, ảnh hưởng ngưỡng confidence và NMS"**. Threshold sweep ở Bước 1–2 (13:00–15:30) mới cho **số liệu tổng hợp** (precision/recall/mAP theo ngưỡng) — vẫn thiếu **ảnh minh họa cụ thể trên 1 tấm ảnh thật** để người đọc "nhìn thấy" hiệu ứng. Bổ sung 3 đoạn code sau (dùng model 640):

**⚠️ Lưu ý quan trọng trước khi làm — YOLOv8 KHÔNG có anchor box:** khác với YOLOv5 (dùng anchor cố định), kiến trúc YOLOv8 là **anchor-free** (dự đoán trực tiếp khoảng cách 4 cạnh từ tâm ô lưới bằng Distribution Focal Loss, không có tập anchor định sẵn). Vì vậy **không thể lấy anchor box từ chính model YOLOv8 ra minh họa** — nếu tự vẽ anchor box giả rồi nói "đây là anchor của mô hình" là sai kiến trúc, có thể bị coi là bịa số liệu. Xử lý trung thực: viết rõ trong báo cáo *"YOLOv8 dùng thiết kế anchor-free nên không có anchor box để trực quan hóa từ mô hình; phần này giải thích bằng lời khái niệm anchor-based (từng dùng ở YOLOv5) làm nền so sánh, nhấn mạnh do đâu YOLOv8 bỏ được anchor"* — không cần minh họa bằng ảnh cho riêng mục anchor box.

**1. Minh họa IoU** (vẽ đè ground-truth và prediction lên cùng 1 ảnh, tính IoU thật):
```python
import cv2
import matplotlib.pyplot as plt

test_images = sorted(glob.glob(f"{base}/test/images/*"))
sample_img_path = test_images[0]   # đổi ảnh khác nếu ảnh đầu không có box rõ ràng

img = cv2.cvtColor(cv2.imread(sample_img_path), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

label_path = sample_img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
with open(label_path) as f:
    cls, xc, yc, bw, bh = map(float, f.readline().split())
gt_box = [(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h]

pred = model.predict(sample_img_path, conf=0.25, verbose=False)[0]
pred_box = pred.boxes.xyxy[0].tolist() if len(pred.boxes) > 0 else gt_box

def iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter+1e-6)

iou_val = iou(gt_box, pred_box)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img)
ax.add_patch(plt.Rectangle((gt_box[0],gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                            fill=False, edgecolor='lime', linewidth=2, label='Ground truth'))
ax.add_patch(plt.Rectangle((pred_box[0],pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                            fill=False, edgecolor='red', linewidth=2, label='Prediction'))
ax.set_title(f"Minh họa IoU = {iou_val:.3f}")
ax.legend()
plt.savefig('/content/drive/MyDrive/BTL_DeTai4/illustration_iou.png')
plt.show()
```

**2. Minh họa NMS** (trước/sau khi loại box trùng — dùng `iou=0.99` để gần như tắt NMS, so với `iou=0.5` mặc định):
```python
pred_before = model.predict(sample_img_path, conf=0.05, iou=0.99, verbose=False)[0]
pred_after  = model.predict(sample_img_path, conf=0.25, iou=0.5,  verbose=False)[0]

fig, axes = plt.subplots(1, 2, figsize=(16,8))
for ax, pred, title in zip(axes, [pred_before, pred_after],
                           ["Trước NMS (iou=0.99, conf=0.05)", "Sau NMS (iou=0.5, conf=0.25)"]):
    ax.imshow(img)
    for box in pred.boxes.xyxy.tolist():
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1],
                                    fill=False, edgecolor='red', linewidth=1.5))
    ax.set_title(f"{title} — {len(pred.boxes)} box")
plt.savefig('/content/drive/MyDrive/BTL_DeTai4/illustration_nms.png')
plt.show()
```

**3. Minh họa ảnh hưởng ngưỡng confidence trên cùng 1 ảnh:**
```python
fig, axes = plt.subplots(1, 3, figsize=(20,7))
for ax, c in zip(axes, [0.1, 0.5, 0.9]):
    pred = model.predict(sample_img_path, conf=c, iou=0.5, verbose=False)[0]
    ax.imshow(img)
    for box in pred.boxes.xyxy.tolist():
        ax.add_patch(plt.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1],
                                    fill=False, edgecolor='red', linewidth=1.5))
    ax.set_title(f"conf={c} — {len(pred.boxes)} box")
plt.savefig('/content/drive/MyDrive/BTL_DeTai4/illustration_conf_effect.png')
plt.show()
```
Nếu ảnh `test_images[0]` không có box nào hoặc chỉ có 1 box quá rõ ràng (không thấy hiệu ứng NMS/confidence), thử đổi sang `test_images[5]`, `test_images[10]`... cho tới khi tìm được ảnh có nhiều biển báo gần nhau — hiệu ứng minh họa sẽ rõ hơn.

---

## 16:00–17:00 — Track A: chuẩn bị & test thử script phân tích lỗi

Viết trước hàm match box theo IoU (chưa cần chạy full, chỉ test trên vài ảnh mẫu để chắc code đúng — chạy full sẽ để sáng ngày 2 khi có đủ prediction từ cả 3 model resolution):

```python
def iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def classify_errors(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """Trả về list lỗi: 'correct' | 'misclass' | 'mislocalized' | 'missed' | 'extra'"""
    matched_gt = set()
    results = []
    order = sorted(range(len(pred_scores)), key=lambda i: -pred_scores[i])
    for i in order:
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            cur = iou(pred_boxes[i], gb)
            if cur > best_iou:
                best_iou, best_j = cur, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            results.append(('correct' if pred_labels[i] == gt_labels[best_j] else 'misclass', i, best_j))
        elif best_iou >= 0.1:
            results.append(('mislocalized', i, best_j))
        else:
            results.append(('extra', i, None))
    missed = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    for j in missed:
        results.append(('missed', None, j))
    return results
```

## 17:00–18:00 — Cả nhóm: sync cuối ngày 1
- Kiểm tra Track C còn bao nhiêu epoch (checkpoint hiện đang ở epoch nào).
- Kiểm tra Track B đã có đủ 3 model (640/416/960) với mAP tương ứng chưa.
- Nếu Track C chưa xong: để notebook tiếp tục chạy qua đêm (không đóng tab), sáng mai chạy tiếp từ checkpoint.

---

# NGÀY 2

## 08:00–09:30 — Track C: hoàn tất train, đo mAP & FPS

**Nếu đây là phiên Colab mới** (qua đêm phiên cũ đã ngắt) — phải chạy lại **toàn bộ Bước 1–3** (cài đặt, mount Drive, copy dataset ra `/content/dataset_local`, định nghĩa `YoloDataset`/`log_experiment`, khởi tạo lại model) trước khi chạy tiếp, vì `/content/` là ổ tạm và mọi biến trong RAM đều mất khi ngắt phiên. Chỉ có `ckpt_path` trên Drive là còn nguyên.

Nếu chưa đủ 18 epoch, chạy tiếp cell Bước 4 ở trên (tự resume). Khi đủ epoch:

**Đo mAP bằng torchmetrics:**
```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision()
model.eval()
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = [i.to('cuda') for i in imgs]
        preds = model(imgs)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)

result = metric.compute()
print("mAP:", result['map'].item(), "mAP50:", result['map_50'].item())
```

**Đo FPS trên đúng 100 ảnh test dùng chung với Track B:**
```python
import time, glob

model.eval()
times = []
with torch.no_grad():
    for img_path in sorted(glob.glob(f"{base}/test/images/*"))[:100]:
        img = Image.open(img_path).convert("RGB")
        img_t = T.ToTensor()(img).to('cuda')
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model([img_t])
        torch.cuda.synchronize()
        times.append(time.time() - t0)
fps_frcnn = 1 / (sum(times)/len(times))
print("FPS Faster R-CNN:", fps_frcnn)
```

**📝 Ghi log** (chạy ngay sau 2 đoạn đo mAP + FPS ở trên):
```python
num_params = sum(p.numel() for p in model.parameters())

log_experiment({
    "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
    "model": "FasterRCNN-ResNet50FPN",
    "run_name": "baseline_640",
    "imgsz": 640, "epochs": 18, "batch": 4, "seed": 42,
    "conf": "", "iou": "",
    "precision": "", "recall": "",
    "map50": result['map_50'].item(), "map50_95": result['map'].item(),
    "fps": round(fps_frcnn, 2), "num_params": num_params,
    "train_time_min": "",  # cộng dồn qua nhiều phiên do resume checkpoint — ghi ước tính tổng vào notes
    "notes": "train qua nhiều phiên (checkpoint/resume) do chạy nền xuyên ngày 1 sang ngày 2, xem thời gian từng phiên trong log Colab"
})
```

## 09:00–10:30 — Track B: inference + đo FPS (song song với Track C)

**Nếu đây là phiên Colab mới** — phải chạy lại Bước 1 (cài đặt, mount Drive, copy dataset ra `/content/dataset_local`, định nghĩa lại `base`, `data_yaml`, `log_experiment`) trước khi chạy các lệnh dưới đây, vì các biến này chỉ tồn tại trong RAM của phiên cũ. Weights (`best.pt`) không mất vì đã lưu trên Drive.

**Inference lưu prediction (dùng model 640 làm bản chính để phân tích lỗi):**
```python
model = YOLO('/content/drive/MyDrive/BTL_DeTai4/runs_yolo/baseline_640/weights/best.pt')
results = model.predict(
    source=f"{base}/test/images",
    conf=0.25, iou=0.5,
    save=True, save_txt=True, save_conf=True,
    project='/content/drive/MyDrive/BTL_DeTai4/runs_yolo', name='test_predictions'
)
```

**Đo FPS trên đúng 100 ảnh test (giống thứ tự Track C dùng):**
```python
times = []
for img_path in sorted(glob.glob(f"{base}/test/images/*"))[:100]:
    t0 = time.time()
    _ = model.predict(img_path, verbose=False)
    times.append(time.time() - t0)
fps_640 = 1 / (sum(times)/len(times))
print("FPS YOLOv8n (640):", fps_640)
```

**📝 Ghi log FPS cho model 640** (cập nhật lại dòng `baseline_640` với FPS đo được — vì lúc train ở Bước 3 chưa đo FPS):
```python
log_experiment({
    "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
    "model": "YOLOv8n", "run_name": "baseline_640_fps",
    "imgsz": 640, "epochs": 50, "batch": 16, "seed": 42,
    "conf": 0.25, "iou": 0.5,
    "precision": "", "recall": "", "map50": "", "map50_95": "",
    "fps": round(fps_640, 2), "num_params": sum(p.numel() for p in model.model.parameters()),
    "train_time_min": "", "notes": "đo FPS trên 100 ảnh test cố định, cùng máy với Track C"
})
```

**Lấy mAP + FPS của 2 model resolution còn lại (416, 960) để có đủ bảng đánh đổi mAP-FPS:**
```python
for sz in [416, 960]:
    m = YOLO(f'/content/drive/MyDrive/BTL_DeTai4/runs_yolo/baseline_{sz}/weights/best.pt')
    metrics = m.val(imgsz=sz)
    times = []
    for img_path in sorted(glob.glob(f"{base}/test/images/*"))[:100]:
        t0 = time.time()
        _ = m.predict(img_path, imgsz=sz, verbose=False)
        times.append(time.time() - t0)
    fps_sz = 1 / (sum(times)/len(times))
    print(sz, metrics.box.map50, metrics.box.map, fps_sz)

    log_experiment({   # 📝 Ghi log — 1 dòng cho mỗi resolution
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "model": "YOLOv8n", "run_name": f"baseline_{sz}_fps",
        "imgsz": sz, "epochs": 50, "batch": "", "seed": 42,
        "conf": 0.25, "iou": 0.5,
        "precision": metrics.box.mp, "recall": metrics.box.mr,
        "map50": metrics.box.map50, "map50_95": metrics.box.map,
        "fps": round(fps_sz, 2), "num_params": sum(p.numel() for p in m.model.parameters()),
        "train_time_min": "", "notes": f"đo mAP + FPS ở imgsz={sz} trên 100 ảnh test cố định"
    })
```

## 10:30–13:00 — Track A + B: chạy phân tích lỗi thật trên model 640

**Bước 1 — Đọc toàn bộ cặp file ground-truth/prediction, phân loại lỗi:**
```python
import glob, os

gt_label_dir = f"{base}/test/labels"
pred_label_dir = "/content/drive/MyDrive/BTL_DeTai4/runs_yolo/test_predictions/labels"

all_results = []   # (img_path, list các (tag, pred_idx, gt_idx))

for img_path in sorted(glob.glob(f"{base}/test/images/*")):
    stem = os.path.basename(img_path).rsplit('.', 1)[0]
    img = Image.open(img_path)
    w, h = img.size

    gt_boxes, gt_labels = [], []
    gt_path = f"{gt_label_dir}/{stem}.txt"
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                gt_boxes.append([(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h])
                gt_labels.append(int(cls))

    pred_boxes, pred_labels, pred_scores = [], [], []
    pred_path = f"{pred_label_dir}/{stem}.txt"
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls, xc, yc, bw, bh = parts[:5]
                conf = parts[5] if len(parts) > 5 else 1.0
                pred_boxes.append([(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h])
                pred_labels.append(int(cls))
                pred_scores.append(conf)

    tags = classify_errors(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
    all_results.append((img_path, gt_boxes, gt_labels, pred_boxes, pred_labels, tags))
```

**Bước 2 — Đếm số lượng mỗi loại lỗi (tổng thể + theo kích thước vật thể):**
```python
from collections import Counter

error_count = Counter()
error_by_size = Counter()

def box_size_group(box):
    area = ((box[2]-box[0]) * (box[3]-box[1])) / (w*h)
    if area < (32/640)**2: return 'small'
    elif area < (96/640)**2: return 'medium'
    else: return 'large'

for img_path, gt_boxes, gt_labels, pred_boxes, pred_labels, tags in all_results:
    for tag, pred_idx, gt_idx in tags:
        error_count[tag] += 1
        ref_box = pred_boxes[pred_idx] if pred_idx is not None else gt_boxes[gt_idx]
        error_by_size[(tag, box_size_group(ref_box))] += 1

print(error_count)
print(error_by_size)

pd.DataFrame(error_count.items(), columns=['error_type', 'count']).to_csv(
    '/content/drive/MyDrive/BTL_DeTai4/error_analysis.csv', index=False)
```

**Bước 3 — Vẽ và lưu ≥ 8 ảnh minh họa (≥2 ảnh/loại lỗi nếu có):**
```python
import cv2

os.makedirs('/content/drive/MyDrive/BTL_DeTai4/error_examples', exist_ok=True)
saved_per_type = Counter()
MAX_PER_TYPE = 3

for img_path, gt_boxes, gt_labels, pred_boxes, pred_labels, tags in all_results:
    error_types_here = set(t[0] for t in tags if t[0] != 'correct')
    if not error_types_here:
        continue
    todo = [t for t in error_types_here if saved_per_type[t] < MAX_PER_TYPE]
    if not todo:
        continue

    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).copy()
    for gb in gt_boxes:
        x1, y1, x2, y2 = map(int, gb)
        cv2.rectangle(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)   # xanh = ground truth
    for pb in pred_boxes:
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(img_arr, (x1, y1), (x2, y2), (255, 0, 0), 2)   # đỏ = prediction

    plt.figure(figsize=(8, 8))
    plt.imshow(img_arr)
    plt.title(f"Lỗi: {', '.join(error_types_here)}")
    plt.axis('off')
    fname = f"/content/drive/MyDrive/BTL_DeTai4/error_examples/{os.path.basename(img_path)}"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    for t in todo:
        saved_per_type[t] += 1

print("Đã lưu theo loại lỗi:", saved_per_type)
```
Kiểm tra `saved_per_type` — nếu tổng số ảnh lưu được < 8 (thường do 1 vài loại lỗi hiếm gặp, ví dụ `mislocalized` ít xảy ra với model đã hội tụ tốt), tăng `MAX_PER_TYPE` lên hoặc chấp nhận ít hơn 2 ảnh cho loại lỗi hiếm — ghi rõ lý do trong báo cáo (ít lỗi loại đó cũng là 1 phát hiện đáng nêu, không phải thiếu sót).

## 13:00–14:00 — Nghỉ trưa

## 14:00–15:30 — Cả nhóm: tổng hợp bảng so sánh đầy đủ

Vì mọi lượt chạy đã được `log_experiment` ghi vào `experiment_log.csv` xuyên suốt 2 ngày, không cần tự điền tay — đọc thẳng và lọc ra bảng cuối:

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/BTL_DeTai4/experiment_log.csv')

# Mỗi run_name có thể có 2 dòng (1 dòng lúc train có mAP, 1 dòng lúc đo FPS) — gộp lại theo model+imgsz
summary = df[df['run_name'].str.contains('baseline', na=False)].groupby(
    ['model', 'imgsz']
).agg({
    'map50': 'max', 'map50_95': 'max', 'fps': 'max',
    'num_params': 'max', 'train_time_min': 'max'
}).reset_index()
summary
```

Kết quả `summary` chính là bảng cần đưa vào báo cáo:

| Mô hình | Resolution | mAP@0.5 | mAP@0.5:0.95 | FPS (T4, 100 ảnh) | Số tham số | Thời gian train |
|---|---|---|---|---|---|---|
| YOLOv8n | 416 | | | | | |
| YOLOv8n | 640 | | | | | |
| YOLOv8n | 960 | | | | | |
| Faster R-CNN (ResNet-50 FPN) | 640 | | | | | |

Từ bảng này vẽ luôn biểu đồ trade-off **mAP vs FPS theo resolution** — đúng yêu cầu "yêu cầu khác" của đề bài, giờ làm bằng số liệu train thật chứ không phải ước lượng.

## 15:30–17:00 — Review chéo, viết README, kiểm tra tái lập
- Đổi notebook cho nhau, chạy thử lại vài cell quan trọng (đặc biệt: model có load đúng checkpoint không, threshold sweep có chạy lại ra cùng số không).
- Viết README: cách chạy từng notebook, đường dẫn Drive chứa weights, bảng số liệu chính, trích dẫn nguồn dataset (link + license đã lưu ở mục 0).

## 17:00–18:00 — Buffer + chốt ngày
- Đẩy code lên GitHub (weights nặng để trên Drive, dẫn link trong README).
- Hoàn thiện nhật ký thí nghiệm CSV — đủ cả 2 ngày, mọi lượt chạy kể cả lượt lỗi.
- Nếu Faster R-CNN không kịp đủ 18 epoch vì lý do phát sinh, ghi rõ số epoch thực tế đã chạy được và lý do — trung thực vẫn tốt hơn số liệu đẹp không giải thích được.

---

## Việc bị lược bỏ ngay cả trong bản 2 ngày (để dành cho phần "sáng tạo" nếu còn dư thời gian)
- So sánh label smoothing/Mixup/CutMix (đây là yêu cầu của Đề tài 3, không bắt buộc ở Đề tài 4).
- Train Faster R-CNN ở cả 3 resolution (chỉ làm ở 640 để tiết kiệm thời gian máy — đây là lựa chọn hợp lý, ghi rõ lý do trong README).
- Video demo / app Gradio-Streamlit (tùy chọn cộng điểm, làm nếu nhóm còn dư thời gian sau 18:00 ngày 2).
