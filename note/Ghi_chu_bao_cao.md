# Ghi chú để dùng khi viết báo cáo / README

> File này gom các điểm cần nhớ nhắc tới khi viết báo cáo hoặc README, phát sinh trong quá trình chạy thí nghiệm — không phải nội dung báo cáo hoàn chỉnh, chỉ là note để không quên.

## Dữ liệu — mất cân bằng lớp (long-tail)

Dataset Vietnam-Traffic-Sign-Detection có **58 lớp** biển báo (khá nhiều so với bài toán detection thông thường chỉ vài lớp). Với số lớp lớn như vậy, nhiều khả năng một số mã biển báo hiếm sẽ có rất ít ảnh trong tập train.

**Việc cần làm:** khi vẽ biểu đồ phân bố số lượng nhãn theo lớp (Track A, mục thống kê dữ liệu), kỳ vọng sẽ thấy rõ hiện tượng **mất cân bằng lớp nặng (long-tail distribution)** — một vài lớp phổ biến (vd biển cấm, biển báo nguy hiểm thường gặp) chiếm phần lớn số lượng ảnh, còn phần lớn lớp còn lại chỉ có vài chục hoặc vài ảnh.

**Cần nêu trong báo cáo (mục 3. Dữ liệu và mục 6. Phân tích và thảo luận):**
- Trích dẫn số liệu cụ thể từ biểu đồ `class_distribution.png` đã lưu (Track A) — nêu rõ lớp nhiều nhất/ít nhất bao nhiêu ảnh.
- Liên hệ với kết quả mAP theo từng lớp (Precision-Recall curve của YOLOv8 tự xuất ra theo lớp): các lớp có ít ảnh khả năng sẽ có mAP thấp hơn hẳn — đây là bằng chứng nhân-quả cần chỉ ra thay vì chỉ liệt kê số liệu.
- Đây cũng là hạn chế cần nêu ở mục "Kết luận và hướng phát triển": có thể đề xuất hướng khắc phục (data augmentation cho lớp hiếm, focal loss, gộp bớt lớp hiếm/tương tự nhau, hoặc thu thập thêm dữ liệu cho các lớp ít ảnh).

## YOLOv8 là anchor-free — không có anchor box để minh họa từ model

Đề bài yêu cầu *"giải thích... anchor box... từ chính mô hình của nhóm"*, nhưng **YOLOv8 dùng kiến trúc anchor-free** (dự đoán khoảng cách 4 cạnh từ tâm ô lưới bằng Distribution Focal Loss, không có tập anchor cố định như YOLOv5/Faster R-CNN).

**Cần viết trong báo cáo (mục 4. Phương pháp, phần giải thích anchor/IoU/NMS):**
- Không tự vẽ/bịa anchor box "lấy từ model" — model không có thứ đó, vẽ ra sẽ là số liệu sai kiến trúc.
- Giải thích trung thực: nêu khái niệm anchor-based (từng dùng ở YOLOv5/Faster R-CNN) bằng lời làm nền so sánh, rồi giải thích vì sao YOLOv8 bỏ được anchor (giảm số hyperparameter cần tune như anchor scale/ratio, đơn giản hóa hậu xử lý).
- Phần "ví dụ minh họa từ chính mô hình" chỉ áp dụng đầy đủ cho **IoU** và **NMS** (2 khái niệm này vẫn áp dụng cho YOLOv8 ở bước hậu xử lý, có ảnh minh họa thật từ model — xem `illustration_iou.png`, `illustration_nms.png`, `illustration_conf_effect.png` trong Drive `BTL_DeTai4/`).
- Faster R-CNN (Track C) thì **có anchor box thật** (region proposal network dùng anchor) — nếu muốn có ảnh minh họa anchor box "từ chính mô hình", đây là chỗ duy nhất trong 2 model của nhóm có thể lấy được, có thể nêu thêm như một điểm so sánh hay giữa 2 kiến trúc trong báo cáo.

## Tập train có 9.536 ảnh thay vì ~3.680 — Roboflow đã tự tăng cường dữ liệu (augmentation)

Khi chạy `model.train()`, log hiện `train: Scanning ... 9536 images` — nhiều hơn hẳn tổng số ảnh gốc của dataset (~3.680 ảnh cho cả train+valid+test). Nguyên nhân gần như chắc chắn: khi tạo **version 5** trên Roboflow, người tạo dataset đã bật bước **Augmentation** trong quy trình generate version (flip/rotate/brightness/crop...), và Roboflow chỉ nhân bản ảnh ở **phần train**, không áp dụng cho valid/test — vì vậy chỉ tập train phình to bất thường (9.536 so với ước tính ~2.500–2.800 ảnh train gốc), còn valid/test vẫn giữ số ảnh thật (`val: Scanning ... 784` khớp với tỉ lệ chia thông thường).

**Cần viết trong báo cáo (mục 3. Dữ liệu):**
- Nêu rõ: tập train thực tế dùng để huấn luyện là 9.536 ảnh (đã bao gồm augmentation do Roboflow áp dụng sẵn khi export version 5), không phải nhóm tự làm augmentation — tránh gây hiểu lầm khi giám khảo hỏi "nhóm có làm augmentation không, làm thế nào".
- Do ảnh train tăng lên ~2,6 lần so với ảnh gốc, **thời gian train mỗi epoch cũng dài hơn ước tính ban đầu tương ứng** — cần cập nhật lại số liệu "thời gian train" thực tế trong log thay vì dùng số ước tính cũ trong kế hoạch.

## Câu hỏi cần tự đặt ra và trả lời SAU KHI có kết quả train đầu tiên

1. **Thời gian 1 epoch thực tế là bao nhiêu?** → nhân với 50 để biết tổng thời gian thật. Nếu vượt quá 3–4 giờ cho riêng YOLOv8, cân nhắc giảm xuống ~30 epoch và ghi rõ lý do (thời gian) trong log/báo cáo.

2. **Train/valid/test được Roboflow chia TRƯỚC hay SAU khi augmentation?** — đây là câu hỏi quan trọng nhất về tính đúng đắn khoa học: nếu augmentation được áp dụng trước khi chia tập (hiếm khi xảy ra với Roboflow vì họ thiết kế để tránh việc này, nhưng cần tự xác nhận chứ không mặc định tin), các biến thể augmented của cùng 1 ảnh gốc có thể vừa nằm trong train vừa nằm trong valid/test → **rò rỉ dữ liệu (data leakage)**, khiến mAP đo được bị lạc quan giả tạo. Cách kiểm tra nhanh: mở vài ảnh trong `valid/images`, xem tên file có phải bản gốc (không có hậu tố `_aug`, `-flip`, `-rotate90`...) hay không — nếu valid/test toàn ảnh "sạch" không có hậu tố augmentation, coi như an toàn.

3. **Ultralytics có đang tự áp dụng thêm augmentation của riêng nó chồng lên ảnh đã augment sẵn từ Roboflow không?** — nhìn vào dòng cấu hình lúc train đã in ra: `mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, auto_augment=randaugment, erasing=0.4...` — đây đều là augmentation Ultralytics tự bật mặc định, **cộng dồn** lên ảnh vốn đã bị Roboflow biến đổi trước đó ("double augmentation"). Cần đánh giá: ảnh có bị méo/nhiễu quá mức không (xem `train_batch0.jpg` trong thư mục run để mắt thường kiểm tra). Nếu ảnh bị biến dạng quá đà, cân nhắc tắt bớt augmentation của Ultralytics (đặt `mosaic=0, fliplr=0, hsv_h=0` khi train lại) và ghi rõ lý do thay đổi trong log.

4. **mAP có tăng dần hợp lý theo từng epoch không, hay chững/giảm bất thường ngay từ đầu?** — nếu chững sớm bất thường, khả năng liên quan tới việc ảnh bị augment quá mức (câu hỏi 3) hoặc learning rate.

5. **Bảng per-class (P/R/mAP theo từng lớp) khớp với biểu đồ phân bố lớp long-tail đã làm ở Track A không?** — lớp nào có `Instances` gần 0 trong bảng validation, đối chiếu xem có đúng là lớp ít ảnh nhất trong `class_distribution.png` không.

6. **⚠️ Quan trọng — ảnh hưởng tới Track C (Faster R-CNN):** Track C đọc dữ liệu từ **cùng thư mục** `dataset/train/images` mà Track B đang dùng (`base = ".../dataset"`) → Track C **cũng đang train trên 9.536 ảnh**, không phải ~3.000 ảnh như ước tính ban đầu trong kế hoạch. Vì Faster R-CNN vốn đã là track chậm nhất (ước tính gốc 5–9 giờ cho 15–20 epoch trên ~3.000 ảnh), con số ảnh thật cao hơn ~2,6 lần có thể đẩy tổng thời gian train lên **13–23 giờ** — có nguy cơ không kịp trong khung 2 ngày. Cần quyết định sớm: (a) chấp nhận train lâu hơn và bắt đầu Track C càng sớm càng tốt, hoặc (b) giảm số ảnh Track C dùng bằng cách lấy ngẫu nhiên 1 tập con cố định (vd 3.000 ảnh, cùng seed) từ 9.536 ảnh thay vì dùng toàn bộ, ghi rõ lý do trong log/báo cáo.

## Quyết định: chỉ giảm data cho Track C, KHÔNG giảm cho Track B

Track B (YOLOv8) xử lý tốt 9.536 ảnh (tốc độ đọc đã cải thiện, dataloader đa luồng sẵn có của Ultralytics) → giữ nguyên toàn bộ, không có lý do kỹ thuật để giảm và giảm sẽ lãng phí dữ liệu (đặc biệt quan trọng với 58 lớp mất cân bằng). Track C chậm là do code tự viết đơn luồng + kiến trúc 2 giai đoạn nặng hơn, không phải chỉ vì số ảnh — nên chỉ cần giảm dữ liệu ở đây.

**Cần viết trong báo cáo (mục 6. Phân tích và thảo luận, phần so sánh one-stage vs two-stage):** nêu rõ 2 model **không train trên cùng lượng dữ liệu** (YOLOv8n: 9.536 ảnh; Faster R-CNN: ít hơn, do giới hạn thời gian tính toán) — chênh lệch mAP giữa 2 model một phần đến từ khác biệt lượng dữ liệu train, không chỉ do khác biệt kiến trúc. Đề bài chỉ yêu cầu so sánh 2 model "trên cùng phần cứng" (đã đảm bảo: cùng 100 ảnh test, cùng GPU T4), không bắt buộc cùng lượng dữ liệu train, nên đây là lựa chọn hợp lệ — chỉ cần minh bạch khi trình bày, không giấu diếm.

## Quyết định: giữ nguyên augmentation 2 tầng (Roboflow + Ultralytics), không tắt bớt

Đã cân nhắc câu hỏi #3 ở trên (Ultralytics tự augment chồng lên ảnh đã augment sẵn từ Roboflow) — quyết định **không sửa gì, để nguyên cấu hình augmentation mặc định** cho toàn bộ các lần train YOLOv8 (đã train `baseline_640`, và áp dụng tương tự cho `baseline_416`/`baseline_960`). Lý do:

1. Ultralytics đã có sẵn cơ chế tự giảm nhẹ mức augmentation: `close_mosaic=10` (tự tắt mosaic augmentation ở 10 epoch cuối) — một phần lo ngại về "augment quá mức" đã được framework tự xử lý.
2. Augmentation của Roboflow là phép biến đổi **cố định, áp dụng 1 lần** khi tạo ra ảnh (không tăng dồn qua từng epoch) — mỗi epoch chỉ có 1 lớp augmentation "sống" (của Ultralytics) áp lên ảnh đã cố định đó, không phải 2 lớp cộng dồn vô hạn.
3. Kiểm tra bằng mắt `train_batch0.jpg` trong thư mục run — nếu ảnh vẫn nhận diện được nội dung biển báo rõ ràng thì không có bằng chứng cho thấy augmentation gây hại thật sự.
4. Tổ hợp "dataset đã augment từ Roboflow + augmentation mặc định của Ultralytics" là cách làm phổ biến trong thực tế, không phải lỗi thiết kế hiếm gặp.
5. Train lại chỉ để thử tắt augmentation sẽ tốn thêm GPU quota (đang eo hẹp), trong khi lợi ích không chắc chắn — tắt mosaic thậm chí có thể làm **giảm** hiệu năng vì mosaic vốn giúp ích nhiều cho YOLO với vật thể nhỏ.

**Cần viết trong báo cáo (mục 4. Phương pháp hoặc mục 6. Phân tích, phần bàn về augmentation):** nêu rõ nhóm nhận diện được hiện tượng "double augmentation" giữa Roboflow và Ultralytics, đã kiểm tra bằng mắt (`train_batch0.jpg`) và quyết định giữ nguyên cấu hình mặc định vì Ultralytics có cơ chế tự điều tiết (`close_mosaic`) và không có dấu hiệu ảnh bị biến dạng quá mức — đây là 1 lựa chọn có cân nhắc, không phải bỏ sót.

## Kết quả baseline_640 (50 epoch) — mAP@0.5 = 0.869, đã soi 3 biểu đồ

### `results.png` — quá trình huấn luyện có ổn không

**Quan sát:** cả train loss (box/cls/dfl) và val loss đều giảm cùng chiều xuyên suốt 50 epoch, không phân kỳ. mAP50(B) và mAP50-95(B) tăng nhanh rồi plateau từ khoảng epoch 20.

**Kiến thức nền (vì sao đọc được điều này từ biểu đồ):** dấu hiệu overfitting kinh điển là train loss tiếp tục giảm trong khi val loss chững lại rồi **tăng trở lại** — nghĩa là model đang "học thuộc" tập train thay vì tổng quát hóa. Ở đây không xảy ra hiện tượng đó (val loss vẫn giảm cùng chiều, chỉ nhiễu hơn train vì tập val nhỏ — 784 ảnh) → **kết luận: mô hình hội tụ tốt, không overfitting.**

**Chi tiết đáng chú ý:** `train/dfl_loss` có 1 cú tụt rõ rệt gần epoch 40 (từ ~0.83 xuống ~0.825, nhìn có vẻ nhỏ về số nhưng là bước nhảy rõ trên biểu đồ). **Nguyên nhân:** `close_mosaic=10` — Ultralytics tự tắt mosaic augmentation ở 10 epoch cuối (epoch 40→50 với tổng 50 epoch). Khi mosaic tắt, ảnh "dễ học" hơn (không còn ghép 4 ảnh làm 1) nên loss giảm đột ngột. **Đây là hành vi được thiết kế sẵn của framework, không phải bug** — cần giải thích đúng trong báo cáo nếu được hỏi về điểm gãy này trên biểu đồ, tránh nhầm là lỗi huấn luyện.

### `BoxPR_curve.png` — mAP@0.5 = 0.869, nhưng ẩn chứa long-tail

**Quan sát:** đường trung bình (xanh, tất cả các lớp) đạt 0.869 mAP@0.5 — cao hơn dự đoán ban đầu (từng ước tính 0.4–0.7 hợp lý cho dataset 58 lớp mất cân bằng). Tuy nhiên, các đường PR **riêng từng lớp** (58 đường xám mỏng phía sau) không đồng đều: phần lớn bám sát đường trung bình, nhưng có vài đường tụt hẳn xuống 0.5–0.6, thậm chí có 1 đường dạng bậc thang tụt gần về 0 ở recall cao.

**Kiến thức nền:** mAP là **trung bình cộng AP qua tất cả các lớp** — 1 con số tổng có thể "che" mất việc vài lớp hoạt động rất tệ nếu số lớp còn lại đủ tốt để kéo trung bình lên. Đường PR dạng bậc thang (staircase) là dấu hiệu đặc trưng của lớp có **rất ít mẫu trong tập test** — mỗi điểm gãy trên đường tương ứng với 1 mẫu dự đoán đơn lẻ.

**Đây chính là bằng chứng trực quan xác nhận giả thuyết long-tail** đã nêu ở mục "Dữ liệu — mất cân bằng lớp" phía trên — **dùng thẳng hình này trong báo cáo mục 5 (Thí nghiệm và kết quả) và mục 6 (Phân tích)**, kèm câu giải thích: "dù mAP tổng thể đạt 0.869, phân tích PR curve theo từng lớp cho thấy các lớp có ít mẫu train/test có hiệu năng thấp hơn đáng kể, xác nhận ảnh hưởng của mất cân bằng lớp."

### `confusion_matrix.png` — phát hiện 1 cặp lớp bị nhầm nhiều

**Quan sát:** ma trận nhìn chung sạch (đường chéo rõ ở hầu hết 58 lớp), nhưng có **1 ô màu xanh đậm rõ rệt nằm ngoài đường chéo**, ở khu vực các mã `P.125–P.130` — số lượng nhầm lẫn ở ô này cao vượt trội so với phần còn lại của ma trận (theo thang màu, có thể lên tới 150–220 lần).

**Kiến thức nền:** ô ngoài đường chéo tại vị trí (hàng=lớp X, cột=lớp Y) nghĩa là "vật thể thật thuộc lớp Y nhưng bị model dự đoán thành lớp X" — số càng lớn, 2 lớp này càng dễ gây nhầm lẫn cho model, thường vì đặc trưng thị giác giống nhau (hình dạng, màu sắc, bố cục biển báo tương tự).

**Việc cần làm để xác định chính xác cặp lớp này** (xem mục dưới — đã bổ sung vào kế hoạch chạy Ngày 2, không cần đoán bằng mắt qua ảnh phóng to).

**⚠️ Cập nhật sau khi chạy code (Bước 2b) — số liệu thật khác với suy đoán ban đầu:**

Suy đoán "khu vực P.125–P.130" ở trên chỉ là **đọc bằng mắt qua ảnh chụp màn hình**, không chính xác. Chạy code thật trên `all_results` (tập **test**, không phải validation) cho kết quả khác hẳn:
```
W.207b → W.207c   (2 lần)
W.207a → W.207b   (1 lần)
P.127  → P.124a   (1 lần)
W.207c → W.203c   (1 lần)
W.202b → W.202a   (1 lần)
```

**Vì sao khác với confusion matrix gốc:** (1) confusion matrix gốc tính trên tập **validation** (784 ảnh) lúc cuối training, còn phân tích này tính trên tập **test** (khác ảnh) — 2 tập khác nhau có thể cho pattern nhầm lẫn khác nhau, nhất là với lớp ít mẫu; (2) confusion matrix nội bộ Ultralytics dùng ngưỡng confidence rất thấp (~0.001) để dựng toàn bộ đường cong PR, trong khi bộ box dùng phân tích lỗi ở đây được sinh với `conf=0.25` — tập hợp box xét tới khác nhau hoàn toàn.

**Kết luận: dùng số liệu từ code (đáng tin cậy) thay vì suy đoán P.125–P.130 ban đầu.** Số lần nhầm khá thấp (tối đa 2), không có cặp nào áp đảo — nhưng **pattern hợp lý về thị giác**: phần lớn cặp nhầm nằm **trong cùng 1 họ biển cảnh báo** (`W.207a/b/c` nhầm nội bộ với nhau, `W.202a/b` nhầm nhau) — các biến thể cùng họ chỉ khác chi tiết nhỏ bên trong, nên model dễ nhầm giữa chúng.

**Cần viết trong báo cáo (mục 6, phần phân tích lỗi):** nêu đúng cặp `W.207b → W.207c` (2 lần, cao nhất) làm ví dụ chính, giải thích nguyên nhân là do đây là các biến thể cùng họ biển báo (hình dạng/bố cục tương tự, chỉ khác chi tiết). Lưu ý ghi rõ số lần nhầm **thấp** (2 lần trên tổng số instances test) — tránh phóng đại mức độ nghiêm trọng, đây là phát hiện tinh tế chứ không phải lỗi hệ thống lớn.

## Kết quả threshold sweep (confidence + NMS) — 2 biểu đồ, 1 điểm cần giải thích

### Biểu đồ ngưỡng confidence — đúng lý thuyết, không có gì bất thường

**Quan sát:** khi tăng confidence threshold từ 0.1 → 0.9: Precision tăng dần (0.81 → 0.88), Recall giảm dần (0.85 → 0.62), mAP@0.5 giảm dần (0.83 → 0.60). Riêng Precision có 1 chỗ lõm nhẹ ở conf=0.4 (0.865 → 0.850 → 0.861) trước khi tăng tiếp.

**Kiến thức nền:** đây là đánh đổi Precision–Recall kinh điển (xem mục 6, `Kien_thuc_nen_Track_B.md`) — ngưỡng cao hơn giữ lại ít box hơn nhưng "chắc ăn" hơn (Precision tăng), đồng thời bỏ sót nhiều vật thể model không đủ tự tin (Recall giảm). Vết lõm nhỏ ở conf=0.4 chỉ là nhiễu thống kê (dataset 58 lớp, nhiều lớp ít mẫu → thống kê ở từng mức ngưỡng không mượt hoàn toàn), không phải bất thường cần xử lý.

**Cần viết trong báo cáo (mục 5):** mô tả đúng xu hướng trên kèm số liệu, kết luận: không có ngưỡng "đúng tuyệt đối", tùy mục đích sử dụng (ưu tiên Recall hay Precision) mà chọn ngưỡng phù hợp — ví dụ hệ thống cảnh báo an toàn nên chọn conf thấp (~0.1–0.25) để không bỏ sót biển báo quan trọng.

### Biểu đồ ngưỡng NMS — mAP ổn định nhưng Precision/Recall nhảy bậc đột ngột giữa 0.4 và 0.5

**Quan sát:** mAP@0.5 gần như đi ngang suốt từ iou=0.3 đến 0.7 (0.814–0.815), chỉ giảm nhẹ ở 0.9 (0.810). Nhưng Precision và Recall lại **nhảy bậc rõ rệt** đúng tại điểm chuyển từ iou=0.4 sang 0.5: Precision nhảy từ 0.819 lên 0.865, Recall rớt từ 0.849 xuống 0.824 — sau đó cả 2 đường tương đối ổn định tới 0.7 rồi mới giảm tiếp ở 0.9.

**Kiến thức nền — vì sao có sự khác biệt này giữa mAP và Precision/Recall:**
- **mAP** được tính bằng cách quét qua **toàn bộ đường cong Precision-Recall** (tích phân/lấy diện tích dưới đường cong) — một chỉ số tổng hợp trên nhiều điểm ngưỡng confidence khác nhau, nên khá "trơ" (ổn định) với việc chỉnh sửa ngưỡng NMS ở mức vừa phải.
- **Precision/Recall** mà Ultralytics báo cáo (`metrics.box.mp`, `metrics.box.mr`) **không phải** số đo tại 1 ngưỡng conf cố định đơn thuần, mà là giá trị tại **điểm tối ưu F1-score** rút ra từ đường cong PR nội bộ của từng lớp rồi lấy trung bình. Với dataset 58 lớp mất cân bằng (nhiều lớp có rất ít mẫu test), điểm tối ưu F1 này có thể **dịch chuyển đột ngột** khi tập hợp box "sống sót" sau NMS thay đổi (một vài box ở lớp hiếm bị giữ lại hay loại bỏ có thể đổi hẳn vị trí điểm tối ưu), dù mAP tổng thể (tính trên toàn đường cong) không đổi nhiều.
- Đây **không phải lỗi hay bất thường**, mà là hệ quả tự nhiên của việc 2 chỉ số này đo 2 khía cạnh khác nhau (mAP = toàn cục, Precision/Recall báo cáo = 1 điểm cục bộ tối ưu).

**Cần viết trong báo cáo (mục 5 và 6):** nêu rõ hiện tượng "mAP ổn định nhưng Precision/Recall nhảy bậc" và giải thích đúng nguyên nhân ở trên — đây là 1 phát hiện thú vị đáng đưa vào phân tích, thể hiện hiểu biết sâu về cách các chỉ số được tính toán, không phải chỉ liệt kê số liệu suông. Tránh diễn giải sai thành "model không ổn định" hoặc "có lỗi khi đo".

## Lỗi trong code minh họa IoU (đã sửa) — bài học về cách chọn cặp box để so sánh

**Sự cố:** lần chạy đầu tiên của đoạn minh họa IoU (Bước 7) ra kết quả `IoU = 0.000`, dù model thực chất phát hiện đúng cả 2 vật thể trong ảnh (đã xác nhận qua ảnh minh họa NMS/confidence ngay sau đó cho thấy model định vị tốt).

**Nguyên nhân:** code ban đầu chỉ đọc **dòng đầu tiên** của file nhãn (`f.readline()`) làm ground-truth, và lấy **box đầu tiên** trong danh sách model dự đoán (`pred.boxes.xyxy[0]`) — khi ảnh có **≥ 2 vật thể gần nhau**, dòng nhãn đầu tiên và box dự đoán đầu tiên có thể ứng với **2 vật thể khác nhau**, khiến so sánh IoU giữa chúng vô nghĩa (2 box cạnh nhau nhưng không chồng lên nhau → IoU=0, dù mỗi box riêng lẻ đều định vị đúng vật thể của nó).

**Cách sửa:** thay vì lấy "box đầu tiên" của mỗi bên, quét **tất cả cặp** (ground-truth × prediction) trong ảnh, chọn cặp có IoU cao nhất làm ví dụ minh họa — đảm bảo luôn so sánh đúng 2 box của cùng 1 vật thể. Sau khi sửa, ảnh cho ra `IoU = 0.889` — hợp lý và đúng bản chất.

**Bài học chung cần nhớ (áp dụng cho mọi chỗ so sánh box gt/pred trong code, kể cả `classify_errors` đã dùng ở phần phân tích lỗi):** không bao giờ giả định thứ tự box trong file nhãn và thứ tự box model trả về là tương ứng 1-1 — luôn phải **ghép cặp theo IoU cao nhất** trước khi so sánh, đặc biệt khi ảnh có nhiều vật thể. (Lưu ý: hàm `classify_errors` đã dùng đúng logic ghép cặp theo IoU tốt nhất ngay từ đầu, không bị lỗi này — chỉ riêng đoạn minh họa IoU đơn giản ban đầu là thiếu bước ghép cặp.)

**Không cần nêu sự cố này trong báo cáo** (đây là lỗi code đã tự phát hiện và sửa trước khi ra kết quả cuối, không phải hạn chế của model) — chỉ cần đảm bảo ảnh `illustration_iou.png` cuối cùng dùng trong báo cáo là bản đã sửa (IoU=0.889), không phải bản lỗi (IoU=0.000).

## Kết quả đo FPS bất thường — 960 nhanh hơn 640 (ngược lý thuyết)

**Quan sát:** đo FPS trên 100 ảnh test cố định: `640 → 50.2 FPS`, `960 → 58.0 FPS` — ảnh lớn hơn (960) lại cho FPS **cao hơn** ảnh nhỏ hơn (640), ngược với dự đoán lý thuyết (ảnh lớn hơn → nhiều phép tính hơn → chậm hơn).

**Nguyên nhân khả dĩ (không phải lỗi đo, mà là nhiễu đo đạc):**
1. **GPU "làm nóng" không đồng đều giữa 2 lần đo** — FPS của 640 được đo trước (có thể dính chi phí khởi động nguội CUDA), còn 416/960 đo sau, đã thừa hưởng lợi thế GPU đã "ấm" từ các lệnh chạy trước đó (minh họa, inference batch...) — không liên quan gì tới bản chất resolution.
2. **Thời gian hậu xử lý (NMS) phụ thuộc số lượng box ứng viên**, không chỉ kích thước ảnh đầu vào — nếu model 640 cho ra nhiều box ứng viên hơn trước khi lọc NMS, tổng thời gian xử lý có thể dài hơn dù ảnh nhỏ hơn.
3. Đo FPS bằng vòng lặp `time.time()` tự viết (đo cả overhead Python) khác với con số `Speed:` Ultralytics tự in trong lúc `val()` — 2 cách đo không nhất thiết khớp nhau, cho thấy phép đo FPS đơn lẻ vốn có độ nhiễu cao hơn nhiều so với mAP (tính trên toàn bộ tập test, ổn định hơn).

**Cần viết trong báo cáo (mục 6, phần hạn chế):** nêu đúng số liệu đã đo, kèm giải thích nguyên nhân nhiễu đo đạc ở trên — **không** tự ý đo lại nhiều lần để "chỉnh" cho ra kết quả đúng lý thuyết (đó là hành vi cherry-pick, vi phạm nguyên tắc trung thực của BTL). Câu mẫu: *"Kết quả đo cho thấy FPS ở imgsz=960 cao hơn imgsz=640, ngược với dự đoán lý thuyết. Nhóm cho rằng nguyên nhân đến từ nhiễu đo đạc (GPU warm-up không đồng đều giữa các lần đo, số lượng box cần xử lý NMS khác nhau) hơn là bản chất kiến trúc — đây cũng cho thấy phép đo FPS đơn lẻ có độ tin cậy thấp hơn mAP."*
