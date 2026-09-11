# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**
**Python / PyTorch / Ultralytics:**
Python: 3.13.15 / 
PyTorch: 2.11.0+cu128 / 
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K",
- Record này mô tả toàn ảnh như thế nào?
    Record này mô tả ảnh giao thông với record hạng 1 là "Cab", với score là 0.510915 có nghĩa là độ tin cậy 51% hoặc độ bao phủ của lớp là 51% và record hạng 2 là "minibus" với score là 0.164284 và 1 vài vật thể khác. Record này chỉ hiển thị 2 điểm chính là class name và score.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    Class list được định nghĩa bởi người gán nhãn hoặc hệ thống gán nhãn, class list này được lấy từ ImageNet-1K

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Cần giữ id vì để dễ dàng truy vấn và đối chiếu bằng máy tính, tên lớp để người đọc nhận biết bằng ngôn ngữ, tên taxonomy để mô hình biết đang sử dụng hệ thống phân loại nào, bộ dữ liệu nào.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

- Vì sao model score không phải ground truth?
    bản chất của score là xác suất, thể hiện độ tin tưởng cẩu models dựa trên các mẫu đã học,
    ground truth là do con người kiểm định dựa trên ngữ cảnh thực tế

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68

- Diễn giải vị trí box bằng lời:
Box này bao quanh người ở vị trí bên phải của ảnh, chiếm phần lớn vị trí của bên phải ảnh với kích thước box là 113.58x279.68 trong ảnh có kích thước 640x427

- So sánh số prediction ở hai threshold:

ở threshold=0.20 có 17 vật thể, threshold=0.60 có 6 vật thể, tăng threshold từ 0.2 lên 0.6, số lượng vật thể giảm đến 75%, vậy threshold càng cao, thì prediction càng giảm nhưng score sẽ cao.



- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
threshold thay đổi đối với độ bao phủ và khối lượng reviewer cần xem, threshold thấp độ bao phủ sẽ cao hơn và khôi lượng reviewer cần xem sẽ cao hơn và ngược lại.

- Đề xuất một quy tắc box chặt:
Box phải sát vào phần có thể nhìn thấy của object, chứa đầy đủ phần nhìn thấy của object, hạn chế backgruond dư thừa và không làm quá nhỏ làm mất phần object có thể nhìn thấy
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?


## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

"instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0...

- Polygon bổ sung chi tiết gì so với box?
Polygon mô tả đường biên của vật thể thay vì box chỉ cho biết hình chữ nhật bao quanh vật thể 
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id dùng thể định danh riêng 1 đối tượng cụ thể trong ảnh, nó không phải class_id, coco_image_id.

- Đề xuất một quy tắc biên mask:
 Mask phải bao phủ toàn bộ pixel có thể nhìn thấy của vật thể, bám sát đường biên của vật thể, hạn chế vùng nền nằm trong mask.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?



## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng và xử lý ảnh/dữ liệu đúng phạm vi được phân công; không tự ý sao chép, chia sẻ, tải lên dịch vụ bên ngoài hoặc sử dụng cho mục đích khác.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Reviewer hoặc người phụ trách dự án (Project Lead/Data Lead) để được hướng dẫn xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
