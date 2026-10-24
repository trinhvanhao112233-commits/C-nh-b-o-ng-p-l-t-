# Hướng dẫn quy trình huấn luyện RT-DETR trên GPU NVIDIA RTX 3060 Ti

Thư mục này chứa đầy đủ code và cấu trúc chuẩn bị huấn luyện mô hình **RT-DETR (Ultralytics)** để nhận diện ngập lụt.

---

## Cấu trúc thư mục

```text
flood_training/
├── flood_dataset/
│   ├── data.yaml            # Cấu hình dataset (đường dẫn và nhãn 0, 1, 2)
│   ├── images/
│   │   ├── train/           # Đặt ảnh dùng để Train vào đây
│   │   └── val/             # Đặt ảnh dùng để Đánh giá (Val) vào đây
│   └── labels/
│       ├── train/           # Đặt file nhãn .txt tương ứng với images/train
│       └── val/             # Đặt file nhãn .txt tương ứng với images/val
├── train.py                 # Script huấn luyện tự động với RTX 3060 Ti
├── export.py                # Script xuất sang ONNX và tự copy sang server
└── README.md
```

---

## 4 Bước thực hiện:

### Bước 1: Bỏ ảnh và nhãn vào `flood_dataset`
1. Copy ảnh của bạn vào `flood_dataset/images/train/` và `flood_dataset/images/val/`.
2. Copy các file nhãn `.txt` tương ứng vào `flood_dataset/labels/train/` và `flood_dataset/labels/val/`.
- Định dạng nhãn chuẩn YOLO: `<class_id> <x_center> <y_center> <width> <height>` (tất cả giá trị từ 0.0 đến 1.0).
  - Class `0`: `duong_ngap`
  - Class `1`: `xe_ngap`
  - Class `2`: `nguoi_loi_nuoc`

### Bước 2: Kiểm tra cấu hình `data.yaml`
Mở file `flood_dataset/data.yaml` để kiểm tra hoặc thêm bớt các class nếu bạn muốn.

### Bước 3: Chạy huấn luyện (Training)
Chạy lệnh sau tại thư mục này:
```bash
python train.py
```
- Script đã cấu hình tối ưu sẵn cho **RTX 3060 Ti (8GB VRAM)**: Batch size = 8, Image size = 640x640, Epochs = 100.
- Kết quả và file trọng số tốt nhất sẽ tự động lưu tại `runs_flood/rtdetr_flood_exp/weights/best.pt`.

### Bước 4: Export sang ONNX
Sau khi train xong, bạn chỉ cần gõ:
```bash
python export.py
```
- Lệnh này sẽ tối ưu hóa ONNX graph (`simplify=True`, `opset=16`) và **tự động copy file `rt_detr_flood.onnx` sang thư mục máy chủ `flood_detector_server/`** để chạy real-time!
