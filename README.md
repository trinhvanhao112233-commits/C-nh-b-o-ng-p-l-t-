# 🌊 Hệ Thống Cảnh Báo Ngập Lụt Thông Minh Thời Gian Thực (RT-DETR + FastAPI + WebSockets)

Dự án cung cấp giải pháp toàn diện cho bài toán **Nhận diện và Cảnh báo ngập lụt theo thời gian thực (Real-time Flood Detection)**, bao gồm:
1. **Pipeline huấn luyện (Training Pipeline)**: Huấn luyện mô hình thị giác tiên tiến **RT-DETR (Real-Time DEtection TRansformer)** tối ưu hóa tăng tốc trên GPU NVIDIA RTX.
2. **Máy chủ suy luận (Inference Server)**: Backend FastAPI hiệu năng cao, kết nối thời gian thực qua **WebSockets**, suy luận ảnh trực tiếp trong bộ nhớ RAM qua **ONNXRuntime** đạt FPS cao.

---

## 📁 Cấu trúc dự án

```text
├── flood_detector_server/         # Máy chủ suy luận thời gian thực (FastAPI & ONNX)
│   ├── main.py                    # Server FastAPI, WebSocket endpoint & ONNX Engine
│   ├── requirements.txt           # Danh sách thư viện phụ thuộc của server
│   ├── rt_detr_flood.onnx         # File mô hình ONNX (được export sau khi train)
│   └── README.md                  # Hướng dẫn chi tiết chạy máy chủ
│
├── flood_training/                # Bộ công cụ huấn luyện mô hình với PyTorch & GPU
│   ├── flood_dataset/
│   │   ├── data.yaml              # Cấu hình dataset và các nhãn class
│   │   ├── images/ (train, val)   # Thư mục chứa ảnh huấn luyện & validation
│   │   └── labels/ (train, val)   # Thư mục chứa file nhãn .txt (chuẩn YOLO)
│   ├── train.py                   # Script tự động huấn luyện RT-DETR trên GPU
│   ├── export.py                  # Script chuyển đổi .pt sang .onnx và copy sang server
│   └── README.md                  # Hướng dẫn huấn luyện & nạp dữ liệu
│
├── .gitignore                     # Cấu hình loại trừ file nặng, checkpoint, cache
└── README.md                      # Tài liệu tổng quan dự án
```

---

## 🏷️ Các nhãn nhận diện (Classes)

Hệ thống được thiết kế nhận diện 3 đối tượng trọng tâm trong đô thị ngập nước:
- **`0: duong_ngap`**: Vùng mặt đường bị ngập nước.
- **`1: xe_ngap`**: Ô tô, xe máy ngập hoặc chết máy trong nước ngập.
- **`2: nguoi_loi_nuoc`**: Người đi bộ / lội qua vùng nước sâu nguy hiểm.

---

## 🚀 Hướng dẫn nhanh (Quick Start)

### Phần 1: Huấn luyện mô hình (Training)

1. **Chuẩn bị dữ liệu**:
   - Copy ảnh vào: `flood_training/flood_dataset/images/train/` và `images/val/`.
   - Copy file nhãn `.txt` tương ứng vào: `flood_training/flood_dataset/labels/train/` và `labels/val/`.

2. **Chạy huấn luyện với GPU RTX 3060 Ti**:
   ```bash
   cd flood_training
   python train.py
   ```
   *(Mô hình mặc định là `rtdetr-l.pt`, batch size = 8, image size = 640x640).*

3. **Xuất mô hình sang ONNX**:
   ```bash
   python export.py
   ```
   Script sẽ tự động tối ưu hóa đồ thị ONNX (`simplify=True`, `opset=16`) và tự động chép file `rt_detr_flood.onnx` sang thư mục `flood_detector_server/`.

---

### Phần 2: Chạy Máy Chủ Suy Luận (Inference Server)

1. **Cài đặt thư viện**:
   ```bash
   cd flood_detector_server
   pip install -r requirements.txt
   ```

2. **Khởi động Server**:
   ```bash
   python main.py
   ```
   Server sẽ lắng nghe tại cổng `http://0.0.0.0:8000`.

3. **Giao tiếp WebSocket Real-Time**:
   - **Endpoint WebSocket**: `ws://localhost:8000/ws/flood-detection`
   - **Health check**: `http://localhost:8000/health`
   - Hỗ trợ client (Vue.js / React / Mobile) gửi frame dạng **Binary (Blob/ArrayBuffer)** hoặc **Base64**.
   - Định dạng dữ liệu server phản hồi:
     ```json
     {
       "status": "success",
       "timestamp": 1729783920.12,
       "latency_ms": 28.5,
       "fps": 35.1,
       "detections": [
         {
           "bbox": { "x_min": 120, "y_min": 240, "x_max": 480, "y_max": 590 },
           "confidence": 0.8921,
           "class_id": 1,
           "label": "xe_ngap",
           "color": "#EF4444"
         }
       ],
       "metadata": {
         "frame_width": 1280,
         "frame_height": 720,
         "objects_count": 1
       }
     }
     ```

---

## 🛠️ Công nghệ sử dụng
- **AI Core**: [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/), PyTorch with CUDA 12.4.
- **Inference Runtime**: Microsoft ONNXRuntime (CPU / CUDA Execution Provider).
- **Backend**: FastAPI, Uvicorn (ASGI), WebSockets.
- **Image Processing**: OpenCV (headless), NumPy (Zero-Disk I/O).
