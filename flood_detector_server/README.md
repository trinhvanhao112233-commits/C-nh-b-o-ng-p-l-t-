# Máy chủ suy luận nhận diện ngập lụt theo thời gian thực (RT-DETR + FastAPI + WebSockets)

Dự án cung cấp Inference Server hiệu năng cao phục vụ nhận diện ngập lụt (đường ngập, xe ngập,...) từ camera/video stream của ứng dụng client (Vue.js) gửi lên qua WebSocket.

## Cấu trúc thư mục

```text
flood_detector_server/
├── main.py                  # Mã nguồn server chính (FastAPI + WebSocket + ONNXRuntime)
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── rt_detr_flood.onnx       # File trọng số mô hình ONNX (bạn copy vào đây)
└── README.md                # Hướng dẫn chạy & triển khai
```

## 1. Cài đặt môi trường

Mở Terminal tại thư mục `flood_detector_server`:

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt trên Windows:
.\venv\Scripts\activate

# Cài đặt thư viện:
pip install -r requirements.txt
```

> **Lưu ý GPU:** Nếu máy chủ hoặc máy tính của bạn có card đồ họa rời NVIDIA và muốn tăng tốc tối đa, cài đặt bản GPU:
> ```bash
> pip uninstall onnxruntime -y
> pip install onnxruntime-gpu
> ```

## 2. Chuẩn bị Model

Đặt file mô hình **`rt_detr_flood.onnx`** vào thẳng trong thư mục `flood_detector_server/`.  
Hoặc thiết lập biến môi trường nếu mô hình ở nơi khác:
```bash
set MODEL_PATH="D:\models\rt_detr_flood.onnx"
```

## 3. Khởi chạy Server

```bash
python main.py
```
Hoặc dùng uvicorn trực tiếp:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- Endpoint WebSocket: `ws://localhost:8000/ws/flood-detection`
- Health check: `http://localhost:8000/health`
