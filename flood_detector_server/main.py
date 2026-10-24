import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & HYPERPARAMETERS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "rt_detr_flood.onnx")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.45

# Danh sách nhãn đối tượng (mapping theo thứ tự output của mô hình đã train)
CLASS_NAMES = [
    "duong_ngap",    # 0: Đường ngập
    "xe_ngap",       # 1: Xe bị ngập nước
    "nguoi_loi_nuoc" # 2: Người lội nước (mở rộng)
]

# Bộ màu hiển thị gợi ý (Hex) để client vẽ Bounding Box
CLASS_COLORS = {
    0: "#3B82F6",  # Xanh dương cho đường ngập
    1: "#EF4444",  # Đỏ cảnh báo cho xe ngập
    2: "#F59E0B"   # Cam cho người lội nước
}


# ==============================================================================
# CLASS ĐIỀU PHỐI SUY LUẬN (RT-DETR INFERENCE ENGINE)
# ==============================================================================
class RTDETRInferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.requires_orig_target_sizes: bool = False

    def load_model(self):
        """Khởi tạo ONNXRuntime InferenceSession với tối ưu hóa luồng và provider."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Không tìm thấy file mô hình ONNX tại: {self.model_path}. "
                f"Vui lòng đặt file 'rt_detr_flood.onnx' vào thư mục server hoặc cấu hình biến môi trường MODEL_PATH."
            )

        # Ưu tiên GPU (CUDA/TensorRT) nếu có, fallback về CPU
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE"
                }
            ))
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)

        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)

        # Trích xuất metadata input/output
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # RT-DETR chuẩn của Paddle/Ultralytics có thể cần 2 inputs: 'images' & 'orig_target_sizes'
        self.requires_orig_target_sizes = len(self.input_names) > 1

        print(f"✅ ONNX Model loaded successfully.")
        print(f"   Providers: {self.session.get_providers()}")
        print(f"   Inputs: {self.input_names} | Outputs: {self.output_names}")

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, int, int]]:
        """
        Tiền xử lý ảnh trực tiếp trong RAM (Zero-disk I/O):
        - Letterbox resize giữ nguyên tỷ lệ khung hình (Aspect Ratio), tránh méo đối tượng.
        - Chuyển màu BGR -> RGB.
        - Chuẩn hóa [0, 255] -> [0.0, 1.0] dạng Float32.
        - HWC -> CHW và thêm chiều Batch Dimension (1, C, H, W).
        """
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Tính toán Letterbox Resize
        scale = min(INPUT_WIDTH / orig_w, INPUT_HEIGHT / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w = (INPUT_WIDTH - new_w) // 2
        pad_h = (INPUT_HEIGHT - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Canvas 640x640 với giá trị padding = 114 (chuẩn phổ biến của DETR/YOLO)
        canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # 2. BGR -> RGB & Chuẩn hóa sang float32 [0.0, 1.0]
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0

        # 3. HWC -> CHW -> NCHW: (1, 3, 640, 640)
        blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None, ...])

        return blob, (orig_w, orig_h), (scale, scale, pad_w, pad_h)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_size: Tuple[int, int],
        pad_info: Tuple[float, float, int, int]
    ) -> List[Dict]:
        """
        Hậu xử lý đầu ra của RT-DETR:
        - Tự động thích ứng cả 2 dạng ONNX export phổ biến:
          + Dạng tách: boxes (1, 300, 4) và scores (1, 300, num_classes)
          + Dạng gộp: (1, 300, 6) gồm [x1, y1, x2, y2, score, class_id]
        - Lọc ngưỡng điểm tin cậy (Confidence Threshold).
        - Ánh xạ tọa độ về khung hình gốc (loại trừ padding & scale).
        """
        orig_w, orig_h = orig_size
        scale_x, scale_y, pad_w, pad_h = pad_info
        detections = []

        if len(outputs) == 2:
            # Case 1: Tách rời boxes và scores/logits
            boxes, scores = outputs[0][0], outputs[1][0]  # (300, 4), (300, num_classes)
            class_ids = np.argmax(scores, axis=-1)
            confidences = np.max(scores, axis=-1)

            # Lọc bằng boolean mask nhanh không dùng vòng lặp
            valid_mask = confidences >= CONFIDENCE_THRESHOLD
            valid_boxes = boxes[valid_mask]
            valid_scores = confidences[valid_mask]
            valid_classes = class_ids[valid_mask]

        elif len(outputs) == 1:
            # Case 2: Output gộp (1, 300, 6)
            data = outputs[0][0]
            if data.shape[-1] >= 6:
                confidences = data[:, 4]
                valid_mask = confidences >= CONFIDENCE_THRESHOLD
                valid_data = data[valid_mask]
                valid_boxes = valid_data[:, :4]
                valid_scores = valid_data[:, 4]
                valid_classes = valid_data[:, 5].astype(int)
            else:
                return detections
        else:
            return detections

        for box, score, cls_id in zip(valid_boxes, valid_scores, valid_classes):
            x1, y1, x2, y2 = box

            # Ánh xạ ngược tọa độ từ ảnh 640x640 đã padding về khung hình gốc của Client
            real_x1 = max(0, int((x1 - pad_w) / scale_x))
            real_y1 = max(0, int((y1 - pad_h) / scale_y))
            real_x2 = min(orig_w, int((x2 - pad_w) / scale_x))
            real_y2 = min(orig_h, int((y2 - pad_h) / scale_y))

            cls_id_int = int(cls_id)
            label = CLASS_NAMES[cls_id_int] if cls_id_int < len(CLASS_NAMES) else f"class_{cls_id_int}"

            detections.append({
                "bbox": {
                    "x_min": real_x1,
                    "y_min": real_y1,
                    "x_max": real_x2,
                    "y_max": real_y2
                },
                "confidence": round(float(score), 4),
                "class_id": cls_id_int,
                "label": label,
                "color": CLASS_COLORS.get(cls_id_int, "#10B981")
            })

        return detections

    def predict(self, img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        t_start = time.perf_counter()

        blob, orig_size, pad_info = self.preprocess(img_bgr)

        # Xây dựng input feed
        inputs_dict = {self.input_names[0]: blob}
        if self.requires_orig_target_sizes:
            # Tensor (1, 2) chứa [H, W] gốc nếu graph mô hình yêu cầu
            inputs_dict[self.input_names[1]] = np.array([[orig_size[1], orig_size[0]]], dtype=np.int64)

        # Chạy inference ONNX
        outputs = self.session.run(self.output_names, inputs_dict)

        # Hậu xử lý trích xuất bounding boxes
        results = self.postprocess(outputs, orig_size, pad_info)
        inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return results, inference_time_ms


# ==============================================================================
# QUẢN LÝ VÒNG ĐỜI ỨNG DỤNG (LIFESPAN EVENT)
# ==============================================================================
engine: Optional[RTDETRInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🚀 Initializing AI Inference Service...")
    try:
        engine = RTDETRInferenceEngine(model_path=MODEL_PATH)
        engine.load_model()
    except FileNotFoundError as fnf_err:
        print(f"⚠️ CẢNH BÁO: {fnf_err}")
        print("ℹ️ Server vẫn đang chạy. Hãy đặt file .onnx vào đúng đường dẫn để sẵn sàng suy luận.")
    yield
    print("🛑 Releasing resources and shutting down...")
    del engine


# ==============================================================================
# FASTAPI APPLICATION & ROUTING
# ==============================================================================
app = FastAPI(
    title="Real-Time Flood Detection Server (RT-DETR)",
    version="1.0.0",
    lifespan=lifespan
)

# Kích hoạt CORS hỗ trợ client Vue.js chạy ở host/port khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image_data(message: Union[str, bytes]) -> Optional[np.ndarray]:
    """
    Giải mã khung hình dạng Binary hoặc Base64 thành NumPy Array (BGR) trực tiếp trong RAM.
    """
    try:
        if isinstance(message, bytes):
            # 1. Định dạng Binary (Buffer trực tiếp từ canvas / blob)
            img_buf = np.frombuffer(message, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            return img

        elif isinstance(message, str):
            # 2. Định dạng Base64 (DataURL ví dụ: 'data:image/jpeg;base64,...')
            if "," in message:
                message = message.split(",", 1)[1]
            img_bytes = base64.b64decode(message)
            img_buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"⚠️ Image decode error: {e}")
    return None


@app.websocket("/ws/flood-detection")
async def websocket_flood_detection(websocket: WebSocket):
    """
    Endpoint WebSocket đón nhận frame liên tục từ Vue.js client,
    suy luận real-time và trả kết quả Bounding Box, FPS, latency.
    """
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    print(f"🔌 Client connected: {client_info}")

    try:
        while True:
            # Nhận message từ client (hỗ trợ binary hoặc text base64)
            data = await websocket.receive()

            raw_frame_data = None
            if "bytes" in data and data["bytes"]:
                raw_frame_data = data["bytes"]
            elif "text" in data and data["text"]:
                raw_frame_data = data["text"]

            if raw_frame_data is None:
                continue

            # Kiểm tra trạng thái mô hình
            if engine is None or engine.session is None:
                await websocket.send_json({
                    "status": "error",
                    "message": "Model not loaded yet. Please place 'rt_detr_flood.onnx' in the server directory."
                })
                continue

            # 1. Giải mã frame sang ma trận ảnh OpenCV
            frame = decode_image_data(raw_frame_data)
            if frame is None:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid or corrupted image frame payload."
                })
                continue

            # 2. Thực thi suy luận với RT-DETR
            detections, latency_ms = engine.predict(frame)
            fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0

            # 3. Phản hồi JSON ngay lập tức cho client
            response_payload = {
                "status": "success",
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "fps": fps,
                "detections": detections,
                "metadata": {
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                    "objects_count": len(detections)
                }
            }
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client_info}")
    except Exception as exc:
        print(f"❌ WebSocket Session Exception: {exc}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
def health_check():
    """Endpoint kiểm tra sức khỏe của dịch vụ suy luận."""
    is_ready = engine is not None and engine.session is not None
    return {
        "status": "online",
        "model_loaded": is_ready,
        "providers": engine.session.get_providers() if is_ready else []
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False  # Tắt access log console để tránh nghẽn I/O khi frame rate cao
    )
