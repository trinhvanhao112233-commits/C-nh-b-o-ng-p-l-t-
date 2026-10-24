import os
import torch
from ultralytics import RTDETR

def run_training():
    print("=" * 60)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN RT-DETR VỚI GPU RTX 3060 Ti")
    print("=" * 60)

    # 1. Kiểm tra cấu hình phần cứng
    device_name = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"🔥 Tìm thấy GPU: {gpu_name} ({vram_gb} GB VRAM)")
        print(f"⚙️ Thiết bị được chọn: CUDA Device ID {device_name}")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy CUDA GPU, sẽ chạy trên CPU (chậm hơn).")

    # 2. Đường dẫn file cấu hình data.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, "flood_dataset", "data.yaml")

    # 3. Khởi tạo mô hình RT-DETR
    # 'rtdetr-l.pt' (RT-DETR Large) là phiên bản tối ưu hoàn hảo nhất cho RTX 3060 Ti (8GB VRAM)
    print("\n📦 Đang nạp pre-trained weights 'rtdetr-l.pt'...")
    model = RTDETR("rtdetr-l.pt")

    # 4. Huấn luyện (Training)
    print("\n🎯 Bắt đầu huấn luyện...")
    model.train(
        data=data_yaml_path,
        epochs=100,          # Số epoch (khuyến nghị 80 - 120 epochs cho bộ dữ liệu ngập)
        imgsz=640,           # Kích thước ảnh chuẩn của RT-DETR
        batch=8,             # RTX 3060 Ti (8GB VRAM) chạy mượt mà ở batch=8 (nếu thiếu VRAM giảm xuống 4)
        device=device_name,  # ID card đồ họa: 0
        workers=4,           # Đa luồng đọc dữ liệu
        project="runs_flood",# Thư mục lưu kết quả thí nghiệm
        name="rtdetr_flood_exp",
        exist_ok=True,
        save=True,           # Tự động lưu checkpoint và best.pt
        plots=True           # Xuất biểu đồ mAP, Loss
    )

    print("\n✅ HUẤN LUYỆN HOÀN TẤT!")
    print("Trọng số tốt nhất nằm tại: runs_flood/rtdetr_flood_exp/weights/best.pt")

if __name__ == "__main__":
    run_training()
