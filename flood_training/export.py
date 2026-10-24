import os
import shutil
from ultralytics import RTDETR

def run_export():
    print("=" * 60)
    print("🔄 BẮT ĐẦU EXPORT RT-DETR SANG ĐỊNH DẠNG ONNX HIỆU NĂNG CAO")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_pt_path = os.path.join(current_dir, "runs_flood", "rtdetr_flood_exp", "weights", "best.pt")

    # Kiểm tra xem đã train xong có file best.pt chưa
    if not os.path.exists(best_pt_path):
        print(f"❌ Không tìm thấy file: {best_pt_path}")
        print("💡 Hãy chạy 'python train.py' trước để tạo file trọng số 'best.pt'.")
        # Fallback thử với pre-trained rtdetr-l.pt nếu người dùng muốn test pipeline export ngay
        test_model = "rtdetr-l.pt"
        if os.path.exists(test_model):
            print(f"⚠️ Thử nghiệm export với file pretrained gốc: {test_model}...")
            best_pt_path = test_model
        else:
            return

    print(f"📦 Đang nạp mô hình: {best_pt_path}")
    model = RTDETR(best_pt_path)

    # Export sang ONNX với thiết lập tối ưu cho ONNXRuntime
    print("⚙️ Đang thực hiện export ONNX (opset=16, simplify=True)...")
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,   # Rút gọn đồ thị tính toán với onnxslim
        opset=16,        # Chuẩn opset tương thích cao
        batch=1          # Inference real-time camera từng frame (batch=1)
    )

    print(f"✅ Export thành công: {exported_path}")

    # Tự động copy file ONNX sang thư mục server 'flood_detector_server'
    server_dir = os.path.join(os.path.dirname(current_dir), "flood_detector_server")
    target_onnx = os.path.join(server_dir, "rt_detr_flood.onnx")

    if os.path.exists(server_dir):
        print(f"\n🚀 Đang tự động sao chép sang máy chủ: {target_onnx}")
        shutil.copy2(exported_path, target_onnx)
        print("🎉 XONG! Máy chủ suy luận 'flood_detector_server' đã sẵn sàng chạy với mô hình mới!")
    else:
        print(f"⚠️ Không tìm thấy thư mục server {server_dir}. File onnx nằm tại: {exported_path}")

if __name__ == "__main__":
    run_export()
