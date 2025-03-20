from pathlib import Path
import openvino as ov
import numpy as np
import time

# 設定模型
MODEL_DIRECTORY_PATH = Path("model")
model_name = "resnet50"
precision = "FP16"
device = "NPU"

# 設定模型路徑
model_path = MODEL_DIRECTORY_PATH / "ir_model" / f"{model_name}_{precision.lower()}.xml"
# model_path = Path("openvino_model.xml")
print(f"載入模型: {model_path}")

weights_path = model_path.with_suffix('.bin')

# 加載 OpenVINO Core
core = ov.Core()

# 檢查是否可以使用 NPU
if device not in core.available_devices:
    print(f"Device {device} is not available.")
else:
    print(f"Using device {device}")

# 加載模型
start_time = time.time()
# 讀取模型
model = core.read_model(model=model_path)
# 編譯模型，選擇 NPU 作為運行設備
compiled_model = core.compile_model(model, device)
print(f"Model compiled in {time.time() - start_time:.2f}s")

# 模型推理
input_layer = compiled_model.input(0)  # 取得模型的輸入層
output_layer = compiled_model.output(0)  # 取得模型的輸出層

# 生成一些隨機的輸入數據（假設模型的輸入形狀為 [1, 3, 224, 224]）
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 開始推理
start_time = time.time()
results = compiled_model([input_data])[output_layer]
print(f"Inference time: {time.time() - start_time:.2f}s")

# 顯示輸出結果
print("Output:", results)
