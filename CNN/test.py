import openvino as ov
import numpy as np
from pathlib import Path

# OpenVINO 設定
core = ov.Core()
device = "NPU"
# model_path = Path("model/ir_model/cnn_model.xml")
model_path = Path("/cnn_model.xml")

# 載入與編譯
model = core.read_model(model_path)
compiled_model = core.compile_model(model, device)

# 模擬輸入一筆手寫數字資料
input_tensor = compiled_model.input(0)
input_data = np.random.rand(1, 28, 28, 1).astype(np.float32)

# 推理
result = compiled_model([input_data])[compiled_model.output(0)]
print("NPU 推理結果：", result)
