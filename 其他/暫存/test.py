import tflite_runtime.interpreter as tflite
import numpy as np

# 加載模型並啟用 NPU 委託
delegate_path = "/usr/lib/libvx_delegate.so"  # NPU 委託的路徑
interpreter = tflite.Interpreter(
    model_path="mnist_cnn_model_quantized.tflite",
    experimental_delegates=[tflite.load_delegate(delegate_path)]
)

interpreter.allocate_tensors()

# 準備測試資料
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 進行推理
input_data = np.expand_dims(test_feature[0], axis=0).astype(np.int8)  # int8 轉換
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 輸出結果
output_data = interpreter.get_tensor(output_details[0]['index'])
print("預測結果:", output_data)
