import numpy as np

# 模擬真實與預測資料
y_true = np.array([3.0, 2.5, 4.0])
y_pred = np.array([2.8, 3.0, 3.7])

# 手動計算 MSE
mse = np.mean((y_true - y_pred) ** 2)
print(f"MSE: {mse:.4f}")