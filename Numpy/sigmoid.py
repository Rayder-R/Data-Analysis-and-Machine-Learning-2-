import numpy as np

# 定義 Sigmoid 函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 計算不同 x 值的 Sigmoid
x_values = [-10, -5, 0, 5, 10]
sigmoid_values = [sigmoid(x) for x in x_values]

print(sigmoid_values)  # 輸出: [4.5397868702434395e-05, 0.0066928509242848554, 0.5, 0.9933071490757153, 0.9999546021312976]
